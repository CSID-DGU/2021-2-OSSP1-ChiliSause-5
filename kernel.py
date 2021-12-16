import numpy as np
import cv2 as cv
from scipy.sparse import lil_matrix, linalg


class Poisson:
    NORMAL_CLONE = 1
    MIXED_CLONE = 2
    # 사용자 인터페이스
    # src와 dst의 크기가 같을 필요는 없고, 채널 수는 같아야 하고, mask는 src 마스크에 해당하고, point는 dst에 해당하는 복사 위치의 중심점입니다.
    @classmethod
    def seamlessClone(cls, src, dst, mask, point, flag=NORMAL_CLONE):
        laplacian = cv.Laplacian(np.float64(src), -1, ksize=1)
        inner = np.nonzero(mask)
        # 암묵적으로 지정한 위치 찾기
        xbegin, xend, ybegin, yend = np.min(inner[0])-1, np.max(inner[0]) + 2,\
            np.min(inner[1])-1, np.max(inner[1])+2

        if(xbegin<1):
            xbegin = 1
        if(ybegin<1):
            ybegin = 1
        cutMask = mask[xbegin:xend, ybegin:yend]
        cutSrc = src[xbegin:xend, ybegin:yend]
        cutLap = laplacian[xbegin:xend, ybegin:yend]
        # 이동 가능한 최소, 최대 위치
        minPoint = ((np.max(inner[0])-np.min(inner[0]))//2+1,
                    (np.max(inner[1])-np.min(inner[1]))//2+1)
        maxPoint = (dst.shape[0]-cutSrc.shape[0]+minPoint[0],
                    dst.shape[1]-cutSrc.shape[1]+minPoint[1])
        # 선택 영역이 너무 큽니다
        if cutSrc.shape[0] > dst.shape[0] or cutSrc.shape[0] > dst.shape[0]:
            raise UserWarning
        # 경계를 넘다
        if point[0] < minPoint[0] or point[1] < minPoint[1] or point[0] > maxPoint[0] or point[1] > maxPoint[1]:
            raise UserWarning
        # 해당 위치로 복사
        reMask = np.zeros((dst.shape[0], dst.shape[1]))
        reSrc = np.zeros_like(dst)
        reLap = np.zeros_like(dst, dtype=np.float64)
        xbegin, xend, ybegin, yend = \
            int(point[0]-minPoint[0]), int(cutMask.shape[0]+point[0]-minPoint[0]), \
            int(point[1]-minPoint[1]), int(cutMask.shape[1]+point[1]-minPoint[1])
        reMask[xbegin:xend, ybegin:yend] = cutMask
        reSrc[xbegin:xend, ybegin:yend] = cutSrc
        reLap[xbegin:xend, ybegin:yend] = cutLap
        if flag == cls.MIXED_CLONE:
            kernels = [np.array([[0, -1, 1]]), np.array([[1, -1, 0]]),
                       np.array([[0], [-1], [1]]), np.array([[1], [-1], [0]])]
            grads = [(cv.filter2D(np.float64(reSrc), -1, kernels[i]),
                      cv.filter2D(np.float64(dst), -1, kernels[i])) for i in range(4)]
            grads = [np.where(np.abs(srcGrad) > np.abs(
                dstGrad), srcGrad, dstGrad) for (srcGrad, dstGrad) in grads]
            reLap = np.sum(grads, axis=0)
        # 채널별 솔루션
        ret = [cls._solve(s, d, reMask, l) for s, d, l in zip(
            cv.split(reSrc), cv.split(dst), cv.split(reLap))]
        retImg = cv.merge(ret)
        return retImg

    @classmethod
    def textureFlattening(cls, src, mask, low_thresh, high_thresh):
        kernels = [np.array([[0, -1, 1]]), np.array([[1, -1, 0]]),
                    np.array([[0], [-1], [1]]), np.array([[1], [-1], [0]])]
        kernelsOfEdge = [np.array([[0, 1, 1]]), np.array([[1, 1, 0]]),
                    np.array([[0], [1], [1]]), np.array([[1], [1], [0]])]
        canny = cv.Canny(src, low_thresh, high_thresh)
        edges = [cv.filter2D(canny, -1, kernelsOfEdge[i]) for i in range(4)]
        grads = [cv.filter2D(np.float64(src), -1, kernels[i]) for i in range(4)]
        for i in range(4):
            grads[i][edges[i] == 0] = 0
        laplacian = np.sum(grads, axis=0)
        ret = [cls._solve(s, d, mask, l) for s, d, l in zip(
            cv.split(src), cv.split(src), cv.split(laplacian))]
        retImg = cv.merge(ret)
        return retImg

    @classmethod
    def illuminationChange(cls, src, mask, alpha=0.2, beta=0.4):
        laplacian = cv.Laplacian(np.float64(src), -1, ksize=1)
        laplacian[mask == 0] = 0
        laplacian = laplacian * (alpha**beta * np.log(np.linalg.norm(laplacian))**(-beta))
        ret = [cls._solve(s, d, mask, l) for s, d, l in zip(
            cv.split(src), cv.split(src), cv.split(laplacian))]
        retImg = cv.merge(ret)
        return retImg

    #요청 형식은 RGB입니다
    @classmethod
    def colorChange(cls, src, mask, red_mul, green_mul, blue_mul):
        r, g, b = cv.split(src)
        newSrc = cv.merge((r*red_mul, g*green_mul, b*blue_mul))
        laplacian = cv.Laplacian(np.float64(newSrc), -1, ksize=1)
        ret = [cls._solve(s, d, mask, l) for s, d, l in zip(
            cv.split(newSrc), cv.split(src), cv.split(laplacian))]
        retImg = cv.merge(ret)
        return retImg

    @classmethod
    def deColor(cls, src, mask):
        laplacian = cv.Laplacian(np.float64(src), -1, ksize=1)
        newSrc = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        newSrc = cv.cvtColor(newSrc, cv.COLOR_GRAY2BGR)
        ret = [cls._solve(s, d, mask, l) for s, d, l in zip(
            cv.split(src), cv.split(newSrc), cv.split(laplacian))]
        retImg = cv.merge(ret)
        return retImg

    # scipy.sparse 희소 행렬 LU 분해를 통해 선형 방정식 풀기
    # 이 내부 기능을 사용하려면 src, dst, mask가 사전 처리되어야 하고, uint8 storage, src, dst가 동일한 크기를 가져야 하고, 단일 채널이 있어야 하며, mask는 복사 위치입니다.
    @classmethod
    def _solve(cls, src, dst, mask, laplacian):
        inner = np.nonzero(mask)
        size = inner[0].shape[0]
        mmap = {(x, y): i for i, (x, y) in enumerate(zip(inner[0], inner[1]))}
        dx, dy = [1, 0, -1, 0], [0, 1, 0, -1]
        A, b = lil_matrix((size, size), dtype=np.float64), np.ndarray(
            (size, ), dtype=np.float64)
        # 계수 행렬 구성
        for i, (x, y) in enumerate(zip(inner[0], inner[1])):
            A[i, i] = -4
            b[i] = laplacian[x, y]
            p = [(x+dx[j], y+dy[j]) for j in range(4)]
            for j in range(4):
                if p[j] in mmap:
                    A[i, mmap[p[j]]] = 1
                else:
                    b[i] -= dst[p[j]]
        A = A.tocsc()
        LU = linalg.splu(A)
        X = LU.solve(b)
        ret = np.copy(dst)
        for i, (x, y) in enumerate(zip(inner[0], inner[1])):
            ret[x, y] = min((255, max((0, X[i]))))
        return ret
