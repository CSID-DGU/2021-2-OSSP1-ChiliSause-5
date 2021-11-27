import numpy as np
import cv2 as cv

# 미리 처리된 클래스, 몬스터 및 복사점 선택에 사용
class PreProcessing:
    def __init__(self):
        # seamlessClone 호출에 필요한 두 멤버
        self.selectedMask = None    # 그레이스케일 행렬
                                    # 선택 영역은 255, 비 선택 영역은 0, 크기는 src와 같으며 경계는 포함되지 않습니다
        self.selectedPoint = None   # 선택한 복사 위치
                                    # (비고: 사용자 정의 seamlessClone에만 적용됩니다. opencv의 seamlessClone, x, y를 실행하려면 반전이 필요합니다)
        
        # 중간 변수, 개인
        self.__edgeList = []        # 선택 경계점 목록
        self.__preview = None       # 그림
        self.__edgeMat = None       # 사용자 그려진 선 그레이 스케일
        self.__reMask = None        # 마스크를 대상 그림 크기로 확장하십시오
        self.__reImg = None         # 대상 이미지 크기로 이미지 확장을 선택하십시오
        self.__prePoint = None      # 이전 선택
        self.__minPoint = None      # 테두리 컨트롤의 최소 유효한 위치
        self.__maxPoint = None      # 경계 제어를위한 최대 유효 위치
    # 선을 그리려면 왼쪽 클릭, 내부 영역을 지정하려면 오른쪽 클릭, 선택을 취소하려면 가운데 클릭:구글
    # 왼쪽 버튼 케이블을 드래그하고, 마우스 오른쪽 버튼을 클릭하여 내부 영역을 지정하고, 중간 단추 선택:확장
    # 왼쪽 버튼 드래그 애니메이션 라인, 오른쪽 버튼 내부 영역 지정, 가운데 버튼 제거 선택:파파고
    def __onMouseAction1(self, event, x, y, flags, param):
        # 시작점을 선택하십시오
        if event == cv.EVENT_LBUTTONDOWN:
            self.__edgeList = [(x, y)]
            self.__edgeMat = np.zeros(
                (param.shape[0], param.shape[1]), np.uint8)
            self.__preview = np.copy(param)
        # 선을 그리다
        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
            self.__edgeMat = cv.line(
                self.__edgeMat, self.__edgeList[-1], (x, y), 255)
            self.__preview = cv.line(
                self.__preview, self.__edgeList[-1], (x, y), (0, 0, 255))
            self.__edgeList.append((x, y))
            cv.imshow('select_mask', self.__preview)
        # 선택 영역 결정
        elif event == cv.EVENT_RBUTTONDOWN:
            retval, image, mask, rect = cv.floodFill(
                np.copy(self.__edgeMat), None, (x, y), 255)
            self.selectedMask = image - self.__edgeMat
            selectedImg = cv.copyTo(param, self.selectedMask)
            cv.imshow('select_mask', selectedImg)
            print(self.__edgeMat)
            type(self.__edgeMat)
        # 선택 영역 비우기
        elif event == cv.EVENT_MBUTTONDOWN:
            self.__edgeList = []
            self.__edgeMat = np.zeros(
                (param.shape[0], param.shape[1]), np.uint8)
            self.__preview = np.copy(param)
            cv.imshow('select_mask', self.__preview)

    # 왼쪽 버튼 드래그 조정 대상 위치
    def __onMouseAction2(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.__prePoint = (x, y)
        # 拖拽选区
        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
            # 마우스 포인트 X는 컬럼을 나타내고, y는 선을 나타내며 매트릭스 프로세스는 변환입니다.
            dx, dy = x-self.__prePoint[0], y-self.__prePoint[1]
            # 교차 처리
            if self.selectedPoint[0] + dy < self.__minPoint[0] or self.selectedPoint[0] + dy > self.__maxPoint[0]:
                dy = 0
            if self.selectedPoint[1] + dx < self.__minPoint[1] or self.selectedPoint[1] + dx > self.__maxPoint[1]:
                dx = 0
            self.__prePoint = (self.__prePoint[0]+dx, self.__prePoint[1]+dy)
            self.selectedPoint = (
                self.selectedPoint[0]+dy, self.selectedPoint[1]+dx)
            # 이동하다
            self.__reMask = cv.warpAffine(self.__reMask, np.array(
                [[1, 0, dx], [0, 1, dy]], dtype=np.float64), (self.__reMask.shape[1], self.__reMask.shape[0]))
            self.__reImg = cv.warpAffine(self.__reImg, np.array(
                [[1, 0, dx], [0, 1, dy]], dtype=np.float64), (self.__reImg.shape[1], self.__reImg.shape[0]))
            _img = np.copy(param)
            _img[self.__reMask != 0] = 0
            _img = _img + self.__reImg
            cv.imshow('select_point', _img)

    # 마스크 선택 및 위치 복사
    # 마스크가 선택되면 마우스 왼쪽 버튼을 클릭하여 드래그하여 선을 그리고 마우스 오른쪽 버튼을 클릭하여 내부 영역을 지정하고 마우스 가운데 버튼을 클릭하여 선택을 취소합니다.
    # 복사 위치 선택 시 왼쪽 버튼을 드래그하여 대상 위치를 조정합니다.
    # 처음으로 아무 키나 눌러 마스크 선택을 확인하고 아무 키나 두 번 눌러 복사 위치를 확인하고 종료합니다.
    def select(self, src, dst):
        # 마스크 선택
        cv.namedWindow('select_mask')
        cv.setMouseCallback('select_mask', lambda event, x, y, flags,
                            param: self.__onMouseAction1(event, x, y, flags, param), src)
        cv.imshow('select_mask', src)
        cv.waitKey(0)
        cv.destroyAllWindows()
        # 결과 처리

        pl = np.nonzero(self.selectedMask)  #
        selectedSrc = cv.copyTo(src, self.selectedMask)
        cutMask = self.selectedMask[np.min(pl[0])-1:np.max(pl[0])+2, np.min(pl[1])-1:np.max(pl[1])+2]
        type(cutMask)
        cutSrc = selectedSrc[np.min(pl[0])-1:np.max(pl[0])+2, np.min(pl[1])-1:np.max(pl[1])+2]
        self.__minPoint = (
            (np.max(pl[0])-np.min(pl[0]))//2+1, (np.max(pl[1])-np.min(pl[1]))//2+1)
        # 선택 영역이 너무 크다
        if dst.shape[0] < cutMask.shape[0] or dst.shape[1] < cutMask.shape[1]:
            raise UserWarning
        # 복사 위치 선택
        cv.namedWindow('select_point')
        cv.setMouseCallback('select_point', lambda event, x, y, flags,
                            param: self.__onMouseAction2(event, x, y, flags, param), dst)
        # 초기화 첫 번째 라운드 디스플레이
        self.__reMask = np.zeros((dst.shape[0], dst.shape[1]))
        self.__reImg = np.zeros_like(dst)
        self.__reMask[:cutMask.shape[0],
                      :cutMask.shape[1]] = cutMask
        self.__reImg[:cutSrc.shape[0],
                     :cutSrc.shape[1]] = cutSrc
        self.selectedPoint = self.__minPoint[:]
        self.__maxPoint = (dst.shape[0]-cutMask.shape[0]+self.selectedPoint[0],
                           dst.shape[1]-cutMask.shape[1]+self.selectedPoint[1])
        _dst = np.copy(dst)
        _dst[self.__reMask != 0] = 0
        _dst = _dst + self.__reImg
        cv.imshow('select_point', _dst)
        cv.waitKey(0)
        cv.destroyAllWindows()
        print(self.selectedMask.shape)
        print(type(self.selectedPoint))
        return self.selectedMask, self.selectedPoint

    # 소스 이미지 영역 만 선택하십시오
    def selectSrc(self, src):
        # 마스크를 선택하십시오
        cv.namedWindow('select_mask')
        cv.setMouseCallback('select_mask', lambda event, x, y, flags,
                            param: self.__onMouseAction1(event, x, y, flags, param), src)
        cv.imshow('select_mask', src)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return self.selectedMask

    def selectMask(self, src, dst, mask):
            # 마스크 선택

            # 결과 처리
            self.selectedMask = mask
            pl = np.nonzero(self.selectedMask)  #
            selectedSrc = cv.copyTo(src, self.selectedMask)
            cutMask = self.selectedMask[np.min(pl[0])-1:np.max(pl[0])+2, np.min(pl[1])-1:np.max(pl[1])+2]
            type(cutMask)
            cutSrc = selectedSrc[np.min(pl[0])-1:np.max(pl[0])+2, np.min(pl[1])-1:np.max(pl[1])+2]
            self.__minPoint = (
                (np.max(pl[0])-np.min(pl[0]))//2+1, (np.max(pl[1])-np.min(pl[1]))//2+1)
            # 선택 영역이 너무 크다
            if dst.shape[0] < cutMask.shape[0] or dst.shape[1] < cutMask.shape[1]:
                raise UserWarning
            # 복사 위치 선택

            return self.selectedMask, self.selectedPoint
