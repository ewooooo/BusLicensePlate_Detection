

#include <opencv2/opencv.hpp> 
#include <algorithm>// min() max();
#include <iostream>
#include <cstdlib>
#include <ctime>

#include <unistd.h> //sockaddr_in, read, write 등
#include <arpa/inet.h>  //htnol, htons, INADDR_ANY, sockaddr_in 등
#include <sys/socket.h>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

using namespace cv;
using namespace std;
using namespace tesseract;

bool debug = false;
Mat imageDebuger;


TessBaseAPI* ocr = new TessBaseAPI();
class BusNumber {
private:
	VideoCapture cap;
	Mat inputimage;
	double maxratio = 0.9, minratio = 0.25, alpha = 0, beta = 500;
	Mat beforProcess(Mat image);
	vector<vector<Rect>> oneCarNumber(vector<vector<Point>> contours);
	vector<Rect> doubleCarNumber(vector<vector<Rect>> resultGroupList);
	bool checkDim(Rect rect, Rect test);
	int count = 0;
	Rect beforeRect;
	int beforeNumber = 0;
	int startTime=0;
	int limitSec;
	int boundX;
	int boundY;

	String OCR(Mat test);
	void EraseSpace(char* inStr);
	String processNumber(String text);
public:
	BusNumber(int lsec, int bx, int by, String filename) {
		limitSec = lsec;
		boundX = bx;
		boundY = by;

		cap = VideoCapture(filename);
		//cap = VideoCapture(0);
		if (!cap.isOpened()) {
			cerr << "에러 - 카메라를 열 수 없습니다.\\\\n";
			exit;
		}

	}

	int BusNumberRectList(int control);
};
Mat BusNumber::beforProcess(Mat image) {

	cvtColor(image, image, COLOR_BGR2GRAY);  //  Convert to gray image.COLOR_BGR2GRAY

	Size userSize = Size(3, 3);
	Mat imgTopHat, imgBlackHat;
	morphologyEx(image, imgTopHat, MORPH_TOPHAT, getStructuringElement(MORPH_ELLIPSE, userSize));
	morphologyEx(image, imgBlackHat, MORPH_BLACKHAT, getStructuringElement(MORPH_ELLIPSE, userSize));

	Mat imgGrayscalePlusTopHat;
	add(image, imgTopHat, imgGrayscalePlusTopHat);
	Mat morphoImage;
	subtract(imgGrayscalePlusTopHat, imgBlackHat, image);

	GaussianBlur(image, image, Size(5, 5), 0);

	Canny(image, image, 100, 300, 3);  //  Getting edges by Canny algorithm.

	return image;
}

vector<vector<Rect>>  BusNumber::oneCarNumber(vector<vector<Point>> contours) {

	vector<Rect> rect_list(contours.size());

	int rectindex = 0;

	for (int idx = 0; idx < contours.size(); idx++) {
		Rect rect = boundingRect(contours[idx]);

		double ratio = (double)rect.width / rect.height;

		if ((ratio >= minratio) && (ratio <= maxratio) && (rect.height >= alpha) && (rect.height <= beta)) {
			Rect newrect(rect);
			rect_list[rectindex] = newrect;
			rectindex++;
		}

	}
	rect_list.resize(rectindex);  //  Resize refinery rectangle array.

	if (rect_list.empty()) {
		return {};
	}

	vector<Rect> opRectList;
	for (int idx = 0; idx < rect_list.size(); idx++) {

		for (int idx2 = 0; idx2 < rect_list.size(); idx2++) {
			if (rect_list[idx] == rect_list[idx2] || rect_list[idx].x >= rect_list[idx2].x)
				continue;

			double gap = rect_list[idx].x + rect_list[idx].width - rect_list[idx2].x;

			if (abs(gap) < max(rect_list[idx].height * 0.2, (double)10)) {

				//���������� �Ǵ�.
				if (gap > rect_list[idx].width * 0.15) {
					continue;
				}

				double diffup = abs(rect_list[idx].tl().y - rect_list[idx2].tl().y);
				double diffdn = abs(rect_list[idx].br().y - rect_list[idx2].br().y);

				if (diffup < max(rect_list[idx].height * 0.3, (double)10) && diffdn < max(rect_list[idx].height * 0.3, (double)10)) {
					double Rgap = rect_list[idx].br().x - rect_list[idx2].br().x;
					double Lgap = rect_list[idx2].x - rect_list[idx].x;
					double Tgap = rect_list[idx2].y - rect_list[idx].y;
					double Bgap = rect_list[idx].br().y - rect_list[idx2].br().y;

					if (!(Rgap >= 0 && Lgap >= 0 && Tgap >= 0 && Bgap >= 0)) {

						if (opRectList.empty()) {
							opRectList.push_back(rect_list[idx]);
							opRectList.push_back(rect_list[idx2]);
						}
						else {
							int test1 = 0, test2 = 0;
							for (int i = 0; i < opRectList.size(); i++) {
								if (test1 == 0 && opRectList[i] == rect_list[idx]) {
									test1 = 1;
								}
								if (test2 == 0 && opRectList[i] == rect_list[idx2]) {
									test2 = 1;
								}
								if (test1 == 1 && test2 == 1) {
									break;
								}
							}
							if (test1 == 0)
								opRectList.push_back(rect_list[idx]);
							if (test2 == 0)
								opRectList.push_back(rect_list[idx2]);
						}
					}
				}
			}
		}
	}

	if (opRectList.empty()) {
		return {};
	}

	for (int a = opRectList.size() - 1; a > 0; a--) {
		for (int j = 0; j < a; j++) {
			if (opRectList[j].tl().x > opRectList[j + 1].tl().x) {

				Rect temp_rect = opRectList[j];

				opRectList[j] = opRectList[j + 1];

				opRectList[j + 1] = temp_rect;

			}
		}
	}

	vector<vector<Rect>> resultGroupList;
	vector<Rect> fiartGroup;

	fiartGroup.push_back(opRectList[0]);
	resultGroupList.push_back(fiartGroup);

	for (int idx = 0; idx < opRectList.size(); idx++) {
		int test = 0;
		for (int i = 0; i < resultGroupList.size(); i++) {

			Rect testRect = resultGroupList[i].back();

			double gap = testRect.x + testRect.width - opRectList[idx].x;
			if (abs(gap) < max(testRect.height * 0.2, (double)10)) {

				//���������� �Ǵ�.
				if (gap > testRect.width * 0.15) {
					continue;
				}

				double diffup = abs(testRect.tl().y - opRectList[idx].tl().y);
				double diffdn = abs(testRect.br().y - opRectList[idx].br().y);

				if (diffup < max(testRect.height * 0.3, (double)10) && diffdn < max(testRect.height * 0.3, (double)10)) {

					if (!(testRect.br().x - opRectList[idx].br().x >= 0 && opRectList[idx].x - testRect.x >= 0 &&
						opRectList[idx].y - testRect.y >= 0 && testRect.br().y - opRectList[idx].br().y >= 0)) {

						resultGroupList[i].push_back(opRectList[idx]);

						test = 1;
					}
				}
			}
		}
		if (test == 0) {
			vector<Rect> newGroup;
			newGroup.push_back(opRectList[idx]);
			resultGroupList.push_back(newGroup);
		}
	}
	return resultGroupList;
}

vector<Rect> BusNumber::doubleCarNumber(vector<vector<Rect>> resultGroupList) {
	if (resultGroupList.empty()) {
		return {};
	}
	vector <Rect> GroupList;
	for (int i = 0; i < resultGroupList.size(); i++) {
		if (resultGroupList[i].size() < 4 || resultGroupList[i].size() > 12)
			continue;
		for (int j = 1; j < resultGroupList.size(); j++) {
			if (resultGroupList[j].size() < 4 || resultGroupList[j].size() > 12)
				continue;

			Rect upBoundingRect(Point(resultGroupList[i][0].tl().x, resultGroupList[i][0].tl().y < resultGroupList[i].back().tl().y ? resultGroupList[i][0].tl().y : resultGroupList[i].back().tl().y),
				Point(resultGroupList[i].back().br().x, resultGroupList[i][0].br().y > resultGroupList[i].back().br().y ? resultGroupList[i][0].br().y : resultGroupList[i].back().br().y));
			Rect downBoundingRect(Point(resultGroupList[j][0].tl().x, resultGroupList[j][0].tl().y < resultGroupList[j].back().tl().y ? resultGroupList[j][0].tl().y : resultGroupList[j].back().tl().y),
				Point(resultGroupList[j].back().br().x, resultGroupList[j][0].br().y > resultGroupList[j].back().br().y ? resultGroupList[j][0].br().y : resultGroupList[j].back().br().y));

			if (upBoundingRect.y > downBoundingRect.y) { //i�� ������ �Ѵ� �ƴϸ� ��
				continue;
			}
			double gap2 = upBoundingRect.br().y - downBoundingRect.tl().y;
			if (gap2 > upBoundingRect.height * 0.15) {
				continue;
			}
			double wg = (downBoundingRect.br().x - downBoundingRect.tl().x);
			double xgap = (upBoundingRect.x - downBoundingRect.x);
			int rectCount = resultGroupList[j].size();
			if (xgap >= (wg / rectCount) && xgap <= 3 * (wg / rectCount)) { //����(7)�̰� 
				double ht = upBoundingRect.height;
				double ygap = downBoundingRect.tl().y - upBoundingRect.br().y;
				if (ygap <= ht * 0.2) {//����(8)�̸� ���ٹ�ȣ��
					Rect boundingRect2(Point(upBoundingRect.x < downBoundingRect.x ? upBoundingRect.x : downBoundingRect.x,
						upBoundingRect.y < downBoundingRect.y ? upBoundingRect.y : downBoundingRect.y),
						Point(upBoundingRect.br().x < downBoundingRect.br().x ? downBoundingRect.br().x : upBoundingRect.br().x,
							upBoundingRect.br().y < downBoundingRect.br().y ? downBoundingRect.br().y : upBoundingRect.br().y));

					GroupList.push_back(boundingRect2);
				}
			}
		}
	}
	return GroupList;
}

String BusNumber::OCR(Mat image) {
	Mat test = image.clone();
	resize(test, test, Size(), 2, 2, INTER_NEAREST);
	string outText;
	imshow("ROI", test);
	ocr->Init(NULL, "eng", OEM_LSTM_ONLY);
	ocr->SetPageSegMode(PSM_AUTO);

	ocr->SetImage(test.data, test.cols, test.rows, 3, test.step);
	outText = string(ocr->GetUTF8Text());
	ocr->End();

	return outText;
}

void BusNumber::EraseSpace(char* inStr)
{
	char* p_dest = inStr; // p_dest �����͵� ap_string �����Ϳ� ������ �޸𸮸� ����Ų��.

	// ���ڿ��� ���� ���������� �ݺ��Ѵ�.
	while (*inStr != 0) {
		// ap_string�� ����Ű�� ���� ���� ���ڰ� �ƴ� ��츸
		// p_dest�� ����Ű�� �޸𸮿� ���� �����Ѵ�.
		if (*inStr != ' ') {
			if (p_dest != inStr) *p_dest = *inStr; // �Ϲ� ���ڸ� �����ϸ� ���� ������ ��ġ�� �̵��Ѵ�.
			p_dest++;
		}
		// ���� ���� ��ġ�� �̵��Ѵ�.
		inStr++;
	}
	// ���ڿ��� ���� NULL ���ڸ� �����Ѵ�.
	*p_dest = 0;
}


String BusNumber::processNumber(String text) {
	char inStr[100];
	char num[] = { '0','1','2','3','4','5','6','7','8','9' };
	strcpy(inStr, text.c_str());

	for (int i = 0; i < strlen(inStr); i++)
	{
		bool test = true;
		for (int j = 0; j < 10; j++) {
			if (inStr[i] == num[j]) {
				test = false;
			}
		}
		if (test) {

			inStr[i] = ' ';

		}

	}
	EraseSpace(inStr);
	return String(inStr);
}

bool BusNumber::checkDim(Rect rect, Rect test) {

	double distBeforAfterFrameX = abs((test.br().x - test.x) / 2) - abs((rect.br().x - rect.x) / 2);
	double distBeforAfterFrameY = abs((test.br().y - test.y) / 2) - abs((rect.br().y - rect.y) / 2);


	if (distBeforAfterFrameX < boundX && distBeforAfterFrameY < boundY) {
		return true;
	}
	return false;

}

int BusNumber::BusNumberRectList(int control) {
	cap.read(inputimage);
	if (inputimage.empty()) {
		cerr << "빈 영상이 캡쳐되었습니다.\\\\n";
		exit;
	}

	if (debug) {
		imageDebuger = inputimage.clone();
		imshow("imageDebuger",imageDebuger);
	}

	Mat processImage = inputimage.clone();


	processImage = beforProcess(processImage);

	vector<vector<Point> > contours;  //  Vectors for 'findContours' function.
	vector<Vec4i> hierarchy;
	findContours(processImage, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE, Point());
	if (contours.empty())
		return 0;
	vector<vector<Rect>> busRectList = oneCarNumber(contours);
	if (busRectList.empty())
		return 0;

	vector<Rect> GroupList = doubleCarNumber(busRectList);
	

    int resultNumber = 0;
	if (!GroupList.empty()) {

		for (int i = 0; i < GroupList.size(); i++) {
			if (debug) {
				rectangle(imageDebuger, GroupList[i].tl(), GroupList[i].br(), Scalar(0, 0, 255), 3);
			}
			
			Rect rectROI = Rect(Point(GroupList[i].tl().x - 25, GroupList[i].tl().y - 25), Point(GroupList[i].br().x + 25, GroupList[i].br().y + 25));
			Mat testimage = inputimage(rectROI);
			String testStr = OCR(testimage);
			String result = processNumber(testStr);
			if (result.length() == 4) {
				resultNumber = std::stoi(result.substr(result.length() - 4));
			}
			else {
				resultNumber = 0;
			}
		}
	}
	return resultNumber;

}

int main(int argc,char *argv[])
{
	BusNumber busTest = BusNumber(3,50,50,argv[1]);
	

	while (1) {
		try {
			cout << busTest.BusNumberRectList(1) << endl;

		}
		catch (int exception) {
			cout << "error" << endl;
		}
		if(debug)
			imshow("imagedebuger", imageDebuger);
		
		int key = waitKey(1);
		if (key > 0)
			break;
	}

	return 0;
}


