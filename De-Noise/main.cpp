/*
group 3:
Nitesh Kumar
Ravi Kumar Upadhyay
Prateek Kishore
Ankit Jaiswal
*/


#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <time.h>

using namespace std;
using namespace cv;

#define max(a,b) a>b?a:b
#define min(a,b) a>b?b:a

//structure for storing pixel values
typedef struct 
{
	unsigned char b;
	unsigned char g;
	unsigned char r;
}Pixel;


//constants
int CHOICE = 1;
int K = 1024;
int S = 3;
int M = 3;
int M2 = 1;
float P = 0.04f;
float TH1 = 0.90f;
float TH2 = 0.87f;
float TH3 = 0.97f;

float arr[100][2] ;

//to obtain pixel value of xth row and yth col of image
Pixel* getPixel(Mat &Image, Pixel *pix, int x, int y)
{
	int channels = Image.channels();
	pix->b = Image.data[Image.step*x + channels*y + 0];
	pix->g = Image.data[Image.step*x + channels*y + 1];
	pix->r = Image.data[Image.step*x + channels*y + 2];
	return pix;
}


//to put pixel pix at xth row and yth col of image
void putPixel(Mat &Image, Pixel *pix, int x, int y)
{
	int channels = Image.channels();
	Image.data[Image.step*x + channels*y + 0] = pix->b;
	Image.data[Image.step*x + channels*y + 1] = pix->g;
	Image.data[Image.step*x + channels*y + 2] = pix->r;
}


//generating a random pixel
void randomPixel(Pixel *pix)
{
	pix->r = rand()%256;
	pix->g = rand()%256;
	pix->b = rand()%256;
}

//generating a random number between 0 to 100
int numGen()
{
	unsigned int n = rand()%100;
	return (int)n;
}

//introducing noise in the image based on percentage values
void putNoise(Mat &Image, int percent)
{
	int i, j;
	Pixel pix1, pix2;
	pix1.r = pix1.g = pix1.b = 0;
	pix2.r = pix2.g = pix2.b = 255;

	for(i=0; i<Image.rows; i++)
	{
		for(j=0; j<Image.cols; j++)
		{
			if(numGen()<=percent)
			{
				if(rand()%2==0)
					putPixel(Image, &pix1, i, j);
				else 
					putPixel(Image, &pix2, i, j);
			}
		}
	}
}



//displaying pixel values
void displayPixel(Pixel *pix)
{
	printf("\n\npix.r=%d\npix.g=%d\npix.b=%d\n",pix->r,pix->g,pix->b);
}

// sorts the values in arr[8][2] in increasing to decreasing order
void insertionSort(int size)
{
	int i,j;
	float key, keyindex;
	for (i=1; i<size; i++) {
		key = arr[i][0];
		keyindex = arr[i][1];
		j = i-1;
		while ( (j >= 0) && (arr[j][0] < key) ) {
			arr[j+1][0] = arr[j][0];
			arr[j+1][1] = arr[j][1];
			j=j-1;
		}
		arr[j+1][0] = key;
		arr[j+1][1] = keyindex;
	}
}



//very same -> one, very different -> zero
float pixelSimilarity(Pixel *pix1, Pixel *pix2)
{
	float a, b, c;
	
	a = ( (float)min(pix1->r, pix2->r) + K ) / (float)( max(pix1->r, pix2->r) + K );
	b = ( (float)min(pix1->g, pix2->g) + K ) / (float)( max(pix1->g, pix2->g) + K );
	c = ( (float)min(pix1->b, pix2->b) + K ) / (float)( max(pix1->b, pix2->b) + K );

	a = min (a, b);
	a = min (a, c);
	return a;
}

//printing the array containing 's' values
void printArr()
{
	for(int i=0; i<S*S-1; i++){
		printf("\tArr[%d][0] = %f\n", i,  arr[i][0]);
	}
}

//calculating the pixel similarity of each pixel in a kernel w.r.t the central pixel('s' values) and sorting them
void initROM(Mat &Image, int x, int y) 
{
	Pixel pix1, pix2;
	int i, temp=0;
	
	//getting pixel value of the central element
	getPixel(Image, &pix1, x, y);

	x = x - (S-1)/2;
	y = y - (S-1)/2;

	//calculating the pixel similarity
	for ( i=0; i<S*S; i++ ){
		if(i== (S*S - 1)/2 ) continue;
		getPixel(Image, &pix2, x+(i/S), y+(i%S));
		arr[temp][0] = pixelSimilarity(&pix1, &pix2);
		arr[temp][1] = (float)i;
		temp++;
	}
	insertionSort(S*S - 1);
}


//classifying pixels into noisy, normal or undiagnosed
void processWindow1( Mat &Image, int *check, int x, int y)
{
	int i, p = 0, j;
	float frod = 1;

	initROM(Image, x, y);
	for(i=0; i<M; i++)
		frod = frod*arr[i][0];
	p = (x-((S-1)/2))*Image.cols + (y-((S-1)/2));

	if(frod > (TH1 + (P*0.175)) ){
		check[(x)*Image.cols + (y)]=1;
		for(i=0; i<M; i++){
			j = (int)arr[i][1];
			check[p + (j/3)*Image.cols + j%3] = 1;
		}
	}

	else if(frod < (TH2 + (P*0.15)) )
		check[(x)*Image.cols + (y)] = -1;
}


//classifying the undiagnosed pixels using th3
void processWindow2( Mat &Image, int *check, int x, int y)
{
	int i, p, j, count = 0;
	float frod = 1;
	initROM(Image, x, y);

	p = (x-(S-1)/2)*Image.cols + (y-(S-1)/2);

	for(i=0; i<(S*S)-1 && count<M2; i++){
		j = (int)arr[i][1];
		if (check[p + (j/S)*Image.cols + (j%S)] == 1){
			frod *= arr[i][0];
			count++;
		}
	}

	if(frod > ( TH3+ (P*0.025) ) )
		check[(x)*Image.cols + (y)]=1;
	
	else
		check[(x)*Image.cols + (y)] = -1;
}

//reducing noise in image using gradient(pixel similarity approach)
void remNoise(Mat &Image, int x, int y)
{
	Pixel pix1, pix2, Pix1, Pix2;

	getPixel(Image, &pix1, x-1, y);
	getPixel(Image, &pix2, x+1, y);
	arr[0][0] = pixelSimilarity(&pix1, &pix2);
	arr[0][1] = 1;

	getPixel(Image, &pix1, x-1, y+1);
	getPixel(Image, &pix2, x+1, y-1);
	arr[1][0] = pixelSimilarity(&pix1, &pix2);
	arr[1][1] = 2;

	getPixel(Image, &pix1, x, y+1);
	getPixel(Image, &pix2, x, y-1);
	arr[2][0] = pixelSimilarity(&pix1, &pix2);
	arr[2][1] = 3;
	
	getPixel(Image, &pix1, x-1, y-1);
	getPixel(Image, &pix2, x+1, y+1);
	arr[3][0] = pixelSimilarity(&pix1, &pix2);
	arr[3][1] = 4;
	
	insertionSort(4);
	switch ((int)arr[0][1])
	{
	case 1:
			
			getPixel(Image, &pix1, x-1, y);
			getPixel(Image, &pix2, x+1, y);
			Pix1.r = (pix1.r)/2 + (pix2.r)/2;
			Pix1.g = (pix1.g)/2 + (pix2.g)/2;
			Pix1.b = (pix1.b)/2 + (pix2.b)/2;
			break;

	case 2:
			getPixel(Image, &pix1, x-1, y+1);
			getPixel(Image, &pix2, x+1, y-1);
			Pix1.r = (pix1.r)/2 + (pix2.r)/2;
			Pix1.g = (pix1.g)/2 + (pix2.g)/2;
			Pix1.b = (pix1.b)/2 + (pix2.b)/2;
			break;

	case 3:
			getPixel(Image, &pix1, x, y-1);
			getPixel(Image, &pix2, x, y+1);
			Pix1.r = (pix1.r)/2 + (pix2.r)/2;
			Pix1.g = (pix1.g)/2 + (pix2.g)/2;
			Pix1.b = (pix1.b)/2 + (pix2.b)/2;
			break;

	case 4:
			getPixel(Image, &pix1, x-1, y-1);
			getPixel(Image, &pix2, x+1, y+1);
			Pix1.r = (pix1.r)/2 + (pix2.r)/2;
			Pix1.g = (pix1.g)/2 + (pix2.g)/2;
			Pix1.b = (pix1.b)/2 + (pix2.b)/2;
			break;
	}

	switch ((int)arr[1][1])
	{
	case 1:
			getPixel(Image, &pix1, x-1, y);
			getPixel(Image, &pix2, x+1, y);
			Pix2.r = (pix1.r)/2 + (pix2.r)/2;
			Pix2.g = (pix1.g)/2 + (pix2.g)/2;
			Pix2.b = (pix1.b)/2 + (pix2.b)/2;
			break;

	case 2:
			getPixel(Image, &pix1, x-1, y+1);
			getPixel(Image, &pix2, x+1, y-1);
			Pix2.r = (pix1.r)/2 + (pix2.r)/2;
			Pix2.g = (pix1.g)/2 + (pix2.g)/2;
			Pix2.b = (pix1.b)/2 + (pix2.b)/2;
			break;

	case 3:
			getPixel(Image, &pix1, x, y-1);
			getPixel(Image, &pix2, x, y+1);
			Pix2.r = (pix1.r)/2 + (pix2.r)/2;
			Pix2.g = (pix1.g)/2 + (pix2.g)/2;
			Pix2.b = (pix1.b)/2 + (pix2.b)/2;
			break;

	case 4:
			getPixel(Image, &pix1, x-1, y-1);
			getPixel(Image, &pix2, x+1, y+1);
			Pix2.r = (pix1.r)/2 + (pix2.r)/2;
			Pix2.g = (pix1.g)/2 + (pix2.g)/2;
			Pix2.b = (pix1.b)/2 + (pix2.b)/2;
			break;
	}
	switch(CHOICE)
	{
	case 1:
		if(pixelSimilarity(&Pix1, &Pix2)>0.89){
			pix1.r=(Pix2.r)/2+(Pix1.r)/2;
			pix1.g=(Pix2.g)/2+(Pix1.g)/2;
			pix1.b=(Pix2.b)/2+(Pix1.b)/2;
			putPixel(Image, &pix1, x, y);
		}
		else{
			putPixel(Image, &Pix1, x, y);
		}
		break;

	case 2:
		putPixel(Image, &Pix1, x, y);
		break;
	}
}




//checking the conditions
void sanityCheck(Mat &Image)
{
	if(Image.data == NULL){
		printf("No image loaded. Check if the image exists\n");
		getchar();
		exit(0);
	}

	if(M>S*S-1 || M2>S*S-1){
		printf("M or M2 greater than elements in kernel\n");
		getchar();
		exit(0);
	}

	if(M2>=M) 
		printf("warning: the value of M2 is greater than M\n");

	if(P>1 || P<0){
		printf("Probability only lies between 1 and 0\n");
		getchar();
		exit(0);
	}

	float limitingth = (float)(K)/(float)(K+256);
	if(TH1<limitingth || TH2<limitingth || TH3<limitingth){
		printf("Threshold values cannot be less than %f", limitingth);
		getchar();
		exit(0);
	}

	if(TH1>1 || TH2>1 || TH3>1){
		printf("Threshold values cannot be more than 1\n");
		getchar();
		exit(0);
	}

	if(S%2!=1){
		printf("S should be odd\n");
		getchar();
		exit(0);
	}
}

//showing multiple images in the same window
void showManyImages(Mat &src1, Mat &src2)
{
	Mat dispImage, temp1, temp2;

	int x =  2*src1.cols+45;
	int y =	 src1.rows+30;

	dispImage = cvCreateImage(cvSize(x,y),8,3);

	temp1 = Mat(dispImage, Range(15, y-15), Range(15, (x-15)/2));
	temp2 = Mat(dispImage, Range(15, y-15), Range(src1.cols+30, x-15));
	
	src1.copyTo(temp1);
	src2.copyTo(temp2);

	imwrite("output_comparison.jpg", dispImage);
	namedWindow("result.jpg",0);
	cvResizeWindow("result.jpg", x, y);
	imshow("result.jpg",dispImage);
	waitKey(0);
}


int main()
{
	int i, j, *check, count=0;
	Mat A,B;
	
	A= imread("test11.bmp", 1);
	sanityCheck(A);
	printf("Processing Image : test.jpg\n\tHeight = %d\n\tWidth  = %d\n", A.cols, A. rows);

	B = A.clone();

	check = (int *) malloc( A.rows * A.cols * sizeof(int) );
	/*  initialising check array with 0  */
	for ( i=0; i < (A.rows*A.cols); i++ )
		check[i] = 0;

	/*  for (> th1) and (< th2)  */
	for (i=(S-1)/2; i<A.rows-(S-1)/2; i++) {
		for (j=(S-1)/2; j<A.cols-(S-1)/2; j++) {
			if ( check[(i*A.cols)+j] != 0) //  (-1 -> noisy) (1 -> non noisy) (0 -> un-diagnosed)
				continue;
			processWindow1(A, check, i, j);
			if(check[i*A.cols + j]==-1) count++;
		}
	}

	for (i=(S-1)/2; i<A.rows-(S-1)/2; i++) {
		for (j=(S-1)/2; j<A.cols-(S-1)/2; j++) {
			if ( check[(i*A.cols)+j] == 0 ){ //  (-1 -> noisy) (1 -> non noisy) (0 -> un-diagnosed)
				processWindow2(A, check, i, j);
				if(check[i*A.cols + j]==-1) count++;
			}
		}
	}

	for (i=1;i<A.rows-1;i++) 
		for (j=1;j<A.cols-1;j++) 
			if (check[i*A.cols+j] == -1)
				remNoise(A, i, j);
	
	
	imwrite("output.jpg", A);
	printf("Original Noise Count - %d\n", count);
	//printf("percentage noise = %f\n", (float)count/(float)(A.rows*A.cols));
	showManyImages(B,A);
	
	return 0;
}
