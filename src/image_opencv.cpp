#ifdef OPENCV

#include "stdio.h"
#include "stdlib.h"
#include "opencv2/opencv.hpp"
#include "image.h"

using namespace cv;

extern "C" {

struct mat_cv : cv::Mat { int a[0]; };
struct cap_cv : cv::VideoCapture { int a[0]; };
struct write_cv : cv::VideoWriter { int a[0]; };

IplImage *image_to_ipl(image im)
{
    int x,y,c;
    IplImage *disp = cvCreateImage(cvSize(im.w,im.h), IPL_DEPTH_8U, im.c);
    int step = disp->widthStep;
    for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            for(c= 0; c < im.c; ++c){
                float val = im.data[c*im.h*im.w + y*im.w + x];
                disp->imageData[y*step + x*im.c + c] = (unsigned char)(val*255);
            }
        }
    }
    return disp;
}

image ipl_to_image(IplImage* src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)src->imageData;
    int step = src->widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
    return im;
}

Mat image_to_mat(image im)
{
    image copy = copy_image(im);
    constrain_image(copy);
    if(im.c == 3) rgbgr_image(copy);

    IplImage *ipl = image_to_ipl(copy);
    Mat m = cvarrToMat(ipl, true);
    cvReleaseImage(&ipl);
    free_image(copy);
    return m;
}

image mat_to_image(Mat m)
{
    IplImage ipl = m;
    image im = ipl_to_image(&ipl);
    rgbgr_image(im);
    return im;
}

void *open_video_stream(const char *f, int c, int w, int h, int fps)
{
    VideoCapture *cap;
    if(f) cap = new VideoCapture(f);
    else cap = new VideoCapture(c);
    if(!cap->isOpened()) return 0;
    if(w) cap->set(CV_CAP_PROP_FRAME_WIDTH, w);
    if(h) cap->set(CV_CAP_PROP_FRAME_HEIGHT, w);
    if(fps) cap->set(CV_CAP_PROP_FPS, w);
    return (void *) cap;
}

image get_image_from_stream(void *p)
{
    VideoCapture *cap = (VideoCapture *)p;
    Mat m;
    *cap >> m;
    if(m.empty()) return make_empty_image(0,0,0);
    return mat_to_image(m);
}

image load_image_cv(char *filename, int channels)
{
    int flag = -1;
    if (channels == 0) flag = -1;
    else if (channels == 1) flag = 0;
    else if (channels == 3) flag = 1;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }
    Mat m;
    m = imread(filename, flag);
    if(!m.data){
        fprintf(stderr, "Cannot load image \"%s\"\n", filename);
        char buff[256];
        sprintf(buff, "echo %s >> bad.list", filename);
        system(buff);
        return make_image(10,10,3);
        //exit(0);
    }
    image im = mat_to_image(m);
    return im;
}

int show_image_cv(image im, const char* name, int ms)
{
    Mat m = image_to_mat(im);
    imshow(name, m);
    int c = waitKey(ms);
    if (c != -1) c = c%256;
    return c;
}

void make_window(char *name, int w, int h, int fullscreen)
{
    namedWindow(name, WINDOW_NORMAL); 
    if (fullscreen) {
        setWindowProperty(name, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    } else {
        resizeWindow(name, w, h);
        if(strcmp(name, "Demo") == 0) moveWindow(name, 0, 0);
    }
}

// ====================================================================
// Image Saving
// ====================================================================
extern int stbi_write_png(char const *filename, int w, int h, int comp, const void  *data, int stride_in_bytes);
extern int stbi_write_jpg(char const *filename, int x, int y, int comp, const void  *data, int quality);

void save_mat_png(cv::Mat img_src, const char *name)
{
    cv::Mat img_rgb;
    if (img_src.channels() >= 3) cv::cvtColor(img_src, img_rgb, cv::COLOR_RGB2BGR);
    stbi_write_png(name, img_rgb.cols, img_rgb.rows, 3, (char *)img_rgb.data, 0);
}
// ----------------------------------------

void save_mat_jpg(cv::Mat img_src, const char *name)
{
    cv::Mat img_rgb;
    if (img_src.channels() >= 3) cv::cvtColor(img_src, img_rgb, cv::COLOR_RGB2BGR);
    stbi_write_jpg(name, img_rgb.cols, img_rgb.rows, 3, (char *)img_rgb.data, 80);
}
// ----------------------------------------


void save_cv_png(mat_cv *img_src, const char *name)
{
    cv::Mat* img = (cv::Mat* )img_src;
    save_mat_png(*img, name);
}
// ----------------------------------------

void save_cv_jpg(mat_cv *img_src, const char *name)
{
    cv::Mat* img = (cv::Mat*)img_src;
    save_mat_jpg(*img, name);
}
// ----------------------------------------

// ====================================================================
// Show Anchors
// ====================================================================
void show_acnhors(int number_of_boxes, int num_of_clusters, float *rel_width_height_array, model anchors_data, int width, int height)
{
    cv::Mat labels = cv::Mat(number_of_boxes, 1, CV_32SC1);
    cv::Mat points = cv::Mat(number_of_boxes, 2, CV_32FC1);
    cv::Mat centers = cv::Mat(num_of_clusters, 2, CV_32FC1);

    for (int i = 0; i < number_of_boxes; ++i) {
        points.at<float>(i, 0) = rel_width_height_array[i * 2];
        points.at<float>(i, 1) = rel_width_height_array[i * 2 + 1];
    }

    for (int i = 0; i < num_of_clusters; ++i) {
        centers.at<float>(i, 0) = anchors_data.centers.vals[i][0];
        centers.at<float>(i, 1) = anchors_data.centers.vals[i][1];
    }

    for (int i = 0; i < number_of_boxes; ++i) {
        labels.at<int>(i, 0) = anchors_data.assignments[i];
    }

    size_t img_size = 700;
    cv::Mat img = cv::Mat(img_size, img_size, CV_8UC3);

    for (int i = 0; i < number_of_boxes; ++i) {
        cv::Point pt;
        pt.x = points.at<float>(i, 0) * img_size / width;
        pt.y = points.at<float>(i, 1) * img_size / height;
        int cluster_idx = labels.at<int>(i, 0);
        int red_id = (cluster_idx * (uint64_t)123 + 55) % 255;
        int green_id = (cluster_idx * (uint64_t)321 + 33) % 255;
        int blue_id = (cluster_idx * (uint64_t)11 + 99) % 255;
        cv::circle(img, pt, 1, CV_RGB(red_id, green_id, blue_id), CV_FILLED, 8, 0);
        //if(pt.x > img_size || pt.y > img_size) printf("\n pt.x = %d, pt.y = %d \n", pt.x, pt.y);
    }

    for (int j = 0; j < num_of_clusters; ++j) {
        cv::Point pt1, pt2;
        pt1.x = pt1.y = 0;
        pt2.x = centers.at<float>(j, 0) * img_size / width;
        pt2.y = centers.at<float>(j, 1) * img_size / height;
        cv::rectangle(img, pt1, pt2, CV_RGB(255, 255, 255), 1, 8, 0);
    }
    save_mat_png(img, "cloud.png");
    cv::imshow("clusters", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

}

#endif
