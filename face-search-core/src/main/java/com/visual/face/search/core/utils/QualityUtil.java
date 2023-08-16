package com.visual.face.search.core.utils;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

public class QualityUtil {

    public static int blurForLaplacian(Mat image){
        Mat imgDesc = null, imgResize = null;
        Mat img2gray = null, laplacian = null;
        MatOfDouble std = null, median = null;
        try {
            //裁剪图片的中心位置区域
            if(image.width() > image.height()){
                int offset = (image.width() - image.height()) / 2;
                imgDesc = new Mat(image.height(), image.height(), CvType.CV_8UC3);
                Mat imgROI = new Mat(image, new Rect(offset, 0, image.height(), image.height()));
                imgROI.copyTo(imgDesc);
            }else if(image.height() < image.width()){
                int offset = (image.height() - image.width()) / 2;
                imgDesc = new Mat(image.width(), image.width(), CvType.CV_8UC3);
                Mat imgROI = new Mat(image, new Rect(0, offset, image.height(), image.height()));
                imgROI.copyTo(imgDesc);
            }else{
                imgDesc = new Mat(image.width(), image.height(), CvType.CV_8UC3);
                image.copyTo(imgDesc);
            }
            //将图片都转换为112X112的分辨率
            imgResize = new Mat();
            Imgproc.resize(imgDesc, imgResize, new Size(112,112), 0, 0, Imgproc.INTER_AREA);
            //图像处理
            img2gray = new Mat();
            Imgproc.cvtColor(imgResize, img2gray, Imgproc.COLOR_BGR2GRAY);
            laplacian = new Mat();
            Imgproc.Laplacian(img2gray, laplacian, CvType.CV_64F);
            //获取模糊分
            std= new MatOfDouble();
            median = new MatOfDouble();
            Core.meanStdDev(laplacian, median , std);
            double var = Math.pow(std.get(0,0)[0], 2);
            //将模糊分设置到0-100的值域中
            var = Math.min(var, 1500);
            double score = Math.sin(var / 1500 * Math.PI * 0.5);
            int normal = Double.valueOf(score * 100).intValue();
            return Math.max(Math.min(normal, 100), 0);
        }catch (Exception e){
            throw new RuntimeException(e);
        }finally {
            if(null != std){
                std.release();
            }
            if(null != median){
                median.release();
            }
            if(null != laplacian){
                laplacian.release();
            }
            if(null != img2gray){
                img2gray.release();
            }
            if(null != imgResize){
                imgResize.release();
            }
            if(null != imgDesc){
                imgDesc.release();
            }
        }
    }

}
