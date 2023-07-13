package com.visual.face.search.core.test.models;

import com.alibaba.fastjson.JSONObject;
import com.visual.face.search.core.domain.FaceInfo;
import com.visual.face.search.core.domain.ImageMat;
import com.visual.face.search.core.models.InsightScrfdFaceDetection;
import com.visual.face.search.core.test.base.BaseTest;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class InsightScrfdFaceDetectionTest extends BaseTest {
    private static String modelPath = "face-search-core/src/main/resources/model/onnx/detection_face_scrfd/scrfd_500m_bnkps.onnx";

//    private static String imagePath = "face-search-core/src/test/resources/images/faces";
    private static String imagePath = "face-search-core/src/test/resources/images/faces/rotate/rotate_0001.jpg";
//    private static String imagePath = "face-search-core/src/test/resources/images/faces/rotate";


    public static void main(String[] args) {
        Map<String, String> map = getImagePathMap(imagePath);
        InsightScrfdFaceDetection infer = new InsightScrfdFaceDetection(modelPath, 2);

        for(String fileName : map.keySet()){
            String imageFilePath = map.get(fileName);
            System.out.println(imageFilePath);
            Mat image = Imgcodecs.imread(imageFilePath);
            long s = System.currentTimeMillis();
            Map<String, Object> params = new JSONObject().fluentPut(InsightScrfdFaceDetection.scrfdFaceNeedCheckFaceAngleParamKey, true);

            List<FaceInfo> faceInfos = infer.inference(ImageMat.fromCVMat(image), 0.48f, 0.7f, params);
            long e = System.currentTimeMillis();
            if(faceInfos.size() > 0){
                System.out.println("fileName="+fileName+",\tcost="+(e-s)+",\t"+faceInfos.get(0).score);
            }else{
                System.out.println("fileName="+fileName+",\tcost="+(e-s)+",\t"+faceInfos);
            }

            //对坐标进行调整
            for(FaceInfo faceInfo : faceInfos){
                FaceInfo.FaceBox box  = faceInfo.rotateFaceBox();
                Imgproc.circle(image, new Point(box.leftTop.x, box.leftTop.y), 3, new Scalar(0,0,255), -1);
                Imgproc.circle(image, new Point(box.rightBottom.x, box.rightBottom.y), 3, new Scalar(0,0,255), -1);
                Imgproc.line(image, new Point(box.leftTop.x, box.leftTop.y), new Point(box.rightTop.x, box.rightTop.y), new Scalar(0,0,255), 1);
                Imgproc.line(image, new Point(box.rightTop.x, box.rightTop.y), new Point(box.rightBottom.x, box.rightBottom.y), new Scalar(255,0,0), 1);
                Imgproc.line(image, new Point(box.rightBottom.x, box.rightBottom.y), new Point(box.leftBottom.x, box.leftBottom.y), new Scalar(255,0,0), 1);
                Imgproc.line(image, new Point(box.leftBottom.x, box.leftBottom.y), new Point(box.leftTop.x, box.leftTop.y), new Scalar(255,0,0), 1);
                Imgproc.putText(image, String.valueOf(faceInfo.angle), new Point(box.leftTop.x, box.leftTop.y), Imgproc.FONT_HERSHEY_PLAIN, 1, new Scalar(0,0,255));

                FaceInfo.Points points = faceInfo.points;
                int pointNum = 1;
                for(FaceInfo.Point keyPoint : points){
                    Imgproc.circle(image, new Point(keyPoint.x, keyPoint.y), 3, new Scalar(0,0,255), -1);
                    Imgproc.putText(image, String.valueOf(pointNum), new Point(keyPoint.x+1, keyPoint.y), Imgproc.FONT_HERSHEY_PLAIN, 1, new Scalar(255,0,0));
                    pointNum ++ ;
                }
            }
            HighGui.imshow(fileName, image);
            HighGui.waitKey();
        }
        System.exit(1);
    }

}
