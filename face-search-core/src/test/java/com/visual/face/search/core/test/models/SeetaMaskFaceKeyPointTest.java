package com.visual.face.search.core.test.models;

import com.visual.face.search.core.domain.FaceInfo;
import com.visual.face.search.core.domain.ImageMat;
import com.visual.face.search.core.domain.QualityInfo;
import com.visual.face.search.core.models.InsightCoordFaceKeyPoint;
import com.visual.face.search.core.models.InsightScrfdFaceDetection;
import com.visual.face.search.core.models.SeetaMaskFaceKeyPoint;
import com.visual.face.search.core.test.base.BaseTest;
import com.visual.face.search.core.utils.CropUtil;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.List;
import java.util.Map;

public class SeetaMaskFaceKeyPointTest extends BaseTest {
    private static String modelDetectionPath = "face-search-core/src/main/resources/model/onnx/detection_face_scrfd/scrfd_500m_bnkps.onnx";
    private static String modelKeypointPath = "face-search-core/src/main/resources/model/onnx/keypoint_seeta_mask/landmarker_005_mask_pts5.onnx";
    private static String imagePath = "face-search-core/src/test/resources/images/faces";
//    private static String imagePath = "face-search-core/src/test/resources/images/faces/compare";
//    private static String imagePath = "face-search-core/src/test/resources/images/faces/compare/1694353163955.jpg";

    public static void main(String[] args) {
        Map<String, String> map = getImagePathMap(imagePath);
        InsightScrfdFaceDetection detectionInfer = new InsightScrfdFaceDetection(modelDetectionPath, 1);
        SeetaMaskFaceKeyPoint keyPointInfer = new SeetaMaskFaceKeyPoint(modelKeypointPath, 1);
        for(String fileName : map.keySet()) {
            System.out.println(fileName);
            String imageFilePath = map.get(fileName);
            Mat image = Imgcodecs.imread(imageFilePath);
            List<FaceInfo> faceInfos = detectionInfer.inference(ImageMat.fromCVMat(image), 0.5f, 0.7f, null);
            for(FaceInfo faceInfo : faceInfos){
                FaceInfo.FaceBox rotateFaceBox = faceInfo.rotateFaceBox();
                Mat cropFace = CropUtil.crop(image, rotateFaceBox.scaling(1.0f));
                ImageMat cropImageMat = ImageMat.fromCVMat(cropFace);
                QualityInfo.MaskPoints maskPoints = keyPointInfer.inference(cropImageMat, null);
                System.out.println(maskPoints);
                for(QualityInfo.MaskPoint maskPoint : maskPoints){
                    if(maskPoint.isMask()){
                        Imgproc.circle(cropFace, new Point(maskPoint.x, maskPoint.y), 3, new Scalar(0, 0, 255), -1);
                    }else{
                        Imgproc.circle(cropFace, new Point(maskPoint.x, maskPoint.y), 3, new Scalar(255, 0, 0), -1);
                    }
                }
                HighGui.imshow(fileName, cropFace);
                HighGui.waitKey();
            }
        }
        System.exit(1);
    }

}
