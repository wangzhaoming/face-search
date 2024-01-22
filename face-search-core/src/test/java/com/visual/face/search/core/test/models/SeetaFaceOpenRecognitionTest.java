package com.visual.face.search.core.test.models;

import com.visual.face.search.core.base.FaceAlignment;
import com.visual.face.search.core.base.FaceKeyPoint;
import com.visual.face.search.core.base.FaceRecognition;
import com.visual.face.search.core.domain.FaceInfo;
import com.visual.face.search.core.domain.ImageMat;
import com.visual.face.search.core.models.InsightCoordFaceKeyPoint;
import com.visual.face.search.core.models.SeetaFaceOpenRecognition;
import com.visual.face.search.core.models.Simple005pFaceAlignment;
import com.visual.face.search.core.test.base.BaseTest;
import com.visual.face.search.core.utils.CropUtil;
import com.visual.face.search.core.utils.Similarity;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class SeetaFaceOpenRecognitionTest extends BaseTest {
    private static String modelCoordPath = "face-search-core/src/main/resources/model/onnx/keypoint_coordinate/coordinate_106_mobilenet_05.onnx";
//    private static String modelSeetaPath = "face-search-core/src/main/resources/model/onnx/recognition_face_seeta/face_recognizer_512.onnx";
    private static String modelSeetaPath = "face-search-core/src/main/resources/model/onnx/recognition_face_seeta/face_recognizer_1024.onnx";

    private static String imagePath = "face-search-core/src/test/resources/images/faces";
//    private static String imagePath1 = "face-search-core/src/test/resources/images/faces/debug/debug_0001.jpg";
//    private static String imagePath2 = "face-search-core/src/test/resources/images/faces/debug/debug_0004.jpeg";
    private static String imagePath1 = "face-search-core/src/test/resources/images/faces/compare/1682052661610.jpg";
//    private static String imagePath2 = "face-search-core/src/test/resources/images/faces/compare/1682052669004.jpg";
    private static String imagePath2 = "face-search-core/src/test/resources/images/faces/compare/1682053163961.jpg";

    public static void main(String[] args) {
        Map<String, Object> params = new HashMap<>();
        params.put("model", "light1");
        FaceAlignment simple005pFaceAlignment = new Simple005pFaceAlignment();
        FaceKeyPoint insightCoordFaceKeyPoint = new InsightCoordFaceKeyPoint(modelCoordPath, 1);
        FaceRecognition insightSeetaFaceRecognition = new SeetaFaceOpenRecognition(modelSeetaPath, 1);

        Mat image1 = Imgcodecs.imread(imagePath1);
//        Mat image2 = Imgcodecs.imread(imagePath2);
//        image1 = CropUtil.crop(image1, FaceInfo.FaceBox.build(54,27,310,380));
//        image2 = CropUtil.crop(image2, FaceInfo.FaceBox.build(48,13,292,333));
//        image2 = CropUtil.crop(image2, FaceInfo.FaceBox.build(52,9,235,263));

//        simple005pFaceAlignment.inference()

        FaceInfo.Embedding embedding1 = insightSeetaFaceRecognition.inference(ImageMat.fromCVMat(image1), params);
        System.out.println(Arrays.toString(embedding1.embeds));
//        FaceInfo.Embedding embedding2 = insightSeetaFaceRecognition.inference(ImageMat.fromCVMat(image2), params);
//        float similarity = Similarity.cosineSimilarity(embedding1.embeds, embedding2.embeds);
//        System.out.println(similarity);
//        System.out.println(Arrays.toString(embedding1.embeds));
//        System.out.println(Arrays.toString(embedding2.embeds));
    }
}
