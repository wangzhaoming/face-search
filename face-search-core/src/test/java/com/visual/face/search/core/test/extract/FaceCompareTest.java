package com.visual.face.search.core.test.extract;

import com.visual.face.search.core.base.*;
import com.visual.face.search.core.domain.ExtParam;
import com.visual.face.search.core.domain.FaceImage;
import com.visual.face.search.core.domain.FaceInfo;
import com.visual.face.search.core.domain.ImageMat;
import com.visual.face.search.core.extract.FaceFeatureExtractor;
import com.visual.face.search.core.extract.FaceFeatureExtractorImpl;
import com.visual.face.search.core.models.*;
import com.visual.face.search.core.test.base.BaseTest;
import com.visual.face.search.core.utils.Similarity;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class FaceCompareTest extends BaseTest {

    private static String modelPcn1Path = "face-search-core/src/main/resources/model/onnx/detection_face_pcn/pcn1_sd.onnx";
    private static String modelPcn2Path = "face-search-core/src/main/resources/model/onnx/detection_face_pcn/pcn2_sd.onnx";
    private static String modelPcn3Path = "face-search-core/src/main/resources/model/onnx/detection_face_pcn/pcn3_sd.onnx";
    private static String modelScrfdPath = "face-search-core/src/main/resources/model/onnx/detection_face_scrfd/scrfd_500m_bnkps.onnx";
    private static String modelCoordPath = "face-search-core/src/main/resources/model/onnx/keypoint_coordinate/coordinate_106_mobilenet_05.onnx";
    private static String modelArcPath = "face-search-core/src/main/resources/model/onnx/recognition_face_arc/glint360k_cosface_r18_fp16_0.1.onnx";
    private static String modelSeetaPath = "face-search-core/src/main/resources/model/onnx/recognition_face_seeta/face_recognizer_512.onnx";
    private static String modelArrPath = "face-search-core/src/main/resources/model/onnx/attribute_gender_age/insight_gender_age.onnx";

    private static String imagePath = "face-search-test/src/main/resources/image/validate/index/马化腾/";
    private static String imagePath3 = "face-search-test/src/main/resources/image/validate/index/雷军/";
//    private static String imagePath1 = "face-search-core/src/test/resources/images/faces/debug/debug_0001.jpg";
//    private static String imagePath2 = "face-search-core/src/test/resources/images/faces/debug/debug_0001.jpg";
//    private static String imagePath1 = "face-search-core/src/test/resources/images/faces/compare/1682052661610.jpg";
//    private static String imagePath2 = "face-search-core/src/test/resources/images/faces/compare/1682052669004.jpg";
//    private static String imagePath2 = "face-search-core/src/test/resources/images/faces/compare/1682053163961.jpg";
//    private static String imagePath1 = "face-search-test/src/main/resources/image/validate/index/张一鸣/1c7abcaf2dabdd2bc08e90c224d4c381.jpeg";
    private static String imagePath1 = "face-search-core/src/test/resources/images/faces/small/1.png";
    private static String imagePath2 = "face-search-core/src/test/resources/images/faces/small/2.png";
    public static void main(String[] args) {
        //口罩模型0.48，light模型0.52，normal模型0.62
        Map<String, String> map1 = getImagePathMap(imagePath1);
        Map<String, String> map2 = getImagePathMap(imagePath2);
        FaceDetection insightScrfdFaceDetection = new InsightScrfdFaceDetection(modelScrfdPath, 1);
        FaceKeyPoint insightCoordFaceKeyPoint = new InsightCoordFaceKeyPoint(modelCoordPath, 1);
        FaceRecognition insightArcFaceRecognition = new InsightArcFaceRecognition(modelArcPath, 1);
        FaceRecognition insightSeetaFaceRecognition = new SeetaFaceOpenRecognition(modelSeetaPath, 1);
        FaceAlignment simple005pFaceAlignment = new Simple005pFaceAlignment();
        FaceAlignment simple106pFaceAlignment = new Simple106pFaceAlignment();
        FaceDetection pcnNetworkFaceDetection = new PcnNetworkFaceDetection(new String[]{modelPcn1Path, modelPcn2Path, modelPcn3Path}, 1);
        FaceAttribute insightFaceAttribute = new InsightAttributeDetection(modelArrPath, 1);

        FaceFeatureExtractor extractor = new FaceFeatureExtractorImpl(
                insightScrfdFaceDetection, pcnNetworkFaceDetection, insightCoordFaceKeyPoint,
                simple005pFaceAlignment, insightArcFaceRecognition, insightFaceAttribute);

        for(String file1 : map1.keySet()){
            for(String file2 : map2.keySet()){
                Mat image1 = Imgcodecs.imread(map1.get(file1));
                long s = System.currentTimeMillis();
                ExtParam extParam = ExtParam.build().setMask(false).setTopK(20).setScoreTh(0).setIouTh(0);
                FaceImage faceImage1 = extractor.extract(ImageMat.fromCVMat(image1), extParam, null);
                List<FaceInfo> faceInfos1 = faceImage1.faceInfos();
                long e = System.currentTimeMillis();
                System.out.println("image1 extract cost:"+(e-s)+"ms");;

                Mat image2 = Imgcodecs.imread(map2.get(file2));
                s = System.currentTimeMillis();
                FaceImage faceImage2 = extractor.extract(ImageMat.fromCVMat(image2), extParam, null);
                List<FaceInfo> faceInfos2 = faceImage2.faceInfos();
                e = System.currentTimeMillis();
                System.out.println("image2 extract cost:"+(e-s)+"ms");
                float similarity = Similarity.cosineSimilarityNorm(faceInfos1.get(0).embedding.embeds, faceInfos2.get(0).embedding.embeds);
                System.out.println(file1 + ","+ file2 + ",face similarity="+similarity);
            }
        }

    }
}
