package com.visual.face.search.core.models;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtSession;
import com.visual.face.search.core.base.BaseOnnxInfer;
import com.visual.face.search.core.base.FaceRecognition;
import com.visual.face.search.core.domain.FaceInfo.Embedding;
import com.visual.face.search.core.domain.ImageMat;
import com.visual.face.search.core.utils.ArrayUtil;
import org.opencv.core.Scalar;
import java.util.Collections;
import java.util.Map;

/**
 * 人脸识别-人脸特征提取
 * git:https://github.com/SeetaFace6Open/index
 */
public class SeetaFaceOpenRecognition extends BaseOnnxInfer implements FaceRecognition {

    /**
     * 构造函数
     * @param modelPath     模型路径
     * @param threads       线程数
     */
    public SeetaFaceOpenRecognition(String modelPath, int threads) {
        super(modelPath, threads);
    }

    /**
     * 人脸识别，人脸特征向量
     * @param image 图像信息
     * @return
     */
    @Override
    public Embedding inference(ImageMat image, Map<String, Object> params) {
        OnnxTensor tensor = null;
        OrtSession.Result output = null;
        try {
            tensor = image.resizeAndNoReleaseMat(112,112)
                    .blobFromImageAndDoReleaseMat(1.0/255, new Scalar(0, 0, 0), false)
                    .to4dFloatOnnxTensorAndDoReleaseMat(true);
            output = getSession().run(Collections.singletonMap(getInputName(), tensor));
            float[] embeds = ((float[][]) output.get(0).getValue())[0];
            double normValue = ArrayUtil.matrixNorm(embeds);
            float[] embedding = ArrayUtil.division(embeds, Double.valueOf(normValue).floatValue());
            return Embedding.build(image.toBase64AndNoReleaseMat(), embedding);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }finally {
            if(null != tensor){
                tensor.close();
            }
            if(null != output){
                output.close();
            }
        }
    }

}
