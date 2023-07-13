package com.visual.face.search.core.models;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtSession;
import com.visual.face.search.core.base.BaseOnnxInfer;
import com.visual.face.search.core.base.FaceMaskPoint;
import com.visual.face.search.core.domain.ImageMat;
import com.visual.face.search.core.domain.QualityInfo;
import com.visual.face.search.core.utils.SoftMaxUtil;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.Collections;
import java.util.Map;

public class SeetaMaskFaceKeyPoint extends BaseOnnxInfer implements FaceMaskPoint {

    private static final int stride = 8;
    private static final int shape = 128;

    /**
     * 构造函数
     * @param modelPath     模型路径
     * @param threads       线程数
     */
    public SeetaMaskFaceKeyPoint(String modelPath, int threads) {
        super(modelPath, threads);
    }

    /**
     * 人脸关键点检测
     *
     * @param imageMat 图像数据
     * @param params   参数信息
     * @return
     */
    @Override
    public QualityInfo.MaskPoints inference(ImageMat imageMat, Map<String, Object> params) {
        Mat borderMat = null;
        Mat resizeMat = null;
        OnnxTensor tensor = null;
        OrtSession.Result output = null;
        try {
            Mat image = imageMat.toCvMat();
            //将图片转换为正方形
            int w = imageMat.getWidth();
            int h = imageMat.getHeight();
            int new_w = Math.max(h, w);
            int new_h = Math.max(h, w);
            if (Math.max(h, w) % stride != 0){
                new_w = new_w + (stride - Math.max(h, w) % stride);
                new_h = new_h + (stride - Math.max(h, w) % stride);
            }
            int ow = (new_w - w) / 2;
            int oh = (new_h - h) / 2;
            borderMat = new Mat();
            Core.copyMakeBorder(image, borderMat, oh, oh, ow, ow, Core.BORDER_CONSTANT, new Scalar(114, 114, 114));
            //对图片进行resize
            float ratio = 1.0f * shape / new_h;
            resizeMat = new Mat();
            Imgproc.resize(borderMat, resizeMat, new Size(shape, shape));
            //模型推理
            tensor = ImageMat.fromCVMat(resizeMat)
                    .blobFromImageAndDoReleaseMat(1.0/32, new Scalar(104, 117, 123), false)
                    .to4dFloatOnnxTensorAndDoReleaseMat(true);
            output = this.getSession().run(Collections.singletonMap(this.getInputName(), tensor));
            float[] value = ((float[][]) output.get(0).getValue())[0];
            //转换为标准的坐标点
            QualityInfo.MaskPoints pointList = QualityInfo.MaskPoints.build();
            for(int i=0; i<5; i++){
                float x = value[i * 4 + 0] / ratio * 128 - ow;
                float y = value[i * 4 + 1] / ratio * 128 - oh;
                double[] softMax = SoftMaxUtil.softMax(new double[]{value[i * 4 + 2], value[i * 4 + 3]});
                pointList.add(QualityInfo.MaskPoint.build(x, y, Double.valueOf(softMax[1]).floatValue()));
            }
            return pointList;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }finally {
            if(null != tensor){
                tensor.close();
            }
            if(null != output){
                output.close();
            }
            if(null != borderMat){
                borderMat.release();
            }
            if(null != resizeMat){
                resizeMat.release();
            }
        }
    }
}
