package com.visual.face.search.core.domain;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import com.visual.face.search.core.utils.MatUtil;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.dnn.Dnn;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Base64;

/**
 * 图片加载工具
 */
public class ImageMat implements Serializable {

    //静态加载动态链接库
    static{ nu.pattern.OpenCV.loadShared(); }
    private OrtEnvironment env = OrtEnvironment.getEnvironment();

    //对象成员
    private Mat mat;
    private ImageMat(Mat mat){
        this.mat = mat;
    }

    /**
     * 读取图片，并转换为Mat
     * @param imagePath 图片地址
     * @return
     */
    public static ImageMat fromImage(String imagePath){
        return new ImageMat(Imgcodecs.imread(imagePath));
    }

    /**
     * 直接读取Mat
     * @param mat 图片mat值
     * @return
     */
    public static ImageMat fromCVMat(Mat mat){
        try {
            return new ImageMat(mat);
        }catch (Exception e){
            throw new RuntimeException(e);
        }
    }

    /**
     * 读取图片，并转换为Mat
     * @param base64Str 图片Base64编码值
     * @return
     */
    public static ImageMat fromBase64(String base64Str){
        InputStream inputStream = null;
        try {
            if(base64Str.contains(",")){
                base64Str = base64Str.substring(base64Str.indexOf(",")+1);
            }
            Base64.Decoder decoder = Base64.getMimeDecoder();
            byte[] data = decoder.decode(base64Str);
            inputStream = new ByteArrayInputStream(data);
            return fromInputStream(inputStream);
        }catch (Exception e){
            throw new RuntimeException(e);
        }finally {
            if(null != inputStream){
                try {
                    inputStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    /**
     * 读取图片，并转换为Mat
     * @param inputStream 图片数据
     * @return
     */
    public static ImageMat fromInputStream(InputStream inputStream){
        try {
            BufferedImage image = ImageIO.read(inputStream);
            return fromBufferedImage(image);
        }catch (Exception e){
            throw new RuntimeException(e);
        }
    }

    /**
     * 读取图片，并转换为Mat
     * @param image 图片数据
     * @return
     */
    public static ImageMat fromBufferedImage(BufferedImage image){
        try {
            if(image.getType() != BufferedImage.TYPE_3BYTE_BGR){
                BufferedImage temp = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
                Graphics2D g = temp.createGraphics();
                try {
                    g.setComposite(AlphaComposite.Src);
                    g.drawImage(image, 0, 0, null);
                } finally {
                    g.dispose();
                }
                image = temp;
            }
            byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
            Mat mat = Mat.eye(image.getHeight(), image.getWidth(), CvType.CV_8UC3);
            mat.put(0, 0, pixels);
            return new ImageMat(mat);
        }catch (Exception e){
            throw new RuntimeException(e);
        }
    }


    /**
     * 显示图片，用于数据调试
     */
    public void imShow() {
        HighGui.imshow("image", mat);
        HighGui.waitKey();
    }



    /**
     *获取数据的宽度
     * @return
     */
    public int getWidth(){
        return (int) mat.size().width;
    }

    /**
     * 获取数据的高度
     * @return
     */
    public int getHeight(){
        return (int) mat.size().height;
    }

    /**
     * 克隆ImageMat
     * @return
     */
    public ImageMat clone(){
        return ImageMat.fromCVMat(this.mat.clone());
    }

    /**
     * 获取图像的中心点
     * @return
     */
    public Point center(){
        return new Point(mat.size(1)/2, mat.size(0)/2);
    }

    /**
     * 获取当前的CV Mat
     * @return
     */
    public Mat toCvMat() {
        return mat;
    }

    /**
     * 数据格式转换,不释放原始图片数据
     * @param code Imgproc.COLOR_*
     * @param release   是否释放参数mat
     */
    public ImageMat cvtColorAndNoReleaseMat(int code, boolean release) {
        return this.cvtColor(code, false);
    }

    /**
     * 数据格式转换,并释放原始图片数据
     * @param code Imgproc.COLOR_*
     * @param release   是否释放参数mat
     */
    public ImageMat cvtColorAndDoReleaseMat(int code, boolean release) {
        return this.cvtColor(code, true);
    }

    /**
     * 数据格式转换
     * @param code Imgproc.COLOR_*
     * @param release   是否释放参数mat
     */
    private ImageMat cvtColor(int code, boolean release) {
        try {
            Mat dst = new Mat();
            Imgproc.cvtColor(mat, dst, code);
            return new ImageMat(dst);
        }finally {
            if(release){
                this.release();
            }
        }
    }

    /**
     * 重新设置图片尺寸,不释放原始图片数据
     * @param width     图片宽度
     * @param height    图片高度
     * @return
     */
    public ImageMat resizeAndNoReleaseMat(int width, int height){
        return this.resize(width, height, false);
    }

    /**
     * 重新设置图片尺寸,并释放原始图片数据
     * @param width     图片宽度
     * @param height    图片高度
     * @return
     */
    public ImageMat resizeAndDoReleaseMat(int width, int height){
        return this.resize(width, height, true);
    }

    /**
     * 重新设置图片尺寸
     * @param width     图片宽度
     * @param height    图片高度
     * @param release   是否释放参数mat
     * @return
     */
    private ImageMat resize(int width, int height, boolean release){
        try {
            Mat dst = new Mat();
            Imgproc.resize(mat, dst, new Size(width,height), 0, 0, Imgproc.INTER_AREA);
            return new ImageMat(dst);
        }finally {
            if(release){
                this.release();
            }
        }
    }

    /**
     * 对图像进行补边操作，不释放原始的图片
     * @param top           向上扩展的高度
     * @param bottom        向下扩展的高度
     * @param left          向左扩展的宽度
     * @param right         向右扩展的宽度
     * @param borderType    补边的类型
     * @return  补边后的图像
     */
    public ImageMat copyMakeBorderAndNotReleaseMat(int top, int bottom, int left, int right, int borderType){
        return this.copyMakeBorder(top, bottom, left, right, borderType, false);
    }

    /**
     * 对图像进行补边操作，并且释放原始的图片
     * @param top           向上扩展的高度
     * @param bottom        向下扩展的高度
     * @param left          向左扩展的宽度
     * @param right         向右扩展的宽度
     * @param borderType    补边的类型
     * @return  补边后的图像
     */
    public ImageMat copyMakeBorderAndDoReleaseMat(int top, int bottom, int left, int right, int borderType){
        return this.copyMakeBorder(top, bottom, left, right, borderType, true);
    }

    /**
     * 对图像进行补边操作
     * @param top           向上扩展的高度
     * @param bottom        向下扩展的高度
     * @param left          向左扩展的宽度
     * @param right         向右扩展的宽度
     * @param borderType    补边的类型
     * @param release       是否释放原始的图片
     * @return  补边后的图像
     */
    private ImageMat copyMakeBorder(int top, int bottom, int left, int right, int borderType, boolean release){
        try {
            Mat tempMat = new Mat();
            Core.copyMakeBorder(mat, tempMat, top, bottom, left, right, borderType);
            return new ImageMat(tempMat);
        }finally {
            if(release){
                this.release();
            }
        }
    }


    /**
     * 对图像进行预处理,不释放原始图片数据
     * @param scale     图像各通道数值的缩放比例
     * @param mean      用于各通道减去的值，以降低光照的影响
     * @param swapRB    交换RB通道，默认为False.
     * @return
     */
    public ImageMat blobFromImageAndNoReleaseMat(double scale, Scalar mean, boolean swapRB){
        return this.blobFromImage(scale, mean, swapRB, false);
    }

    /**
     * 对图像进行预处理,并释放原始图片数据:（先交换RB通道（swapRB），再减法（mean），最后缩放（scale））
     * @param scale     图像各通道数值的缩放比例
     * @param mean      用于各通道减去的值，以降低光照的影响
     * @param swapRB    交换RB通道，默认为False.
     * @return
     */
    public ImageMat blobFromImageAndDoReleaseMat(double scale, Scalar mean, boolean swapRB){
        return this.blobFromImage(scale, mean, swapRB, true);
    }

    /**
     * 对图像进行预处理
     * @param scale     图像各通道数值的缩放比例
     * @param mean      用于各通道减去的值，以降低光照的影响
     * @param swapRB    交换RB通道，默认为False.
     * @param release   是否释放参数mat
     * @return
     */
    private ImageMat blobFromImage(double scale, Scalar mean, boolean swapRB, boolean release){
        try {
            Mat dst = Dnn.blobFromImage(mat, scale, new Size( mat.cols(), mat.rows()), mean, swapRB);
            java.util.List<Mat> mats = new ArrayList<>();
            Dnn.imagesFromBlob(dst, mats);
            dst.release();
            return new ImageMat(mats.get(0));
        }finally {
            if(release){
                this.release();
            }
        }
    }


    /**
     * 转换为base64,不释放原始图片数据
     * @return
     */
    public String toBase64AndNoReleaseMat(){
        return toBase64(false);
    }

    /**
     * 转换为base64,并释放原始图片数据
     * @return
     */
    public String toBase64AndDoReleaseMat(){
        return toBase64(true);
    }

    /**
     * 转换为base64
     * @param release   是否释放参数mat
     * @return
     */
    private String toBase64(boolean release){
        if(null != mat){
            try {
                return MatUtil.matToBase64(mat);
            }finally {
                if(release){
                    this.release();
                }
            }
        }else{
            return null;
        }
    }

    /**
     * 转换为整形数组,不释放原始图片数据
     * @param firstChannel
     * @return
     */
    public int[][][][] to4dIntArrayAndNoReleaseMat(boolean firstChannel){
        return this.to4dIntArray(firstChannel, false);
    }

    /**
     * 转换为整形数组,并释放原始图片数据
     * @param firstChannel
     * @return
     */
    public int[][][][] to4dIntArrayAndDoReleaseMat(boolean firstChannel){
        return this.to4dIntArray(firstChannel, true);
    }


    /**
     * 转换为整形数组
     * @param firstChannel
     * @param release   是否释放参数mat
     * @return
     */
    private int[][][][] to4dIntArray(boolean firstChannel, boolean release){
        try {
            int width = this.mat.cols();
            int height = this.mat.rows();
            int channel = this.mat.channels();
            int[][][][] array;
            if(firstChannel){
                array = new int[1][channel][height][width];
                for(int i=0; i<height; i++){
                    for(int j=0; j<width; j++){
                        double[] c = mat.get(i, j);
                        for(int k=0; k< channel; k++){
                            array[0][k][i][j] = (int) Math.round(c[k]);
                        }
                    }
                }
            }else{
                array = new int[1][height][width][channel];
                for(int i=0; i<height; i++){
                    for(int j=0; j<width; j++){
                        double[] c = mat.get(i, j);
                        for(int k=0; k< channel; k++){
                            array[0][i][j][k] = (int) Math.round(c[k]);
                        }
                    }
                }
            }
            return array;
        }finally {
            if(release){
                this.release();
            }
        }
    }


    /**
     * 转换为长整形数组,不释放原始图片数据
     * @param firstChannel
     * @return
     */
    public long[][][][] to4dLongArrayAndNoReleaseMat(boolean firstChannel){
        return this.to4dLongArray(firstChannel, false);
    }

    /**
     * 转换为长整形数组,并释放原始图片数据
     * @param firstChannel
     * @return
     */
    public long[][][][] to4dLongArrayAndDoReleaseMat(boolean firstChannel){
        return this.to4dLongArray(firstChannel, true);
    }
    /**
     * 转换为长整形数组
     * @param firstChannel
     * @param release   是否释放参数mat
     * @return
     */
    private long[][][][] to4dLongArray(boolean firstChannel, boolean release){
        try {
            int width = this.mat.cols();
            int height = this.mat.rows();
            int channel = this.mat.channels();
            long[][][][] array;
            if(firstChannel){
                array = new long[1][channel][height][width];
                for(int i=0; i<height; i++){
                    for(int j=0; j<width; j++){
                        double[] c = mat.get(i, j);
                        for(int k=0; k< channel; k++){
                            array[0][k][i][j] = Math.round(c[k]);
                        }
                    }
                }
            }else{
                array = new long[1][height][width][channel];
                for(int i=0; i<height; i++){
                    for(int j=0; j<width; j++){
                        double[] c = mat.get(i, j);
                        for(int k=0; k< channel; k++){
                            array[0][i][j][k] = Math.round(c[k]);
                        }
                    }
                }
            }
            return array;
        }finally {
            if(release){
                this.release();
            }
        }
    }


    /**
     * 转换为单精度形数组,不释放原始图片数据
     * @param firstChannel
     * @return
     */
    public float[][][][] to4dFloatArrayAndNoReleaseMat(boolean firstChannel){
        return this.to4dFloatArray(firstChannel, false);
    }

    /**
     * 转换为单精度形数组,并释放原始图片数据
     * @param firstChannel
     * @return
     */
    public float[][][][] to4dFloatArrayAndDoReleaseMat(boolean firstChannel){
        return this.to4dFloatArray(firstChannel, true);
    }

    /**
     * 转换为单精度形数组
     * @param firstChannel
     * @param release   是否释放参数mat
     * @return
     */
    private float[][][][] to4dFloatArray(boolean firstChannel, boolean release){
        try {
            int width = this.mat.cols();
            int height = this.mat.rows();
            int channel = this.mat.channels();
            float[][][][] array;
            if(firstChannel){
                array = new float[1][channel][height][width];
                for(int i=0; i<height; i++){
                    for(int j=0; j<width; j++){
                        double[] c = mat.get(i, j);
                        for(int k=0; k< channel; k++){
                            array[0][k][i][j] = (float) c[k];
                        }
                    }
                }
            }else{
                array = new float[1][height][width][channel];
                for(int i=0; i<height; i++){
                    for(int j=0; j<width; j++){
                        double[] c = mat.get(i, j);
                        for(int k=0; k< channel; k++){
                            array[0][i][j][k] = (float) c[k];
                        }
                    }
                }
            }
            return array;
        }finally {
            if(release){
                this.release();
            }
        }
    }


    /**
     * 转换为双精度形数组,不释放原始图片数据
     * @param firstChannel
     * @return
     */
    public double[][][][] to4dDoubleArrayAndNoReleaseMat(boolean firstChannel){
        return this.to4dDoubleArray(firstChannel, false);
    }

    /**
     * 转换为双精度形数组,并释放原始图片数据
     * @param firstChannel
     * @return
     */
    public double[][][][] to4dDoubleArrayAndDoReleaseMat(boolean firstChannel){
        return this.to4dDoubleArray(firstChannel, true);
    }

    /**
     * 转换为双精度形数组
     * @param firstChannel
     * @param release   是否释放参数mat
     * @return
     */
    private double[][][][] to4dDoubleArray(boolean firstChannel, boolean release){
        try {
            int width = this.mat.cols();
            int height = this.mat.rows();
            int channel = this.mat.channels();
            double[][][][] array;
            if(firstChannel){
                array = new double[1][channel][height][width];
                for(int i=0; i<height; i++){
                    for(int j=0; j<width; j++){
                        double[] c = mat.get(i, j);
                        for(int k=0; k< channel; k++){
                            array[0][k][i][j] = c[k];
                        }
                    }
                }
            }else{
                array = new double[1][height][width][channel];
                for(int i=0; i<height; i++){
                    for(int j=0; j<width; j++){
                        double[] c = mat.get(i, j);
                        for(int k=0; k< channel; k++){
                            array[0][i][j][k] = c[k];
                        }
                    }
                }
            }
            return array;
        }finally {
            if(release){
                this.release();
            }
        }
    }


    /**
     * 转换为整形OnnxTensor,不释放原始图片数据
     * @param firstChannel
     * @return
     */
    public OnnxTensor to4dIntOnnxTensorAndNoReleaseMat(boolean firstChannel){
        try {
            return OnnxTensor.createTensor(env, this.to4dIntArrayAndNoReleaseMat(firstChannel));
        }catch (Exception e){
            throw new RuntimeException(e);
        }
    }

    /**
     * 转换为整形OnnxTensor,并释放原始图片数据
     * @param firstChannel
     * @return
     */
    public OnnxTensor to4dIntOnnxTensorAndDoReleaseMat(boolean firstChannel){
        try {
            return OnnxTensor.createTensor(env, this.to4dIntArrayAndDoReleaseMat(firstChannel));
        }catch (Exception e){
            throw new RuntimeException(e);
        }
    }

    /**
     * 转换为长整形OnnxTensor,不释放原始图片数据
     * @param firstChannel
     * @return
     */
    public OnnxTensor to4dLongOnnxTensorAndNoReleaseMat(boolean firstChannel) {
        try {
            return OnnxTensor.createTensor(env, this.to4dLongArrayAndNoReleaseMat(firstChannel));
        }catch (Exception e){
            throw new RuntimeException(e);
        }
    }

    /**
     * 转换为长整形OnnxTensor,并释放原始图片数据
     * @param firstChannel
     * @return
     */
    public OnnxTensor to4dLongOnnxTensorAndDoReleaseMat(boolean firstChannel) {
        try {
            return OnnxTensor.createTensor(env, this.to4dLongArrayAndDoReleaseMat(firstChannel));
        }catch (Exception e){
            throw new RuntimeException(e);
        }
    }

    /**
     * 转换为单精度形OnnxTensor,不释放原始图片数据
     * @param firstChannel
     * @return
     */
    public OnnxTensor to4dFloatOnnxTensorAndNoReleaseMat(boolean firstChannel) {
        try {
            return OnnxTensor.createTensor(env, this.to4dFloatArrayAndNoReleaseMat(firstChannel));
        }catch (Exception e){
            throw new RuntimeException(e);
        }
    }

    /**
     * 转换为单精度形OnnxTensor,并释放原始图片数据
     * @param firstChannel
     * @return
     */
    public OnnxTensor to4dFloatOnnxTensorAndDoReleaseMat(boolean firstChannel) {
        try {
            // return OnnxTensor.createTensor(env, this.to4dFloatArrayAndDoReleaseMat(firstChannel));
            // 经过测试，直接使用floatbuffer创建OnnxTensor,识别接口[/visual/search/do]的QPS，性能(qps)大约提升了100%。
            long[] shape = new long[4];
            return OnnxTensor.createTensor(env, this.toFloatBuffer(shape,firstChannel,true),shape);
        }catch (Exception e){
            throw new RuntimeException(e);
        }
    }

    /**
     * 获取指定行列和通道对应的值
     * @param rowIndex 行号
     * @param colIndex 列号
     * @param channelIndex 通道号
     * @param array 图像一维数组
     * @param cols 图像的列数
     * @param channels 图像的通道数
     * @return 指定行列和通道对应的值
     */
    public static float get(int rowIndex,int colIndex,int channelIndex,float[] array,int cols,int channels){
        return array[rowIndex*cols*channels+colIndex*channels+channelIndex];
    }

    /**
     * 将mat转换成floatbuffer
     * @param shape 用于接收数组的维度信息，传递参数为 new long[4] 的空数组
     * @param firstChannel
     * @param release 是否释放mat
     * @return mat对应的floatbuffer
     */
    public FloatBuffer toFloatBuffer(long[] shape, boolean firstChannel, boolean release){
        try{
            return toFloatBuffer(shape,firstChannel,this.mat);
        }finally {
            if(release){
                this.release();
            }
        }
    }

    /**
     * 将mat转换成floatbuffer
     * @param shape 用于接收数组的维度信息，传递参数为 new long[4] 的空数组
     * @param firstChannel
     * @param mat
     * @return mat对应的floatbuffer
     */
    public static FloatBuffer toFloatBuffer(long[] shape,boolean firstChannel,Mat mat) {
        //为了提高性能将mat转换成1维数组，遍历一维数组的性能远高于频繁的mat.get
        float[] data = new float[(int) (mat.total() * mat.channels())];
        mat.get(0, 0, data);

        FloatBuffer floatBuffer = FloatBuffer.allocate(data.length );
        int width = mat.cols();
        int height =mat.rows();
        int channel =mat.channels();
        if(firstChannel){
            for(int k=0; k< channel; k++){
                for(int i=0; i<height; i++){
                    for(int j=0; j<width; j++){
                        floatBuffer.put(get(i,j,k,data,width,channel));
                    }
                }
            }
            shape[0] = 1;
            shape[1] = channel;
            shape[2] = height;
            shape[3] = width;
        }else{
            for(int i=0; i<height; i++){
                for(int j=0; j<width; j++){
                    for(int k=0; k< channel; k++){
                        floatBuffer.put(get(i,j,k,data,width,channel));
                    }
                }
            }
            shape[0] = 1;
            shape[1] = height;
            shape[2] = width;
            shape[3] = channel;
        }
        floatBuffer.rewind();
        return floatBuffer;
    }


    /**
     * 转换为双精度形OnnxTensor,不释放原始图片数据
     * @param firstChannel
     * @return
     */
    public OnnxTensor to4dDoubleOnnxTensorAndNoReleaseMat(boolean firstChannel) {
        try {
            return OnnxTensor.createTensor(env, this.to4dDoubleArrayAndNoReleaseMat(firstChannel));
        }catch (Exception e){
            throw new RuntimeException(e);
        }
    }

    /**
     * 转换为双精度形OnnxTensor,并释放原始图片数据
     * @param firstChannel
     * @return
     */
    public OnnxTensor to4dDoubleOnnxTensorAndDoReleaseMat(boolean firstChannel) {
        try {
            return OnnxTensor.createTensor(env, this.to4dDoubleArrayAndDoReleaseMat(firstChannel));
        }catch (Exception e){
            throw new RuntimeException(e);
        }
    }

    /**
     * 释放资源
     */
    public void release(){
        if(this.mat != null){
            try {
                this.mat.release();
                this.mat = null;
            }catch (Exception e){
                e.printStackTrace();
            }
        }
    }
}
