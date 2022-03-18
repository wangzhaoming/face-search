package com.visual.face.search.core.utils;

import org.opencv.core.Mat;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.util.Base64;
import java.util.Objects;

public class MatUtil {

    /**
     * 将Mat转换为BufferedImage
     * @param mat
     * @return  BufferedImage
     */
    public static BufferedImage matToBufferedImage(Mat mat) {
        int dataSize = mat.cols() * mat.rows() * (int) mat.elemSize();
        byte[] data = new byte[dataSize];
        mat.get(0, 0, data);
        int type = mat.channels() == 1 ? BufferedImage.TYPE_BYTE_GRAY : BufferedImage.TYPE_3BYTE_BGR;
        if (type == BufferedImage.TYPE_3BYTE_BGR) {
            for (int i = 0; i < dataSize; i += 3) {
                byte blue = data[i + 0];
                data[i + 0] = data[i + 2];
                data[i + 2] = blue;
            }
        }
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        image.getRaster().setDataElements(0, 0, mat.cols(), mat.rows(), data);
        return image;
    }

    /**
     * 将Mat转换为 Base64
     * @param mat
     * @return  Base64
     */
    public static String matToBase64(Mat mat) {
        ByteArrayOutputStream byteArrayOutputStream = null;
        try {
            byteArrayOutputStream = new ByteArrayOutputStream();
            ImageIO.write(matToBufferedImage(mat), "jpg", byteArrayOutputStream);
            byte[] bytes = byteArrayOutputStream.toByteArray();
            // 新版本JDK被移除，替换为Base64.Encoder
//            BASE64Encoder encoder = new BASE64Encoder();
//            return encoder.encodeBuffer(Objects.requireNonNull(bytes));
            Base64.Encoder encoder = Base64.getMimeEncoder();
            return encoder.encodeToString(Objects.requireNonNull(bytes));
        }catch (Exception e){
            throw new RuntimeException(e);
        }finally {
            if(null != byteArrayOutputStream){
                try {
                    byteArrayOutputStream.close();
                } catch (Exception e) {}
            }
        }
    }

}
