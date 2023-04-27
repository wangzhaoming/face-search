package com.visual.face.search.core.domain;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

public class QualityInfo {

    public MaskPoints maskPoints;

    private QualityInfo(MaskPoints maskPoints) {
        this.maskPoints = maskPoints;
    }

    public static QualityInfo build(MaskPoints maskPoints){
        return new QualityInfo(maskPoints);
    }

    public MaskPoints getMaskPoints() {
        return maskPoints;
    }


    public boolean isMask(){
        return true;
    }


    /**
     * 遮挡类
     */
    public static class Mask implements Serializable {
        /**遮挡分数*/
        public float score;

        public static Mask build(float score){
            return new QualityInfo.Mask(score);
        }

        private Mask(float score) {
            this.score = score;
        }

        public float getScore() {
            return score;
        }

        @Override
        public String toString() {
            return "Mask{" + "score=" + score + '}';
        }
    }

    /**
     * 点遮挡类
     */
    public static class MaskPoint extends Mask{
        /**坐标X的值**/
        public float x;
        /**坐标Y的值**/
        public float y;

        public static MaskPoint build(float x, float y, float score){
            return new MaskPoint(x, y, score);
        }

        private MaskPoint(float x, float y, float score) {
            super(score);
            this.x = x;
            this.y = y;
        }

        public float getX() {
            return x;
        }

        public float getY() {
            return y;
        }

        @Override
        public String toString() {
            return "MaskPoint{" + "x=" + x + ", y=" + y + ", score=" + score + '}';
        }
    }

    /**
     * 点遮挡类集合
     */
    public static class MaskPoints extends ArrayList<MaskPoint> {

        private MaskPoints(){}

        /**
         * 构建一个集合
         * @return
         */
        public static MaskPoints build(){
            return new MaskPoints();
        }

        /**
         * 添加点
         * @param point
         * @return
         */
        public MaskPoints add(MaskPoint...point){
            super.addAll(Arrays.asList(point));
            return this;
        }
    }
}
