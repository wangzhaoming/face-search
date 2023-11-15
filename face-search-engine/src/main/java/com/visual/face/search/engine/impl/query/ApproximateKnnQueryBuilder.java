package com.visual.face.search.engine.impl.query;

import com.visual.face.search.engine.conf.Constant;
import org.apache.lucene.search.Query;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.query.AbstractQueryBuilder;
import org.opensearch.index.query.QueryShardContext;

import java.io.IOException;
import java.util.Map;
import java.util.Objects;

/**
 * 近似KNN搜索
 * @Author Foy Lian
 * @Date 2023/9/13 14:26
 **/

public class ApproximateKnnQueryBuilder extends AbstractQueryBuilder<ApproximateKnnQueryBuilder> {

    private Map<String, Object> mParams;

    public ApproximateKnnQueryBuilder(Map<String, Object> params){
        this.mParams = params;
    }

    @Override
    protected void doWriteTo(StreamOutput streamOutput) throws IOException {
    }

    @Override
    protected void doXContent(XContentBuilder xContentBuilder, Params params) throws IOException {
        xContentBuilder.startObject("knn");
        xContentBuilder.field(Constant.ColumnNameFaceVector,mParams);
        xContentBuilder.endObject();
    }

    @Override
    protected Query doToQuery(QueryShardContext queryShardContext) throws IOException {
        return null;
    }

    @Override
    protected boolean doEquals(ApproximateKnnQueryBuilder approximateKnnQueryBuilder) {
        return Objects.equals(this.mParams, approximateKnnQueryBuilder.mParams);
    }

    @Override
    protected int doHashCode() {
        return Objects.hash(new Object[]{this.mParams});
    }

    @Override
    public String getWriteableName() {
        return "knn";
    }
}
