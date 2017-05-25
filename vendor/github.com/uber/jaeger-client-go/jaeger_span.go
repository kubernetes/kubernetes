// Copyright (c) 2017 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package jaeger

import (
	"time"

	"github.com/opentracing/opentracing-go"

	j "github.com/uber/jaeger-client-go/thrift-gen/jaeger"
	"github.com/uber/jaeger-client-go/utils"
)

// buildJaegerSpan builds jaeger span based on internal span.
func buildJaegerSpan(span *Span) *j.Span {
	startTime := utils.TimeToMicrosecondsSinceEpochInt64(span.startTime)
	duration := span.duration.Nanoseconds() / int64(time.Microsecond)
	jaegerSpan := &j.Span{
		TraceIdLow:    int64(span.context.traceID.Low),
		TraceIdHigh:   int64(span.context.traceID.High),
		SpanId:        int64(span.context.spanID),
		ParentSpanId:  int64(span.context.parentID),
		OperationName: span.operationName,
		Flags:         int32(span.context.flags),
		StartTime:     startTime,
		Duration:      duration,
		Tags:          buildTags(span.tags),
		Logs:          buildLogs(span.logs),
		References:    buildReferences(span.references),
	}
	return jaegerSpan
}

func buildTags(tags []Tag) []*j.Tag {
	jTags := make([]*j.Tag, 0, len(tags))
	for _, tag := range tags {
		jTag := buildTag(&tag)
		jTags = append(jTags, jTag)
	}
	return jTags
}

func buildLogs(logs []opentracing.LogRecord) []*j.Log {
	jLogs := make([]*j.Log, 0, len(logs))
	for _, log := range logs {
		jLog := &j.Log{
			Timestamp: utils.TimeToMicrosecondsSinceEpochInt64(log.Timestamp),
			Fields:    ConvertLogsToJaegerTags(log.Fields),
		}
		jLogs = append(jLogs, jLog)
	}
	return jLogs
}

func buildTag(tag *Tag) *j.Tag {
	jTag := &j.Tag{Key: tag.key}
	switch value := tag.value.(type) {
	case string:
		vStr := truncateString(value)
		jTag.VStr = &vStr
		jTag.VType = j.TagType_STRING
	case []byte:
		if len(value) > maxAnnotationLength {
			value = value[:maxAnnotationLength]
		}
		jTag.VBinary = value
		jTag.VType = j.TagType_BINARY
	case int:
		vLong := int64(value)
		jTag.VLong = &vLong
		jTag.VType = j.TagType_LONG
	case uint:
		vLong := int64(value)
		jTag.VLong = &vLong
		jTag.VType = j.TagType_LONG
	case int8:
		vLong := int64(value)
		jTag.VLong = &vLong
		jTag.VType = j.TagType_LONG
	case uint8:
		vLong := int64(value)
		jTag.VLong = &vLong
		jTag.VType = j.TagType_LONG
	case int16:
		vLong := int64(value)
		jTag.VLong = &vLong
		jTag.VType = j.TagType_LONG
	case uint16:
		vLong := int64(value)
		jTag.VLong = &vLong
		jTag.VType = j.TagType_LONG
	case int32:
		vLong := int64(value)
		jTag.VLong = &vLong
		jTag.VType = j.TagType_LONG
	case uint32:
		vLong := int64(value)
		jTag.VLong = &vLong
		jTag.VType = j.TagType_LONG
	case int64:
		vLong := int64(value)
		jTag.VLong = &vLong
		jTag.VType = j.TagType_LONG
	case uint64:
		vLong := int64(value)
		jTag.VLong = &vLong
		jTag.VType = j.TagType_LONG
	case float32:
		vDouble := float64(value)
		jTag.VDouble = &vDouble
		jTag.VType = j.TagType_DOUBLE
	case float64:
		vDouble := float64(value)
		jTag.VDouble = &vDouble
		jTag.VType = j.TagType_DOUBLE
	case bool:
		vBool := value
		jTag.VBool = &vBool
		jTag.VType = j.TagType_BOOL
	default:
		vStr := truncateString(stringify(value))
		jTag.VStr = &vStr
		jTag.VType = j.TagType_STRING
	}
	return jTag
}

func buildReferences(references []Reference) []*j.SpanRef {
	retMe := make([]*j.SpanRef, 0, len(references))
	for _, ref := range references {
		if ref.Type == opentracing.ChildOfRef {
			retMe = append(retMe, spanRef(ref.Context, j.SpanRefType_CHILD_OF))
		} else if ref.Type == opentracing.FollowsFromRef {
			retMe = append(retMe, spanRef(ref.Context, j.SpanRefType_FOLLOWS_FROM))
		}
	}
	return retMe
}

func spanRef(ctx SpanContext, refType j.SpanRefType) *j.SpanRef {
	return &j.SpanRef{
		RefType:     refType,
		TraceIdLow:  int64(ctx.traceID.Low),
		TraceIdHigh: int64(ctx.traceID.High),
		SpanId:      int64(ctx.spanID),
	}
}
