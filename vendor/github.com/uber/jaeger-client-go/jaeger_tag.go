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
	"fmt"

	"github.com/opentracing/opentracing-go/log"

	j "github.com/uber/jaeger-client-go/thrift-gen/jaeger"
)

type tags []*j.Tag

// ConvertLogsToJaegerTags converts log Fields into jaeger tags.
func ConvertLogsToJaegerTags(logFields []log.Field) []*j.Tag {
	fields := tags(make([]*j.Tag, 0, len(logFields)))
	for _, field := range logFields {
		field.Marshal(&fields)
	}
	return fields
}

func (t *tags) EmitString(key, value string) {
	*t = append(*t, &j.Tag{Key: key, VType: j.TagType_STRING, VStr: &value})
}

func (t *tags) EmitBool(key string, value bool) {
	*t = append(*t, &j.Tag{Key: key, VType: j.TagType_BOOL, VBool: &value})
}

func (t *tags) EmitInt(key string, value int) {
	vLong := int64(value)
	*t = append(*t, &j.Tag{Key: key, VType: j.TagType_LONG, VLong: &vLong})
}

func (t *tags) EmitInt32(key string, value int32) {
	vLong := int64(value)
	*t = append(*t, &j.Tag{Key: key, VType: j.TagType_LONG, VLong: &vLong})
}

func (t *tags) EmitInt64(key string, value int64) {
	*t = append(*t, &j.Tag{Key: key, VType: j.TagType_LONG, VLong: &value})
}

func (t *tags) EmitUint32(key string, value uint32) {
	vLong := int64(value)
	*t = append(*t, &j.Tag{Key: key, VType: j.TagType_LONG, VLong: &vLong})
}

func (t *tags) EmitUint64(key string, value uint64) {
	vLong := int64(value)
	*t = append(*t, &j.Tag{Key: key, VType: j.TagType_LONG, VLong: &vLong})
}

func (t *tags) EmitFloat32(key string, value float32) {
	vDouble := float64(value)
	*t = append(*t, &j.Tag{Key: key, VType: j.TagType_DOUBLE, VDouble: &vDouble})
}

func (t *tags) EmitFloat64(key string, value float64) {
	*t = append(*t, &j.Tag{Key: key, VType: j.TagType_DOUBLE, VDouble: &value})
}

func (t *tags) EmitObject(key string, value interface{}) {
	vStr := fmt.Sprintf("%+v", value)
	*t = append(*t, &j.Tag{Key: key, VType: j.TagType_STRING, VStr: &vStr})
}

func (t *tags) EmitLazyLogger(value log.LazyLogger) {
	value(t)
}
