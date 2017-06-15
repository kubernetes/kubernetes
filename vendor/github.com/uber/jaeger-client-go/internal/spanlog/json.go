// Copyright (c) 2016 Uber Technologies, Inc.

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

package spanlog

import (
	"encoding/json"
	"fmt"

	"github.com/opentracing/opentracing-go/log"
)

type fieldsAsMap map[string]string

// MaterializeWithJSON converts log Fields into JSON string
// TODO refactor into pluggable materializer
func MaterializeWithJSON(logFields []log.Field) ([]byte, error) {
	fields := fieldsAsMap(make(map[string]string, len(logFields)))
	for _, field := range logFields {
		field.Marshal(fields)
	}
	if event, ok := fields["event"]; ok && len(fields) == 1 {
		return []byte(event), nil
	}
	return json.Marshal(fields)
}

func (ml fieldsAsMap) EmitString(key, value string) {
	ml[key] = value
}

func (ml fieldsAsMap) EmitBool(key string, value bool) {
	ml[key] = fmt.Sprintf("%t", value)
}

func (ml fieldsAsMap) EmitInt(key string, value int) {
	ml[key] = fmt.Sprintf("%d", value)
}

func (ml fieldsAsMap) EmitInt32(key string, value int32) {
	ml[key] = fmt.Sprintf("%d", value)
}

func (ml fieldsAsMap) EmitInt64(key string, value int64) {
	ml[key] = fmt.Sprintf("%d", value)
}

func (ml fieldsAsMap) EmitUint32(key string, value uint32) {
	ml[key] = fmt.Sprintf("%d", value)
}

func (ml fieldsAsMap) EmitUint64(key string, value uint64) {
	ml[key] = fmt.Sprintf("%d", value)
}

func (ml fieldsAsMap) EmitFloat32(key string, value float32) {
	ml[key] = fmt.Sprintf("%f", value)
}

func (ml fieldsAsMap) EmitFloat64(key string, value float64) {
	ml[key] = fmt.Sprintf("%f", value)
}

func (ml fieldsAsMap) EmitObject(key string, value interface{}) {
	ml[key] = fmt.Sprintf("%+v", value)
}

func (ml fieldsAsMap) EmitLazyLogger(value log.LazyLogger) {
	value(ml)
}
