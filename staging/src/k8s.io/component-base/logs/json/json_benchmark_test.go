/*
Copyright 2020 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package logs

import (
	"fmt"
	"testing"

	"go.uber.org/zap/zapcore"
)

var writer = zapcore.AddSync(&writeSyncer{})

func BenchmarkInfoLoggerInfo(b *testing.B) {
	logger, _ := NewJSONLogger(writer, writer)
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			logger.Info("test",
				"str", "foo",
				"int64-1", int64(1),
				"int64-2", int64(1),
				"float64", float64(1.0),
				"string1", "\n",
				"string2", "💩",
				"string3", "🤔",
				"string4", "🙊",
				"bool", true,
				"request", struct {
					Method  string `json:"method"`
					Timeout int    `json:"timeout"`
					secret  string `json:"secret"`
				}{
					Method:  "GET",
					Timeout: 10,
					secret:  "pony",
				},
			)
		}
	})
}

func BenchmarkZapLoggerError(b *testing.B) {
	logger, _ := NewJSONLogger(writer, writer)
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			logger.Error(fmt.Errorf("test for error:%s", "default"),
				"test",
				"str", "foo",
				"int64-1", int64(1),
				"int64-2", int64(1),
				"float64", float64(1.0),
				"string1", "\n",
				"string2", "💩",
				"string3", "🤔",
				"string4", "🙊",
				"bool", true,
				"request", struct {
					Method  string `json:"method"`
					Timeout int    `json:"timeout"`
					secret  string `json:"secret"`
				}{
					Method:  "GET",
					Timeout: 10,
					secret:  "pony",
				},
			)
		}
	})
}

func BenchmarkZapLoggerV(b *testing.B) {
	logger, _ := NewJSONLogger(writer, writer)
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			logger.V(1).Info("test",
				"str", "foo",
				"int64-1", int64(1),
				"int64-2", int64(1),
				"float64", float64(1.0),
				"string1", "\n",
				"string2", "💩",
				"string3", "🤔",
				"string4", "🙊",
				"bool", true,
				"request", struct {
					Method  string `json:"method"`
					Timeout int    `json:"timeout"`
					secret  string `json:"secret"`
				}{
					Method:  "GET",
					Timeout: 10,
					secret:  "pony",
				},
			)
		}
	})
}

type writeSyncer struct{}

var _ zapcore.WriteSyncer = (*writeSyncer)(nil)

func (w writeSyncer) Write(p []byte) (n int, err error) {
	return len(p), nil
}

func (w writeSyncer) Sync() error {
	return nil
}
