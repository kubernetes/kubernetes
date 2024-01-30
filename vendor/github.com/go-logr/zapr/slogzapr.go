//go:build go1.21
// +build go1.21

/*
Copyright 2023 The logr Authors.

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

package zapr

import (
	"context"
	"log/slog"
	"runtime"

	"github.com/go-logr/logr/slogr"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

var _ slogr.SlogSink = &zapLogger{}

func (zl *zapLogger) Handle(_ context.Context, record slog.Record) error {
	zapLevel := zap.InfoLevel
	intLevel := 0
	isError := false
	switch {
	case record.Level >= slog.LevelError:
		zapLevel = zap.ErrorLevel
		isError = true
	case record.Level >= slog.LevelWarn:
		zapLevel = zap.WarnLevel
	case record.Level >= 0:
		// Already set above -> info.
	default:
		zapLevel = zapcore.Level(record.Level)
		intLevel = int(-zapLevel)
	}

	if checkedEntry := zl.l.Check(zapLevel, record.Message); checkedEntry != nil {
		checkedEntry.Time = record.Time
		checkedEntry.Caller = pcToCallerEntry(record.PC)
		var fieldsBuffer [2]zap.Field
		fields := fieldsBuffer[:0]
		if !isError && zl.numericLevelKey != "" {
			// Record verbosity for info entries.
			fields = append(fields, zap.Int(zl.numericLevelKey, intLevel))
		}
		// Inline all attributes.
		fields = append(fields, zap.Inline(zapcore.ObjectMarshalerFunc(func(enc zapcore.ObjectEncoder) error {
			record.Attrs(func(attr slog.Attr) bool {
				encodeSlog(enc, attr)
				return true
			})
			return nil
		})))
		checkedEntry.Write(fields...)
	}
	return nil
}

func encodeSlog(enc zapcore.ObjectEncoder, attr slog.Attr) {
	if attr.Equal(slog.Attr{}) {
		// Ignore empty attribute.
		return
	}

	// Check in order of expected frequency, most common ones first.
	//
	// Usage statistics for parameters from Kubernetes 152876a3e,
	// calculated with k/k/test/integration/logs/benchmark:
	//
	// kube-controller-manager -v10:
	// strings: 10043 (85%)
	// with API objects: 2 (0% of all arguments)
	//   types and their number of usage: NodeStatus:2
	// numbers: 792 (6%)
	// ObjectRef: 292 (2%)
	// others: 595 (5%)
	//
	// kube-scheduler -v10:
	// strings: 1325 (40%)
	// with API objects: 109 (3% of all arguments)
	//   types and their number of usage: PersistentVolume:50 PersistentVolumeClaim:59
	// numbers: 473 (14%)
	// ObjectRef: 1305 (39%)
	// others: 176 (5%)

	kind := attr.Value.Kind()
	switch kind {
	case slog.KindString:
		enc.AddString(attr.Key, attr.Value.String())
	case slog.KindLogValuer:
		// This includes klog.KObj.
		encodeSlog(enc, slog.Attr{
			Key:   attr.Key,
			Value: attr.Value.Resolve(),
		})
	case slog.KindInt64:
		enc.AddInt64(attr.Key, attr.Value.Int64())
	case slog.KindUint64:
		enc.AddUint64(attr.Key, attr.Value.Uint64())
	case slog.KindFloat64:
		enc.AddFloat64(attr.Key, attr.Value.Float64())
	case slog.KindBool:
		enc.AddBool(attr.Key, attr.Value.Bool())
	case slog.KindDuration:
		enc.AddDuration(attr.Key, attr.Value.Duration())
	case slog.KindTime:
		enc.AddTime(attr.Key, attr.Value.Time())
	case slog.KindGroup:
		attrs := attr.Value.Group()
		if attr.Key == "" {
			// Inline group.
			for _, attr := range attrs {
				encodeSlog(enc, attr)
			}
			return
		}
		if len(attrs) == 0 {
			// Ignore empty group.
			return
		}
		_ = enc.AddObject(attr.Key, marshalAttrs(attrs))
	default:
		// We have to go through reflection in zap.Any to get support
		// for e.g. fmt.Stringer.
		zap.Any(attr.Key, attr.Value.Any()).AddTo(enc)
	}
}

type marshalAttrs []slog.Attr

func (attrs marshalAttrs) MarshalLogObject(enc zapcore.ObjectEncoder) error {
	for _, attr := range attrs {
		encodeSlog(enc, attr)
	}
	return nil
}

var _ zapcore.ObjectMarshaler = marshalAttrs(nil)

func pcToCallerEntry(pc uintptr) zapcore.EntryCaller {
	if pc == 0 {
		return zapcore.EntryCaller{}
	}
	// Same as https://cs.opensource.google/go/x/exp/+/642cacee:slog/record.go;drc=642cacee5cc05231f45555a333d07f1005ffc287;l=70
	fs := runtime.CallersFrames([]uintptr{pc})
	f, _ := fs.Next()
	if f.File == "" {
		return zapcore.EntryCaller{}
	}
	return zapcore.EntryCaller{
		Defined:  true,
		PC:       pc,
		File:     f.File,
		Line:     f.Line,
		Function: f.Function,
	}
}

func (zl *zapLogger) WithAttrs(attrs []slog.Attr) slogr.SlogSink {
	newLogger := *zl
	newLogger.l = newLogger.l.With(zap.Inline(marshalAttrs(attrs)))
	return &newLogger
}

func (zl *zapLogger) WithGroup(name string) slogr.SlogSink {
	newLogger := *zl
	newLogger.l = newLogger.l.With(zap.Namespace(name))
	return &newLogger
}
