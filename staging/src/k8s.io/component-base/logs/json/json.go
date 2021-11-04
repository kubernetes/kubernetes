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
	"os"
	"time"

	"github.com/go-logr/logr"
	"github.com/go-logr/zapr"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"k8s.io/component-base/config"
	"k8s.io/component-base/logs/registry"
)

var (
	// timeNow stubbed out for testing
	timeNow = time.Now
)

// NewJSONLogger creates a new json logr.Logger and its associated
// flush function. The separate error stream is optional and may be nil.
func NewJSONLogger(infoStream, errorStream zapcore.WriteSyncer) (logr.Logger, func()) {
	encoder := zapcore.NewJSONEncoder(encoderConfig)
	var core zapcore.Core
	if errorStream == nil {
		core = zapcore.NewCore(encoder, zapcore.AddSync(infoStream), zapcore.Level(-127))
	} else {
		// Set up writing of error messages to stderr and info messages
		// to stdout. Info messages get optionally buffered and flushed
		// - through klog.FlushLogs -> zapr Flush -> zap Sync
		// - when an error gets logged
		//
		// The later is important when both streams get merged into a single
		// stream by the consumer (same console for a command line tool, pod
		// log for a container) because without it, messages get reordered.
		flushError := writeWithFlushing{
			WriteSyncer: errorStream,
			other:       infoStream,
		}
		highPriority := zap.LevelEnablerFunc(func(lvl zapcore.Level) bool {
			return lvl >= zapcore.ErrorLevel
		})
		lowPriority := zap.LevelEnablerFunc(func(lvl zapcore.Level) bool {
			return lvl < zapcore.ErrorLevel
		})
		core = zapcore.NewTee(
			zapcore.NewCore(encoder, flushError, highPriority),
			zapcore.NewCore(encoder, infoStream, lowPriority),
		)
	}
	l := zap.New(core, zap.WithCaller(true))
	return zapr.NewLoggerWithOptions(l, zapr.LogInfoLevel("v"), zapr.ErrorKey("err")), func() {
		l.Sync()
	}
}

var encoderConfig = zapcore.EncoderConfig{
	MessageKey:     "msg",
	CallerKey:      "caller",
	TimeKey:        "ts",
	EncodeTime:     epochMillisTimeEncoder,
	EncodeDuration: zapcore.StringDurationEncoder,
	EncodeCaller:   zapcore.ShortCallerEncoder,
}

func epochMillisTimeEncoder(_ time.Time, enc zapcore.PrimitiveArrayEncoder) {
	nanos := timeNow().UnixNano()
	millis := float64(nanos) / float64(time.Millisecond)
	enc.AppendFloat64(millis)
}

// Factory produces JSON logger instances.
type Factory struct{}

var _ registry.LogFormatFactory = Factory{}

func (f Factory) Create(options config.FormatOptions) (logr.Logger, func()) {
	if options.JSON.SplitStream {
		// stdout for info messages, stderr for errors.
		infoStream := zapcore.Lock(os.Stdout)
		size := options.JSON.InfoBufferSize.Value()
		if size > 0 {
			// Prevent integer overflow.
			if size > 2*1024*1024*1024 {
				size = 2 * 1024 * 1024 * 1024
			}
			infoStream = &zapcore.BufferedWriteSyncer{
				WS:   infoStream,
				Size: int(size),
			}
		}
		return NewJSONLogger(infoStream, zapcore.Lock(os.Stderr))
	}
	// The default is to write to stderr (same as in klog's text output,
	// doesn't get mixed with normal program output).
	out := zapcore.Lock(os.Stderr)
	return NewJSONLogger(out, out)
}

// writeWithFlushing is a wrapper around an output stream which flushes another
// output stream before each write.
type writeWithFlushing struct {
	zapcore.WriteSyncer
	other zapcore.WriteSyncer
}

func (f writeWithFlushing) Write(bs []byte) (int, error) {
	f.other.Sync()
	return f.WriteSyncer.Write(bs)
}
