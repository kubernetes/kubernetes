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
		core = zapcore.NewCore(encoder, infoStream, zapcore.Level(-127))
	} else {
		highPriority := zap.LevelEnablerFunc(func(lvl zapcore.Level) bool {
			return lvl >= zapcore.ErrorLevel
		})
		lowPriority := zap.LevelEnablerFunc(func(lvl zapcore.Level) bool {
			return lvl < zapcore.ErrorLevel
		})
		core = zapcore.NewTee(
			zapcore.NewCore(encoder, errorStream, highPriority),
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
	stderr := zapcore.Lock(os.Stderr)
	if options.JSON.SplitStream {
		stdout := zapcore.Lock(os.Stdout)
		size := options.JSON.InfoBufferSize.Value()
		if size > 0 {
			// Prevent integer overflow.
			if size > 2*1024*1024*1024 {
				size = 2 * 1024 * 1024 * 1024
			}
			stdout = &zapcore.BufferedWriteSyncer{
				WS:   stdout,
				Size: int(size),
			}
		}
		// stdout for info messages, stderr for errors.
		return NewJSONLogger(stdout, stderr)
	}
	// Write info messages and errors to stderr to prevent mixing with normal program output.
	return NewJSONLogger(stderr, nil)
}
