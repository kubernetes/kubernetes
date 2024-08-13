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

package json

import (
	"io"
	"sync/atomic"
	"time"

	"github.com/go-logr/logr"
	"github.com/go-logr/zapr"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"k8s.io/component-base/featuregate"
	logsapi "k8s.io/component-base/logs/api/v1"
)

var (
	// timeNow stubbed out for testing
	timeNow = time.Now
)

type runtime struct {
	v uint32
}

func (r *runtime) ZapV() zapcore.Level {
	// zap levels are inverted: everything with a verbosity >= threshold gets logged.
	return -zapcore.Level(atomic.LoadUint32(&r.v))
}

// Enabled implements the zapcore.LevelEnabler interface.
func (r *runtime) Enabled(level zapcore.Level) bool {
	return level >= r.ZapV()
}

func (r *runtime) SetVerbosityLevel(v uint32) error {
	atomic.StoreUint32(&r.v, v)
	return nil
}

var _ zapcore.LevelEnabler = &runtime{}

// NewJSONLogger creates a new json logr.Logger and its associated
// control interface. The separate error stream is optional and may be nil.
// The encoder config is also optional.
func NewJSONLogger(v logsapi.VerbosityLevel, infoStream, errorStream zapcore.WriteSyncer, encoderConfig *zapcore.EncoderConfig) (logr.Logger, logsapi.RuntimeControl) {
	r := &runtime{v: uint32(v)}

	if encoderConfig == nil {
		encoderConfig = &zapcore.EncoderConfig{
			MessageKey:     "msg",
			CallerKey:      "caller",
			NameKey:        "logger",
			TimeKey:        "ts",
			EncodeTime:     epochMillisTimeEncoder,
			EncodeDuration: zapcore.StringDurationEncoder,
			EncodeCaller:   zapcore.ShortCallerEncoder,
		}
	}

	encoder := zapcore.NewJSONEncoder(*encoderConfig)
	var core zapcore.Core
	if errorStream == nil {
		core = zapcore.NewCore(encoder, infoStream, r)
	} else {
		highPriority := zap.LevelEnablerFunc(func(lvl zapcore.Level) bool {
			return lvl >= zapcore.ErrorLevel && r.Enabled(lvl)
		})
		lowPriority := zap.LevelEnablerFunc(func(lvl zapcore.Level) bool {
			return lvl < zapcore.ErrorLevel && r.Enabled(lvl)
		})
		core = zapcore.NewTee(
			zapcore.NewCore(encoder, errorStream, highPriority),
			zapcore.NewCore(encoder, infoStream, lowPriority),
		)
	}
	l := zap.New(core, zap.WithCaller(true))
	return zapr.NewLoggerWithOptions(l, zapr.LogInfoLevel("v"), zapr.ErrorKey("err")),
		logsapi.RuntimeControl{
			SetVerbosityLevel: r.SetVerbosityLevel,
			Flush: func() {
				_ = l.Sync()
			},
		}
}

func epochMillisTimeEncoder(_ time.Time, enc zapcore.PrimitiveArrayEncoder) {
	nanos := timeNow().UnixNano()
	millis := float64(nanos) / float64(time.Millisecond)
	enc.AppendFloat64(millis)
}

// Factory produces JSON logger instances.
type Factory struct{}

var _ logsapi.LogFormatFactory = Factory{}

func (f Factory) Feature() featuregate.Feature {
	return logsapi.LoggingBetaOptions
}

func (f Factory) Create(c logsapi.LoggingConfiguration, o logsapi.LoggingOptions) (logr.Logger, logsapi.RuntimeControl) {
	// We intentionally avoid all os.File.Sync calls. Output is unbuffered,
	// therefore we don't need to flush, and calling the underlying fsync
	// would just slow down writing.
	//
	// The assumption is that logging only needs to ensure that data gets
	// written to the output stream before the process terminates, but
	// doesn't need to worry about data not being written because of a
	// system crash or powerloss.
	stderr := zapcore.Lock(AddNopSync(o.ErrorStream))
	if c.Options.JSON.SplitStream {
		stdout := zapcore.Lock(AddNopSync(o.InfoStream))
		size := c.Options.JSON.InfoBufferSize.Value()
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
		return NewJSONLogger(c.Verbosity, stdout, stderr, nil)
	}
	// Write info messages and errors to stderr to prevent mixing with normal program output.
	return NewJSONLogger(c.Verbosity, stderr, nil, nil)
}

// AddNoSync adds a NOP Sync implementation.
func AddNopSync(writer io.Writer) zapcore.WriteSyncer {
	return nopSync{Writer: writer}
}

type nopSync struct {
	io.Writer
}

func (f nopSync) Sync() error {
	return nil
}
