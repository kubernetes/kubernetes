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
)

var (
	// JSONLogger is global json log format logr
	JSONLogger logr.Logger

	// timeNow stubbed out for testing
	timeNow = time.Now
)

func init() {
	JSONLogger = NewJSONLogger(zapcore.Lock(os.Stdout))
}

// NewJSONLogger creates a new json logr.Logger using the given Zap Logger to log.
func NewJSONLogger(w zapcore.WriteSyncer) logr.Logger {
	encoder := zapcore.NewJSONEncoder(encoderConfig)
	// The log level intentionally gets set as low as possible to
	// ensure that all messages are printed when this logger gets
	// called by klog. The actual verbosity check happens in klog.
	core := zapcore.NewCore(encoder, zapcore.AddSync(w), zapcore.Level(-127))
	l := zap.New(core, zap.WithCaller(true))
	return zapr.NewLoggerWithOptions(l, zapr.LogInfoLevel("v"), zapr.ErrorKey("err"))
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
