// Copyright 2019 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package logutil

import (
	"fmt"

	"github.com/coreos/pkg/capnslog"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

var DefaultLogLevel = "info"

// ConvertToZapLevel converts log level string to zapcore.Level.
func ConvertToZapLevel(lvl string) zapcore.Level {
	switch lvl {
	case "debug":
		return zap.DebugLevel
	case "info":
		return zap.InfoLevel
	case "warn":
		return zap.WarnLevel
	case "error":
		return zap.ErrorLevel
	case "dpanic":
		return zap.DPanicLevel
	case "panic":
		return zap.PanicLevel
	case "fatal":
		return zap.FatalLevel
	default:
		panic(fmt.Sprintf("unknown level %q", lvl))
	}
}

// ConvertToCapnslogLogLevel convert log level string to capnslog.LogLevel.
// TODO: deprecate this in 3.5
func ConvertToCapnslogLogLevel(lvl string) capnslog.LogLevel {
	switch lvl {
	case "debug":
		return capnslog.DEBUG
	case "info":
		return capnslog.INFO
	case "warn":
		return capnslog.WARNING
	case "error":
		return capnslog.ERROR
	case "dpanic":
		return capnslog.CRITICAL
	case "panic":
		return capnslog.CRITICAL
	case "fatal":
		return capnslog.CRITICAL
	default:
		panic(fmt.Sprintf("unknown level %q", lvl))
	}
}
