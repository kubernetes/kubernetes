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
	"slices"
	"time"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// CreateDefaultZapLogger creates a logger with default zap configuration
func CreateDefaultZapLogger(level zapcore.Level) (*zap.Logger, error) {
	lcfg := DefaultZapLoggerConfig
	lcfg.Level = zap.NewAtomicLevelAt(level)
	c, err := lcfg.Build()
	if err != nil {
		return nil, err
	}
	return c, nil
}

// DefaultZapLoggerConfig defines default zap logger configuration.
var DefaultZapLoggerConfig = zap.Config{
	Level: zap.NewAtomicLevelAt(ConvertToZapLevel(DefaultLogLevel)),

	Development: false,
	Sampling: &zap.SamplingConfig{
		Initial:    100,
		Thereafter: 100,
	},

	Encoding: DefaultLogFormat,

	// copied from "zap.NewProductionEncoderConfig" with some updates
	EncoderConfig: zapcore.EncoderConfig{
		TimeKey:       "ts",
		LevelKey:      "level",
		NameKey:       "logger",
		CallerKey:     "caller",
		MessageKey:    "msg",
		StacktraceKey: "stacktrace",
		LineEnding:    zapcore.DefaultLineEnding,
		EncodeLevel:   zapcore.LowercaseLevelEncoder,

		// Custom EncodeTime function to ensure we match format and precision of historic capnslog timestamps
		EncodeTime: func(t time.Time, enc zapcore.PrimitiveArrayEncoder) {
			enc.AppendString(t.Format("2006-01-02T15:04:05.000000Z0700"))
		},

		EncodeDuration: zapcore.StringDurationEncoder,
		EncodeCaller:   zapcore.ShortCallerEncoder,
	},

	// Use "/dev/null" to discard all
	OutputPaths:      []string{"stderr"},
	ErrorOutputPaths: []string{"stderr"},
}

// MergeOutputPaths merges logging output paths, resolving conflicts.
func MergeOutputPaths(cfg zap.Config) zap.Config {
	cfg.OutputPaths = mergePaths(cfg.OutputPaths)
	cfg.ErrorOutputPaths = mergePaths(cfg.ErrorOutputPaths)
	return cfg
}

func mergePaths(old []string) []string {
	if len(old) == 0 {
		// the original implementation ensures the result is non-nil
		return []string{}
	}
	// use "/dev/null" to discard all
	if slices.Contains(old, "/dev/null") {
		return []string{"/dev/null"}
	}
	// clone a new one; don't modify the original, in case it matters.
	dup := slices.Clone(old)
	slices.Sort(dup)
	return slices.Compact(dup)
}
