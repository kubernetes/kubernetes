// Copyright 2018 The etcd Authors
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
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"google.golang.org/grpc/grpclog"
)

// NewGRPCLoggerV2 converts "*zap.Logger" to "grpclog.LoggerV2".
// It discards all INFO level logging in gRPC, if debug level
// is not enabled in "*zap.Logger".
func NewGRPCLoggerV2(lcfg zap.Config) (grpclog.LoggerV2, error) {
	lg, err := lcfg.Build(zap.AddCallerSkip(1)) // to annotate caller outside of "logutil"
	if err != nil {
		return nil, err
	}
	return &zapGRPCLogger{lg: lg, sugar: lg.Sugar()}, nil
}

// NewGRPCLoggerV2FromZapCore creates "grpclog.LoggerV2" from "zap.Core"
// and "zapcore.WriteSyncer". It discards all INFO level logging in gRPC,
// if debug level is not enabled in "*zap.Logger".
func NewGRPCLoggerV2FromZapCore(cr zapcore.Core, syncer zapcore.WriteSyncer) grpclog.LoggerV2 {
	// "AddCallerSkip" to annotate caller outside of "logutil"
	lg := zap.New(cr, zap.AddCaller(), zap.AddCallerSkip(1), zap.ErrorOutput(syncer)).Named("grpc")
	return &zapGRPCLogger{lg: lg, sugar: lg.Sugar()}
}

type zapGRPCLogger struct {
	lg    *zap.Logger
	sugar *zap.SugaredLogger
}

func (zl *zapGRPCLogger) Info(args ...interface{}) {
	if !zl.lg.Core().Enabled(zapcore.DebugLevel) {
		return
	}
	zl.sugar.Info(args...)
}

func (zl *zapGRPCLogger) Infoln(args ...interface{}) {
	if !zl.lg.Core().Enabled(zapcore.DebugLevel) {
		return
	}
	zl.sugar.Info(args...)
}

func (zl *zapGRPCLogger) Infof(format string, args ...interface{}) {
	if !zl.lg.Core().Enabled(zapcore.DebugLevel) {
		return
	}
	zl.sugar.Infof(format, args...)
}

func (zl *zapGRPCLogger) Warning(args ...interface{}) {
	zl.sugar.Warn(args...)
}

func (zl *zapGRPCLogger) Warningln(args ...interface{}) {
	zl.sugar.Warn(args...)
}

func (zl *zapGRPCLogger) Warningf(format string, args ...interface{}) {
	zl.sugar.Warnf(format, args...)
}

func (zl *zapGRPCLogger) Error(args ...interface{}) {
	zl.sugar.Error(args...)
}

func (zl *zapGRPCLogger) Errorln(args ...interface{}) {
	zl.sugar.Error(args...)
}

func (zl *zapGRPCLogger) Errorf(format string, args ...interface{}) {
	zl.sugar.Errorf(format, args...)
}

func (zl *zapGRPCLogger) Fatal(args ...interface{}) {
	zl.sugar.Fatal(args...)
}

func (zl *zapGRPCLogger) Fatalln(args ...interface{}) {
	zl.sugar.Fatal(args...)
}

func (zl *zapGRPCLogger) Fatalf(format string, args ...interface{}) {
	zl.sugar.Fatalf(format, args...)
}

func (zl *zapGRPCLogger) V(l int) bool {
	// infoLog == 0
	if l <= 0 { // debug level, then we ignore info level in gRPC
		return !zl.lg.Core().Enabled(zapcore.DebugLevel)
	}
	return true
}
