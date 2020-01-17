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
	"errors"

	"go.etcd.io/etcd/raft"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// NewRaftLogger builds "raft.Logger" from "*zap.Config".
func NewRaftLogger(lcfg *zap.Config) (raft.Logger, error) {
	if lcfg == nil {
		return nil, errors.New("nil zap.Config")
	}
	lg, err := lcfg.Build(zap.AddCallerSkip(1)) // to annotate caller outside of "logutil"
	if err != nil {
		return nil, err
	}
	return &zapRaftLogger{lg: lg, sugar: lg.Sugar()}, nil
}

// NewRaftLoggerZap converts "*zap.Logger" to "raft.Logger".
func NewRaftLoggerZap(lg *zap.Logger) raft.Logger {
	return &zapRaftLogger{lg: lg, sugar: lg.Sugar()}
}

// NewRaftLoggerFromZapCore creates "raft.Logger" from "zap.Core"
// and "zapcore.WriteSyncer".
func NewRaftLoggerFromZapCore(cr zapcore.Core, syncer zapcore.WriteSyncer) raft.Logger {
	// "AddCallerSkip" to annotate caller outside of "logutil"
	lg := zap.New(cr, zap.AddCaller(), zap.AddCallerSkip(1), zap.ErrorOutput(syncer))
	return &zapRaftLogger{lg: lg, sugar: lg.Sugar()}
}

type zapRaftLogger struct {
	lg    *zap.Logger
	sugar *zap.SugaredLogger
}

func (zl *zapRaftLogger) Debug(args ...interface{}) {
	zl.sugar.Debug(args...)
}

func (zl *zapRaftLogger) Debugf(format string, args ...interface{}) {
	zl.sugar.Debugf(format, args...)
}

func (zl *zapRaftLogger) Error(args ...interface{}) {
	zl.sugar.Error(args...)
}

func (zl *zapRaftLogger) Errorf(format string, args ...interface{}) {
	zl.sugar.Errorf(format, args...)
}

func (zl *zapRaftLogger) Info(args ...interface{}) {
	zl.sugar.Info(args...)
}

func (zl *zapRaftLogger) Infof(format string, args ...interface{}) {
	zl.sugar.Infof(format, args...)
}

func (zl *zapRaftLogger) Warning(args ...interface{}) {
	zl.sugar.Warn(args...)
}

func (zl *zapRaftLogger) Warningf(format string, args ...interface{}) {
	zl.sugar.Warnf(format, args...)
}

func (zl *zapRaftLogger) Fatal(args ...interface{}) {
	zl.sugar.Fatal(args...)
}

func (zl *zapRaftLogger) Fatalf(format string, args ...interface{}) {
	zl.sugar.Fatalf(format, args...)
}

func (zl *zapRaftLogger) Panic(args ...interface{}) {
	zl.sugar.Panic(args...)
}

func (zl *zapRaftLogger) Panicf(format string, args ...interface{}) {
	zl.sugar.Panicf(format, args...)
}
