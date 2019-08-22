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

import "google.golang.org/grpc/grpclog"

// Logger defines logging interface.
// TODO: deprecate in v3.5.
type Logger interface {
	grpclog.LoggerV2

	// Lvl returns logger if logger's verbosity level >= "lvl".
	// Otherwise, logger that discards everything.
	Lvl(lvl int) grpclog.LoggerV2
}

// assert that "defaultLogger" satisfy "Logger" interface
var _ Logger = &defaultLogger{}

// NewLogger wraps "grpclog.LoggerV2" that implements "Logger" interface.
//
// For example:
//
//  var defaultLogger Logger
//  g := grpclog.NewLoggerV2WithVerbosity(os.Stderr, os.Stderr, os.Stderr, 4)
//  defaultLogger = NewLogger(g)
//
func NewLogger(g grpclog.LoggerV2) Logger { return &defaultLogger{g: g} }

type defaultLogger struct {
	g grpclog.LoggerV2
}

func (l *defaultLogger) Info(args ...interface{})                    { l.g.Info(args...) }
func (l *defaultLogger) Infoln(args ...interface{})                  { l.g.Info(args...) }
func (l *defaultLogger) Infof(format string, args ...interface{})    { l.g.Infof(format, args...) }
func (l *defaultLogger) Warning(args ...interface{})                 { l.g.Warning(args...) }
func (l *defaultLogger) Warningln(args ...interface{})               { l.g.Warning(args...) }
func (l *defaultLogger) Warningf(format string, args ...interface{}) { l.g.Warningf(format, args...) }
func (l *defaultLogger) Error(args ...interface{})                   { l.g.Error(args...) }
func (l *defaultLogger) Errorln(args ...interface{})                 { l.g.Error(args...) }
func (l *defaultLogger) Errorf(format string, args ...interface{})   { l.g.Errorf(format, args...) }
func (l *defaultLogger) Fatal(args ...interface{})                   { l.g.Fatal(args...) }
func (l *defaultLogger) Fatalln(args ...interface{})                 { l.g.Fatal(args...) }
func (l *defaultLogger) Fatalf(format string, args ...interface{})   { l.g.Fatalf(format, args...) }
func (l *defaultLogger) V(lvl int) bool                              { return l.g.V(lvl) }
func (l *defaultLogger) Lvl(lvl int) grpclog.LoggerV2 {
	if l.g.V(lvl) {
		return l
	}
	return &discardLogger{}
}
