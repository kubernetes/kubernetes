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
	"github.com/coreos/pkg/capnslog"
	"google.golang.org/grpc/grpclog"
)

// assert that "packageLogger" satisfy "Logger" interface
var _ Logger = &packageLogger{}

// NewPackageLogger wraps "*capnslog.PackageLogger" that implements "Logger" interface.
//
// For example:
//
//  var defaultLogger Logger
//  defaultLogger = NewPackageLogger("go.etcd.io/etcd", "snapshot")
//
func NewPackageLogger(repo, pkg string) Logger {
	return &packageLogger{p: capnslog.NewPackageLogger(repo, pkg)}
}

type packageLogger struct {
	p *capnslog.PackageLogger
}

func (l *packageLogger) Info(args ...interface{})                    { l.p.Info(args...) }
func (l *packageLogger) Infoln(args ...interface{})                  { l.p.Info(args...) }
func (l *packageLogger) Infof(format string, args ...interface{})    { l.p.Infof(format, args...) }
func (l *packageLogger) Warning(args ...interface{})                 { l.p.Warning(args...) }
func (l *packageLogger) Warningln(args ...interface{})               { l.p.Warning(args...) }
func (l *packageLogger) Warningf(format string, args ...interface{}) { l.p.Warningf(format, args...) }
func (l *packageLogger) Error(args ...interface{})                   { l.p.Error(args...) }
func (l *packageLogger) Errorln(args ...interface{})                 { l.p.Error(args...) }
func (l *packageLogger) Errorf(format string, args ...interface{})   { l.p.Errorf(format, args...) }
func (l *packageLogger) Fatal(args ...interface{})                   { l.p.Fatal(args...) }
func (l *packageLogger) Fatalln(args ...interface{})                 { l.p.Fatal(args...) }
func (l *packageLogger) Fatalf(format string, args ...interface{})   { l.p.Fatalf(format, args...) }
func (l *packageLogger) V(lvl int) bool {
	return l.p.LevelAt(capnslog.LogLevel(lvl))
}
func (l *packageLogger) Lvl(lvl int) grpclog.LoggerV2 {
	if l.p.LevelAt(capnslog.LogLevel(lvl)) {
		return l
	}
	return &discardLogger{}
}
