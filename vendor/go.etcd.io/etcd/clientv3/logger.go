// Copyright 2016 The etcd Authors
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

package clientv3

import (
	"io/ioutil"
	"sync"

	"go.etcd.io/etcd/pkg/logutil"

	"google.golang.org/grpc/grpclog"
)

var (
	lgMu sync.RWMutex
	lg   logutil.Logger
)

type settableLogger struct {
	l  grpclog.LoggerV2
	mu sync.RWMutex
}

func init() {
	// disable client side logs by default
	lg = &settableLogger{}
	SetLogger(grpclog.NewLoggerV2(ioutil.Discard, ioutil.Discard, ioutil.Discard))
}

// SetLogger sets client-side Logger.
func SetLogger(l grpclog.LoggerV2) {
	lgMu.Lock()
	lg = logutil.NewLogger(l)
	// override grpclog so that any changes happen with locking
	grpclog.SetLoggerV2(lg)
	lgMu.Unlock()
}

// GetLogger returns the current logutil.Logger.
func GetLogger() logutil.Logger {
	lgMu.RLock()
	l := lg
	lgMu.RUnlock()
	return l
}

// NewLogger returns a new Logger with logutil.Logger.
func NewLogger(gl grpclog.LoggerV2) logutil.Logger {
	return &settableLogger{l: gl}
}

func (s *settableLogger) get() grpclog.LoggerV2 {
	s.mu.RLock()
	l := s.l
	s.mu.RUnlock()
	return l
}

// implement the grpclog.LoggerV2 interface

func (s *settableLogger) Info(args ...interface{})                 { s.get().Info(args...) }
func (s *settableLogger) Infof(format string, args ...interface{}) { s.get().Infof(format, args...) }
func (s *settableLogger) Infoln(args ...interface{})               { s.get().Infoln(args...) }
func (s *settableLogger) Warning(args ...interface{})              { s.get().Warning(args...) }
func (s *settableLogger) Warningf(format string, args ...interface{}) {
	s.get().Warningf(format, args...)
}
func (s *settableLogger) Warningln(args ...interface{}) { s.get().Warningln(args...) }
func (s *settableLogger) Error(args ...interface{})     { s.get().Error(args...) }
func (s *settableLogger) Errorf(format string, args ...interface{}) {
	s.get().Errorf(format, args...)
}
func (s *settableLogger) Errorln(args ...interface{})               { s.get().Errorln(args...) }
func (s *settableLogger) Fatal(args ...interface{})                 { s.get().Fatal(args...) }
func (s *settableLogger) Fatalf(format string, args ...interface{}) { s.get().Fatalf(format, args...) }
func (s *settableLogger) Fatalln(args ...interface{})               { s.get().Fatalln(args...) }
func (s *settableLogger) Print(args ...interface{})                 { s.get().Info(args...) }
func (s *settableLogger) Printf(format string, args ...interface{}) { s.get().Infof(format, args...) }
func (s *settableLogger) Println(args ...interface{})               { s.get().Infoln(args...) }
func (s *settableLogger) V(l int) bool                              { return s.get().V(l) }
func (s *settableLogger) Lvl(lvl int) grpclog.LoggerV2 {
	s.mu.RLock()
	l := s.l
	s.mu.RUnlock()
	if l.V(lvl) {
		return s
	}
	return logutil.NewDiscardLogger()
}
