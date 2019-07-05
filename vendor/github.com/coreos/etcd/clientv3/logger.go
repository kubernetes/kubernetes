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

	"google.golang.org/grpc/grpclog"
)

// Logger is the logger used by client library.
// It implements grpclog.LoggerV2 interface.
type Logger interface {
	grpclog.LoggerV2

	// Lvl returns logger if logger's verbosity level >= "lvl".
	// Otherwise, logger that discards all logs.
	Lvl(lvl int) Logger

	// to satisfy capnslog

	Print(args ...interface{})
	Printf(format string, args ...interface{})
	Println(args ...interface{})
}

var (
	loggerMu sync.RWMutex
	logger   Logger
)

type settableLogger struct {
	l  grpclog.LoggerV2
	mu sync.RWMutex
}

func init() {
	// disable client side logs by default
	logger = &settableLogger{}
	SetLogger(grpclog.NewLoggerV2(ioutil.Discard, ioutil.Discard, ioutil.Discard))
}

// SetLogger sets client-side Logger.
func SetLogger(l grpclog.LoggerV2) {
	loggerMu.Lock()
	logger = NewLogger(l)
	// override grpclog so that any changes happen with locking
	grpclog.SetLoggerV2(logger)
	loggerMu.Unlock()
}

// GetLogger returns the current logger.
func GetLogger() Logger {
	loggerMu.RLock()
	l := logger
	loggerMu.RUnlock()
	return l
}

// NewLogger returns a new Logger with grpclog.LoggerV2.
func NewLogger(gl grpclog.LoggerV2) Logger {
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
func (s *settableLogger) Lvl(lvl int) Logger {
	s.mu.RLock()
	l := s.l
	s.mu.RUnlock()
	if l.V(lvl) {
		return s
	}
	return &noLogger{}
}

type noLogger struct{}

func (*noLogger) Info(args ...interface{})                    {}
func (*noLogger) Infof(format string, args ...interface{})    {}
func (*noLogger) Infoln(args ...interface{})                  {}
func (*noLogger) Warning(args ...interface{})                 {}
func (*noLogger) Warningf(format string, args ...interface{}) {}
func (*noLogger) Warningln(args ...interface{})               {}
func (*noLogger) Error(args ...interface{})                   {}
func (*noLogger) Errorf(format string, args ...interface{})   {}
func (*noLogger) Errorln(args ...interface{})                 {}
func (*noLogger) Fatal(args ...interface{})                   {}
func (*noLogger) Fatalf(format string, args ...interface{})   {}
func (*noLogger) Fatalln(args ...interface{})                 {}
func (*noLogger) Print(args ...interface{})                   {}
func (*noLogger) Printf(format string, args ...interface{})   {}
func (*noLogger) Println(args ...interface{})                 {}
func (*noLogger) V(l int) bool                                { return false }
func (ng *noLogger) Lvl(lvl int) Logger                       { return ng }
