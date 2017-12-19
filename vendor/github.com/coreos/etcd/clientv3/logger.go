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
	"log"
	"sync"

	"google.golang.org/grpc/grpclog"
)

// Logger is the logger used by client library.
// It implements grpclog.Logger interface.
type Logger grpclog.Logger

var (
	logger settableLogger
)

type settableLogger struct {
	l  grpclog.Logger
	mu sync.RWMutex
}

func init() {
	// disable client side logs by default
	logger.mu.Lock()
	logger.l = log.New(ioutil.Discard, "", 0)

	// logger has to override the grpclog at initialization so that
	// any changes to the grpclog go through logger with locking
	// instead of through SetLogger
	//
	// now updates only happen through settableLogger.set
	grpclog.SetLogger(&logger)
	logger.mu.Unlock()
}

// SetLogger sets client-side Logger. By default, logs are disabled.
func SetLogger(l Logger) {
	logger.set(l)
}

// GetLogger returns the current logger.
func GetLogger() Logger {
	return logger.get()
}

func (s *settableLogger) set(l Logger) {
	s.mu.Lock()
	logger.l = l
	s.mu.Unlock()
}

func (s *settableLogger) get() Logger {
	s.mu.RLock()
	l := logger.l
	s.mu.RUnlock()
	return l
}

// implement the grpclog.Logger interface

func (s *settableLogger) Fatal(args ...interface{})                 { s.get().Fatal(args...) }
func (s *settableLogger) Fatalf(format string, args ...interface{}) { s.get().Fatalf(format, args...) }
func (s *settableLogger) Fatalln(args ...interface{})               { s.get().Fatalln(args...) }
func (s *settableLogger) Print(args ...interface{})                 { s.get().Print(args...) }
func (s *settableLogger) Printf(format string, args ...interface{}) { s.get().Printf(format, args...) }
func (s *settableLogger) Println(args ...interface{})               { s.get().Println(args...) }
