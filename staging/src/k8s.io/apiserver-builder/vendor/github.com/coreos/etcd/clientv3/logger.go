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
	"log"
	"os"
	"sync"

	"google.golang.org/grpc/grpclog"
)

type Logger grpclog.Logger

var (
	logger settableLogger
)

type settableLogger struct {
	l  grpclog.Logger
	mu sync.RWMutex
}

func init() {
	// use go's standard logger by default like grpc
	logger.mu.Lock()
	logger.l = log.New(os.Stderr, "", log.LstdFlags)
	grpclog.SetLogger(&logger)
	logger.mu.Unlock()
}

func (s *settableLogger) Set(l Logger) {
	s.mu.Lock()
	logger.l = l
	s.mu.Unlock()
}

func (s *settableLogger) Get() Logger {
	s.mu.RLock()
	l := logger.l
	s.mu.RUnlock()
	return l
}

// implement the grpclog.Logger interface

func (s *settableLogger) Fatal(args ...interface{})                 { s.Get().Fatal(args...) }
func (s *settableLogger) Fatalf(format string, args ...interface{}) { s.Get().Fatalf(format, args...) }
func (s *settableLogger) Fatalln(args ...interface{})               { s.Get().Fatalln(args...) }
func (s *settableLogger) Print(args ...interface{})                 { s.Get().Print(args...) }
func (s *settableLogger) Printf(format string, args ...interface{}) { s.Get().Printf(format, args...) }
func (s *settableLogger) Println(args ...interface{})               { s.Get().Println(args...) }
