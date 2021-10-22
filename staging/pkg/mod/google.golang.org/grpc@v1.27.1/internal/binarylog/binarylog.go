/*
 *
 * Copyright 2018 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package binarylog implementation binary logging as defined in
// https://github.com/grpc/proposal/blob/master/A16-binary-logging.md.
package binarylog

import (
	"fmt"
	"os"

	"google.golang.org/grpc/grpclog"
)

// Logger is the global binary logger. It can be used to get binary logger for
// each method.
type Logger interface {
	getMethodLogger(methodName string) *MethodLogger
}

// binLogger is the global binary logger for the binary. One of this should be
// built at init time from the configuration (environment variable or flags).
//
// It is used to get a methodLogger for each individual method.
var binLogger Logger

// SetLogger sets the binarg logger.
//
// Only call this at init time.
func SetLogger(l Logger) {
	binLogger = l
}

// GetMethodLogger returns the methodLogger for the given methodName.
//
// methodName should be in the format of "/service/method".
//
// Each methodLogger returned by this method is a new instance. This is to
// generate sequence id within the call.
func GetMethodLogger(methodName string) *MethodLogger {
	if binLogger == nil {
		return nil
	}
	return binLogger.getMethodLogger(methodName)
}

func init() {
	const envStr = "GRPC_BINARY_LOG_FILTER"
	configStr := os.Getenv(envStr)
	binLogger = NewLoggerFromConfigString(configStr)
}

type methodLoggerConfig struct {
	// Max length of header and message.
	hdr, msg uint64
}

type logger struct {
	all      *methodLoggerConfig
	services map[string]*methodLoggerConfig
	methods  map[string]*methodLoggerConfig

	blacklist map[string]struct{}
}

// newEmptyLogger creates an empty logger. The map fields need to be filled in
// using the set* functions.
func newEmptyLogger() *logger {
	return &logger{}
}

// Set method logger for "*".
func (l *logger) setDefaultMethodLogger(ml *methodLoggerConfig) error {
	if l.all != nil {
		return fmt.Errorf("conflicting global rules found")
	}
	l.all = ml
	return nil
}

// Set method logger for "service/*".
//
// New methodLogger with same service overrides the old one.
func (l *logger) setServiceMethodLogger(service string, ml *methodLoggerConfig) error {
	if _, ok := l.services[service]; ok {
		return fmt.Errorf("conflicting service rules for service %v found", service)
	}
	if l.services == nil {
		l.services = make(map[string]*methodLoggerConfig)
	}
	l.services[service] = ml
	return nil
}

// Set method logger for "service/method".
//
// New methodLogger with same method overrides the old one.
func (l *logger) setMethodMethodLogger(method string, ml *methodLoggerConfig) error {
	if _, ok := l.blacklist[method]; ok {
		return fmt.Errorf("conflicting blacklist rules for method %v found", method)
	}
	if _, ok := l.methods[method]; ok {
		return fmt.Errorf("conflicting method rules for method %v found", method)
	}
	if l.methods == nil {
		l.methods = make(map[string]*methodLoggerConfig)
	}
	l.methods[method] = ml
	return nil
}

// Set blacklist method for "-service/method".
func (l *logger) setBlacklist(method string) error {
	if _, ok := l.blacklist[method]; ok {
		return fmt.Errorf("conflicting blacklist rules for method %v found", method)
	}
	if _, ok := l.methods[method]; ok {
		return fmt.Errorf("conflicting method rules for method %v found", method)
	}
	if l.blacklist == nil {
		l.blacklist = make(map[string]struct{})
	}
	l.blacklist[method] = struct{}{}
	return nil
}

// getMethodLogger returns the methodLogger for the given methodName.
//
// methodName should be in the format of "/service/method".
//
// Each methodLogger returned by this method is a new instance. This is to
// generate sequence id within the call.
func (l *logger) getMethodLogger(methodName string) *MethodLogger {
	s, m, err := parseMethodName(methodName)
	if err != nil {
		grpclog.Infof("binarylogging: failed to parse %q: %v", methodName, err)
		return nil
	}
	if ml, ok := l.methods[s+"/"+m]; ok {
		return newMethodLogger(ml.hdr, ml.msg)
	}
	if _, ok := l.blacklist[s+"/"+m]; ok {
		return nil
	}
	if ml, ok := l.services[s]; ok {
		return newMethodLogger(ml.hdr, ml.msg)
	}
	if l.all == nil {
		return nil
	}
	return newMethodLogger(l.all.hdr, l.all.msg)
}
