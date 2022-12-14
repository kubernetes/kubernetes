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
	"google.golang.org/grpc/internal/grpcutil"
)

// Logger is the global binary logger. It can be used to get binary logger for
// each method.
type Logger interface {
	GetMethodLogger(methodName string) MethodLogger
}

// binLogger is the global binary logger for the binary. One of this should be
// built at init time from the configuration (environment variable or flags).
//
// It is used to get a MethodLogger for each individual method.
var binLogger Logger

var grpclogLogger = grpclog.Component("binarylog")

// SetLogger sets the binary logger.
//
// Only call this at init time.
func SetLogger(l Logger) {
	binLogger = l
}

// GetLogger gets the binary logger.
//
// Only call this at init time.
func GetLogger() Logger {
	return binLogger
}

// GetMethodLogger returns the MethodLogger for the given methodName.
//
// methodName should be in the format of "/service/method".
//
// Each MethodLogger returned by this method is a new instance. This is to
// generate sequence id within the call.
func GetMethodLogger(methodName string) MethodLogger {
	if binLogger == nil {
		return nil
	}
	return binLogger.GetMethodLogger(methodName)
}

func init() {
	const envStr = "GRPC_BINARY_LOG_FILTER"
	configStr := os.Getenv(envStr)
	binLogger = NewLoggerFromConfigString(configStr)
}

// MethodLoggerConfig contains the setting for logging behavior of a method
// logger. Currently, it contains the max length of header and message.
type MethodLoggerConfig struct {
	// Max length of header and message.
	Header, Message uint64
}

// LoggerConfig contains the config for loggers to create method loggers.
type LoggerConfig struct {
	All      *MethodLoggerConfig
	Services map[string]*MethodLoggerConfig
	Methods  map[string]*MethodLoggerConfig

	Blacklist map[string]struct{}
}

type logger struct {
	config LoggerConfig
}

// NewLoggerFromConfig builds a logger with the given LoggerConfig.
func NewLoggerFromConfig(config LoggerConfig) Logger {
	return &logger{config: config}
}

// newEmptyLogger creates an empty logger. The map fields need to be filled in
// using the set* functions.
func newEmptyLogger() *logger {
	return &logger{}
}

// Set method logger for "*".
func (l *logger) setDefaultMethodLogger(ml *MethodLoggerConfig) error {
	if l.config.All != nil {
		return fmt.Errorf("conflicting global rules found")
	}
	l.config.All = ml
	return nil
}

// Set method logger for "service/*".
//
// New MethodLogger with same service overrides the old one.
func (l *logger) setServiceMethodLogger(service string, ml *MethodLoggerConfig) error {
	if _, ok := l.config.Services[service]; ok {
		return fmt.Errorf("conflicting service rules for service %v found", service)
	}
	if l.config.Services == nil {
		l.config.Services = make(map[string]*MethodLoggerConfig)
	}
	l.config.Services[service] = ml
	return nil
}

// Set method logger for "service/method".
//
// New MethodLogger with same method overrides the old one.
func (l *logger) setMethodMethodLogger(method string, ml *MethodLoggerConfig) error {
	if _, ok := l.config.Blacklist[method]; ok {
		return fmt.Errorf("conflicting blacklist rules for method %v found", method)
	}
	if _, ok := l.config.Methods[method]; ok {
		return fmt.Errorf("conflicting method rules for method %v found", method)
	}
	if l.config.Methods == nil {
		l.config.Methods = make(map[string]*MethodLoggerConfig)
	}
	l.config.Methods[method] = ml
	return nil
}

// Set blacklist method for "-service/method".
func (l *logger) setBlacklist(method string) error {
	if _, ok := l.config.Blacklist[method]; ok {
		return fmt.Errorf("conflicting blacklist rules for method %v found", method)
	}
	if _, ok := l.config.Methods[method]; ok {
		return fmt.Errorf("conflicting method rules for method %v found", method)
	}
	if l.config.Blacklist == nil {
		l.config.Blacklist = make(map[string]struct{})
	}
	l.config.Blacklist[method] = struct{}{}
	return nil
}

// getMethodLogger returns the MethodLogger for the given methodName.
//
// methodName should be in the format of "/service/method".
//
// Each MethodLogger returned by this method is a new instance. This is to
// generate sequence id within the call.
func (l *logger) GetMethodLogger(methodName string) MethodLogger {
	s, m, err := grpcutil.ParseMethod(methodName)
	if err != nil {
		grpclogLogger.Infof("binarylogging: failed to parse %q: %v", methodName, err)
		return nil
	}
	if ml, ok := l.config.Methods[s+"/"+m]; ok {
		return NewTruncatingMethodLogger(ml.Header, ml.Message)
	}
	if _, ok := l.config.Blacklist[s+"/"+m]; ok {
		return nil
	}
	if ml, ok := l.config.Services[s]; ok {
		return NewTruncatingMethodLogger(ml.Header, ml.Message)
	}
	if l.config.All == nil {
		return nil
	}
	return NewTruncatingMethodLogger(l.config.All.Header, l.config.All.Message)
}
