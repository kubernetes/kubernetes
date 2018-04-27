/*
 *
 * Copyright 2015 gRPC authors.
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

package grpclog

// Logger mimics golang's standard Logger as an interface.
// Deprecated: use LoggerV2.
type Logger interface {
	Fatal(args ...interface{})
	Fatalf(format string, args ...interface{})
	Fatalln(args ...interface{})
	Print(args ...interface{})
	Printf(format string, args ...interface{})
	Println(args ...interface{})
}

// SetLogger sets the logger that is used in grpc. Call only from
// init() functions.
// Deprecated: use SetLoggerV2.
func SetLogger(l Logger) {
	logger = &loggerWrapper{Logger: l}
}

// loggerWrapper wraps Logger into a LoggerV2.
type loggerWrapper struct {
	Logger
}

func (g *loggerWrapper) Info(args ...interface{}) {
	g.Logger.Print(args...)
}

func (g *loggerWrapper) Infoln(args ...interface{}) {
	g.Logger.Println(args...)
}

func (g *loggerWrapper) Infof(format string, args ...interface{}) {
	g.Logger.Printf(format, args...)
}

func (g *loggerWrapper) Warning(args ...interface{}) {
	g.Logger.Print(args...)
}

func (g *loggerWrapper) Warningln(args ...interface{}) {
	g.Logger.Println(args...)
}

func (g *loggerWrapper) Warningf(format string, args ...interface{}) {
	g.Logger.Printf(format, args...)
}

func (g *loggerWrapper) Error(args ...interface{}) {
	g.Logger.Print(args...)
}

func (g *loggerWrapper) Errorln(args ...interface{}) {
	g.Logger.Println(args...)
}

func (g *loggerWrapper) Errorf(format string, args ...interface{}) {
	g.Logger.Printf(format, args...)
}

func (g *loggerWrapper) V(l int) bool {
	// Returns true for all verbose level.
	return true
}
