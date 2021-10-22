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

// Package glogger defines glog-based logging for grpc.
// Importing this package will install glog as the logger used by grpclog.
package glogger

import (
	"fmt"

	"github.com/golang/glog"
	"google.golang.org/grpc/grpclog"
)

func init() {
	grpclog.SetLoggerV2(&glogger{})
}

type glogger struct{}

func (g *glogger) Info(args ...interface{}) {
	glog.InfoDepth(2, args...)
}

func (g *glogger) Infoln(args ...interface{}) {
	glog.InfoDepth(2, fmt.Sprintln(args...))
}

func (g *glogger) Infof(format string, args ...interface{}) {
	glog.InfoDepth(2, fmt.Sprintf(format, args...))
}

func (g *glogger) Warning(args ...interface{}) {
	glog.WarningDepth(2, args...)
}

func (g *glogger) Warningln(args ...interface{}) {
	glog.WarningDepth(2, fmt.Sprintln(args...))
}

func (g *glogger) Warningf(format string, args ...interface{}) {
	glog.WarningDepth(2, fmt.Sprintf(format, args...))
}

func (g *glogger) Error(args ...interface{}) {
	glog.ErrorDepth(2, args...)
}

func (g *glogger) Errorln(args ...interface{}) {
	glog.ErrorDepth(2, fmt.Sprintln(args...))
}

func (g *glogger) Errorf(format string, args ...interface{}) {
	glog.ErrorDepth(2, fmt.Sprintf(format, args...))
}

func (g *glogger) Fatal(args ...interface{}) {
	glog.FatalDepth(2, args...)
}

func (g *glogger) Fatalln(args ...interface{}) {
	glog.FatalDepth(2, fmt.Sprintln(args...))
}

func (g *glogger) Fatalf(format string, args ...interface{}) {
	glog.FatalDepth(2, fmt.Sprintf(format, args...))
}

func (g *glogger) V(l int) bool {
	return bool(glog.V(glog.Level(l)))
}
