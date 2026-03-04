/*
 *
 * Copyright 2020 gRPC authors.
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

package channelz

import (
	"fmt"

	"google.golang.org/grpc/grpclog"
)

var logger = grpclog.Component("channelz")

// Info logs and adds a trace event if channelz is on.
func Info(l grpclog.DepthLoggerV2, e Entity, args ...any) {
	AddTraceEvent(l, e, 1, &TraceEvent{
		Desc:     fmt.Sprint(args...),
		Severity: CtInfo,
	})
}

// Infof logs and adds a trace event if channelz is on.
func Infof(l grpclog.DepthLoggerV2, e Entity, format string, args ...any) {
	AddTraceEvent(l, e, 1, &TraceEvent{
		Desc:     fmt.Sprintf(format, args...),
		Severity: CtInfo,
	})
}

// Warning logs and adds a trace event if channelz is on.
func Warning(l grpclog.DepthLoggerV2, e Entity, args ...any) {
	AddTraceEvent(l, e, 1, &TraceEvent{
		Desc:     fmt.Sprint(args...),
		Severity: CtWarning,
	})
}

// Warningf logs and adds a trace event if channelz is on.
func Warningf(l grpclog.DepthLoggerV2, e Entity, format string, args ...any) {
	AddTraceEvent(l, e, 1, &TraceEvent{
		Desc:     fmt.Sprintf(format, args...),
		Severity: CtWarning,
	})
}

// Error logs and adds a trace event if channelz is on.
func Error(l grpclog.DepthLoggerV2, e Entity, args ...any) {
	AddTraceEvent(l, e, 1, &TraceEvent{
		Desc:     fmt.Sprint(args...),
		Severity: CtError,
	})
}

// Errorf logs and adds a trace event if channelz is on.
func Errorf(l grpclog.DepthLoggerV2, e Entity, format string, args ...any) {
	AddTraceEvent(l, e, 1, &TraceEvent{
		Desc:     fmt.Sprintf(format, args...),
		Severity: CtError,
	})
}
