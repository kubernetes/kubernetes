//go:build grpcnotrace

/*
 *
 * Copyright 2024 gRPC authors.
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

package grpc

// grpcnotrace can be used to avoid importing golang.org/x/net/trace, which in
// turn enables binaries using gRPC-Go for dead code elimination, which can
// yield 10-15% improvements in binary size when tracing is not needed.

import (
	"context"
	"fmt"
)

type notrace struct{}

func (notrace) LazyLog(x fmt.Stringer, sensitive bool) {}
func (notrace) LazyPrintf(format string, a ...any)     {}
func (notrace) SetError()                              {}
func (notrace) SetRecycler(f func(any))                {}
func (notrace) SetTraceInfo(traceID, spanID uint64)    {}
func (notrace) SetMaxEvents(m int)                     {}
func (notrace) Finish()                                {}

func newTrace(family, title string) traceLog {
	return notrace{}
}

func newTraceContext(ctx context.Context, tr traceLog) context.Context {
	return ctx
}

func newTraceEventLog(family, title string) traceEventLog {
	return nil
}
