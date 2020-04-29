// Copyright The OpenTelemetry Authors
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

package trace

import (
	"math/rand"
	"sync"

	"go.opentelemetry.io/otel/api/core"
	"go.opentelemetry.io/otel/sdk/trace/internal"
)

type defaultIDGenerator struct {
	sync.Mutex
	randSource *rand.Rand
}

var _ internal.IDGenerator = &defaultIDGenerator{}

// NewSpanID returns a non-zero span ID from a randomly-chosen sequence.
func (gen *defaultIDGenerator) NewSpanID() core.SpanID {
	gen.Lock()
	defer gen.Unlock()
	sid := core.SpanID{}
	gen.randSource.Read(sid[:])
	return sid
}

// NewTraceID returns a non-zero trace ID from a randomly-chosen sequence.
// mu should be held while this function is called.
func (gen *defaultIDGenerator) NewTraceID() core.TraceID {
	gen.Lock()
	defer gen.Unlock()
	tid := core.TraceID{}
	gen.randSource.Read(tid[:])
	return tid
}
