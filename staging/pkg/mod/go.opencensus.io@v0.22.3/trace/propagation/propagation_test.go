// Copyright 2017, OpenCensus Authors
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

package propagation

import (
	"bytes"
	"fmt"
	"testing"

	. "go.opencensus.io/trace"
)

func TestBinary(t *testing.T) {
	tid := TraceID{0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f}
	sid := SpanID{0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68}
	b := []byte{
		0, 0, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 1, 97, 98, 99, 100,
		101, 102, 103, 104, 2, 1,
	}
	if b2 := Binary(SpanContext{
		TraceID:      tid,
		SpanID:       sid,
		TraceOptions: 1,
	}); !bytes.Equal(b2, b) {
		t.Errorf("Binary: got serialization %02x want %02x", b2, b)
	}

	sc, ok := FromBinary(b)
	if !ok {
		t.Errorf("FromBinary: got ok==%t, want true", ok)
	}
	if got := sc.TraceID; got != tid {
		t.Errorf("FromBinary: got trace ID %s want %s", got, tid)
	}
	if got := sc.SpanID; got != sid {
		t.Errorf("FromBinary: got span ID %s want %s", got, sid)
	}

	b[0] = 1
	sc, ok = FromBinary(b)
	if ok {
		t.Errorf("FromBinary: decoding bytes containing an unsupported version: got ok==%t want false", ok)
	}

	b = []byte{0, 1, 97, 98, 99, 100, 101, 102, 103, 104, 2, 1}
	sc, ok = FromBinary(b)
	if ok {
		t.Errorf("FromBinary: decoding bytes without a TraceID: got ok==%t want false", ok)
	}

	if b := Binary(SpanContext{}); b != nil {
		t.Errorf("Binary(SpanContext{}): got serialization %02x want nil", b)
	}
}

func TestFromBinary(t *testing.T) {
	validData := []byte{0, 0, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 1, 97, 98, 99, 100, 101, 102, 103, 104, 2, 1}
	tests := []struct {
		name        string
		data        []byte
		wantTraceID TraceID
		wantSpanID  SpanID
		wantOpts    TraceOptions
		wantOk      bool
	}{
		{
			name:   "nil data",
			data:   nil,
			wantOk: false,
		},
		{
			name:   "short data",
			data:   []byte{0, 0, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77},
			wantOk: false,
		},
		{
			name:   "wrong field number",
			data:   []byte{0, 1, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77},
			wantOk: false,
		},
		{
			name:        "valid data",
			data:        validData,
			wantTraceID: TraceID{64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79},
			wantSpanID:  SpanID{97, 98, 99, 100, 101, 102, 103, 104},
			wantOpts:    1,
			wantOk:      true,
		},
	}
	for _, tt := range tests {
		sc, gotOk := FromBinary(tt.data)
		gotTraceID, gotSpanID, gotOpts := sc.TraceID, sc.SpanID, sc.TraceOptions
		if gotTraceID != tt.wantTraceID {
			t.Errorf("%s: Decode() gotTraceID = %v, want %v", tt.name, gotTraceID, tt.wantTraceID)
		}
		if gotSpanID != tt.wantSpanID {
			t.Errorf("%s: Decode() gotSpanID = %v, want %v", tt.name, gotSpanID, tt.wantSpanID)
		}
		if gotOpts != tt.wantOpts {
			t.Errorf("%s: Decode() gotOpts = %v, want %v", tt.name, gotOpts, tt.wantOpts)
		}
		if gotOk != tt.wantOk {
			t.Errorf("%s: Decode() gotOk = %v, want %v", tt.name, gotOk, tt.wantOk)
		}
	}
}

func BenchmarkBinary(b *testing.B) {
	tid := TraceID{0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f}
	sid := SpanID{0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68}
	sc := SpanContext{
		TraceID: tid,
		SpanID:  sid,
	}
	var x byte
	for i := 0; i < b.N; i++ {
		bin := Binary(sc)
		x += bin[0]
	}
	if x == 1 {
		fmt.Println(x) // try to prevent optimizing-out
	}
}

func BenchmarkFromBinary(b *testing.B) {
	bin := []byte{
		0, 0, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 1, 97, 98, 99, 100,
		101, 102, 103, 104, 2, 1,
	}
	var x byte
	for i := 0; i < b.N; i++ {
		sc, _ := FromBinary(bin)
		x += sc.TraceID[0]
	}
	if x == 1 {
		fmt.Println(x) // try to prevent optimizing-out
	}
}
