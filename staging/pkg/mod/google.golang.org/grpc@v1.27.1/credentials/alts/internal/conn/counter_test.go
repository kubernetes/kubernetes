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

package conn

import (
	"bytes"
	"testing"

	core "google.golang.org/grpc/credentials/alts/internal"
)

const (
	testOverflowLen = 5
)

func TestCounterSides(t *testing.T) {
	for _, side := range []core.Side{core.ClientSide, core.ServerSide} {
		outCounter := NewOutCounter(side, testOverflowLen)
		inCounter := NewInCounter(side, testOverflowLen)
		for i := 0; i < 1024; i++ {
			value, _ := outCounter.Value()
			if g, w := CounterSide(value), side; g != w {
				t.Errorf("after %d iterations, CounterSide(outCounter.Value()) = %v, want %v", i, g, w)
				break
			}
			value, _ = inCounter.Value()
			if g, w := CounterSide(value), side; g == w {
				t.Errorf("after %d iterations, CounterSide(inCounter.Value()) = %v, want %v", i, g, w)
				break
			}
			outCounter.Inc()
			inCounter.Inc()
		}
	}
}

func TestCounterInc(t *testing.T) {
	for _, test := range []struct {
		counter []byte
		want    []byte
	}{
		{
			counter: []byte{0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			want:    []byte{0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			counter: []byte{0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x80},
			want:    []byte{0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x80},
		},
		{
			counter: []byte{0xff, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			want:    []byte{0x00, 0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			counter: []byte{0x42, 0xff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			want:    []byte{0x43, 0xff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			counter: []byte{0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
			want:    []byte{0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
		},
		{
			counter: []byte{0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80},
			want:    []byte{0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80},
		},
	} {
		c := CounterFromValue(test.counter, overflowLenAES128GCM)
		c.Inc()
		value, _ := c.Value()
		if g, w := value, test.want; !bytes.Equal(g, w) || c.invalid {
			t.Errorf("counter(%v).Inc() =\n%v, want\n%v", test.counter, g, w)
		}
	}
}

func TestRolloverCounter(t *testing.T) {
	for _, test := range []struct {
		desc        string
		value       []byte
		overflowLen int
	}{
		{
			desc:        "testing overflow without rekeying 1",
			value:       []byte{0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80},
			overflowLen: 5,
		},
		{
			desc:        "testing overflow without rekeying 2",
			value:       []byte{0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
			overflowLen: 5,
		},
		{
			desc:        "testing overflow for rekeying mode 1",
			value:       []byte{0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x80},
			overflowLen: 8,
		},
		{
			desc:        "testing overflow for rekeying mode 2",
			value:       []byte{0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00},
			overflowLen: 8,
		},
	} {
		c := CounterFromValue(test.value, overflowLenAES128GCM)

		// First Inc() + Value() should work.
		c.Inc()
		_, err := c.Value()
		if err != nil {
			t.Errorf("%v: first Inc() + Value() unexpectedly failed: %v, want <nil> error", test.desc, err)
		}
		// Second Inc() + Value() should fail.
		c.Inc()
		_, err = c.Value()
		if err != errInvalidCounter {
			t.Errorf("%v: second Inc() + Value() unexpectedly succeeded: want %v", test.desc, errInvalidCounter)
		}
		// Third Inc() + Value() should also fail because the counter is
		// already in an invalid state.
		c.Inc()
		_, err = c.Value()
		if err != errInvalidCounter {
			t.Errorf("%v: Third Inc() + Value() unexpectedly succeeded: want %v", test.desc, errInvalidCounter)
		}
	}
}
