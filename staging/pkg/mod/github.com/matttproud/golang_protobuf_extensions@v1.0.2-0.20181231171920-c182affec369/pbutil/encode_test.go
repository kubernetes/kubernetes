// Copyright 2016 Matt T. Proud
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

package pbutil

import (
	"bytes"
	"errors"
	"testing"

	"github.com/golang/protobuf/proto"
)

var errMarshal = errors.New("pbutil: can't marshal")

type cantMarshal struct{ proto.Message }

func (cantMarshal) Marshal() ([]byte, error) { return nil, errMarshal }

var _ proto.Message = cantMarshal{}

func TestWriteDelimitedMarshalErr(t *testing.T) {
	t.Parallel()
	var data cantMarshal
	var buf bytes.Buffer
	n, err := WriteDelimited(&buf, data)
	if got, want := n, 0; got != want {
		t.Errorf("WriteDelimited(buf, %#v) = %#v, ?; want = %v#, ?", data, got, want)
	}
	if got, want := err, errMarshal; got != want {
		t.Errorf("WriteDelimited(buf, %#v) = ?, %#v; want = ?, %#v", data, got, want)
	}
}

type canMarshal struct{ proto.Message }

func (canMarshal) Marshal() ([]byte, error) { return []byte{0, 1, 2, 3, 4, 5}, nil }

var errWrite = errors.New("pbutil: can't write")

type cantWrite struct{}

func (cantWrite) Write([]byte) (int, error) { return 0, errWrite }

func TestWriteDelimitedWriteErr(t *testing.T) {
	t.Parallel()
	var data canMarshal
	var buf cantWrite
	n, err := WriteDelimited(buf, data)
	if got, want := n, 0; got != want {
		t.Errorf("WriteDelimited(buf, %#v) = %#v, ?; want = %v#, ?", data, got, want)
	}
	if got, want := err, errWrite; got != want {
		t.Errorf("WriteDelimited(buf, %#v) = ?, %#v; want = ?, %#v", data, got, want)
	}
}
