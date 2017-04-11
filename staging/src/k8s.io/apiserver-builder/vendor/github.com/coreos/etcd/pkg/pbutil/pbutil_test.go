// Copyright 2015 The etcd Authors
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
	"errors"
	"reflect"
	"testing"
)

func TestMarshaler(t *testing.T) {
	data := []byte("test data")
	m := &fakeMarshaler{data: data}
	if g := MustMarshal(m); !reflect.DeepEqual(g, data) {
		t.Errorf("data = %s, want %s", g, m)
	}
}

func TestMarshalerPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("recover = nil, want error")
		}
	}()
	m := &fakeMarshaler{err: errors.New("blah")}
	MustMarshal(m)
}

func TestUnmarshaler(t *testing.T) {
	data := []byte("test data")
	m := &fakeUnmarshaler{}
	MustUnmarshal(m, data)
	if !reflect.DeepEqual(m.data, data) {
		t.Errorf("data = %s, want %s", m.data, m)
	}
}

func TestUnmarshalerPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("recover = nil, want error")
		}
	}()
	m := &fakeUnmarshaler{err: errors.New("blah")}
	MustUnmarshal(m, nil)
}

func TestGetBool(t *testing.T) {
	tests := []struct {
		b    *bool
		wb   bool
		wset bool
	}{
		{nil, false, false},
		{Boolp(true), true, true},
		{Boolp(false), false, true},
	}
	for i, tt := range tests {
		b, set := GetBool(tt.b)
		if b != tt.wb {
			t.Errorf("#%d: value = %v, want %v", i, b, tt.wb)
		}
		if set != tt.wset {
			t.Errorf("#%d: set = %v, want %v", i, set, tt.wset)
		}
	}
}

type fakeMarshaler struct {
	data []byte
	err  error
}

func (m *fakeMarshaler) Marshal() ([]byte, error) {
	return m.data, m.err
}

type fakeUnmarshaler struct {
	data []byte
	err  error
}

func (m *fakeUnmarshaler) Unmarshal(data []byte) error {
	m.data = data
	return m.err
}
