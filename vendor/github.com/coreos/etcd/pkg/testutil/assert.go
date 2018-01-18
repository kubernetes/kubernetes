// Copyright 2017 The etcd Authors
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

package testutil

import (
	"fmt"
	"reflect"
	"testing"
)

func AssertEqual(t *testing.T, e, a interface{}, msg ...string) {
	if (e == nil || a == nil) && (isNil(e) && isNil(a)) {
		return
	}
	if reflect.DeepEqual(e, a) {
		return
	}
	s := ""
	if len(msg) > 1 {
		s = msg[0] + ": "
	}
	s = fmt.Sprintf("%sexpected %+v, got %+v", s, e, a)
	FatalStack(t, s)
}

func AssertNil(t *testing.T, v interface{}) {
	AssertEqual(t, nil, v)
}

func AssertNotNil(t *testing.T, v interface{}) {
	if v == nil {
		t.Fatalf("expected non-nil, got %+v", v)
	}
}

func AssertTrue(t *testing.T, v bool, msg ...string) {
	AssertEqual(t, true, v, msg...)
}

func AssertFalse(t *testing.T, v bool, msg ...string) {
	AssertEqual(t, false, v, msg...)
}

func isNil(v interface{}) bool {
	if v == nil {
		return true
	}
	rv := reflect.ValueOf(v)
	return rv.Kind() != reflect.Struct && rv.IsNil()
}
