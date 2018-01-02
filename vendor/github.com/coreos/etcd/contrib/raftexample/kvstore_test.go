// Copyright 2016 The etcd Authors
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

package main

import (
	"reflect"
	"testing"
)

func Test_kvstore_snapshot(t *testing.T) {
	tm := map[string]string{"foo": "bar"}
	s := &kvstore{kvStore: tm}

	v, _ := s.Lookup("foo")
	if v != "bar" {
		t.Fatalf("foo has unexpected value, got %s", v)
	}

	data, err := s.getSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	s.kvStore = nil

	if err := s.recoverFromSnapshot(data); err != nil {
		t.Fatal(err)
	}
	v, _ = s.Lookup("foo")
	if v != "bar" {
		t.Fatalf("foo has unexpected value, got %s", v)
	}
	if !reflect.DeepEqual(s.kvStore, tm) {
		t.Fatalf("store expected %+v, got %+v", tm, s.kvStore)
	}
}
