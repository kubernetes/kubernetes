/*
Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// This file contains only basic checks. The fake is effectively tested by the
// logging client unit tests.

package testing

import (
	"reflect"
	"testing"
	"time"

	tspb "github.com/golang/protobuf/ptypes/timestamp"
	logpb "google.golang.org/genproto/googleapis/logging/v2"
	grpc "google.golang.org/grpc"
)

func TestNewServer(t *testing.T) {
	// Confirm that we can create and use a working gRPC server.
	addr, err := NewServer()
	if err != nil {
		t.Fatal(err)
	}
	conn, err := grpc.Dial(addr, grpc.WithInsecure())
	if err != nil {
		t.Fatal(err)
	}
	// Avoid "connection is closing; please retry" message from gRPC.
	time.Sleep(300 * time.Millisecond)
	conn.Close()
}

func TestParseFilter(t *testing.T) {
	for _, test := range []struct {
		filter  string
		want    string
		wantErr bool
	}{
		{"", "", false},
		{"logName = syslog", "syslog", false},
		{"logname = syslog", "", true},
		{"logName = 'syslog'", "", true},
		{"logName == syslog", "", true},
	} {
		got, err := parseFilter(test.filter)
		if err != nil {
			if !test.wantErr {
				t.Errorf("%q: got %v, want no error", test.filter, err)
			}
			continue
		}
		if test.wantErr {
			t.Errorf("%q: got no error, want one", test.filter)
			continue
		}
		if got != test.want {
			t.Errorf("%q: got %q, want %q", test.filter, got, test.want)
		}
	}
}

func TestSortEntries(t *testing.T) {
	entries := []*logpb.LogEntry{
		/* 0 */ {Timestamp: &tspb.Timestamp{Seconds: 30}},
		/* 1 */ {Timestamp: &tspb.Timestamp{Seconds: 10}},
		/* 2 */ {Timestamp: &tspb.Timestamp{Seconds: 20}, InsertId: "b"},
		/* 3 */ {Timestamp: &tspb.Timestamp{Seconds: 20}, InsertId: "a"},
		/* 4 */ {Timestamp: &tspb.Timestamp{Seconds: 20}, InsertId: "c"},
	}
	for _, test := range []struct {
		orderBy string
		want    []int // slice of index into entries; nil == error
	}{
		{"", []int{1, 3, 2, 4, 0}},
		{"timestamp asc", []int{1, 3, 2, 4, 0}},
		{"timestamp desc", []int{0, 4, 2, 3, 1}},
		{"something else", nil},
	} {
		got := make([]*logpb.LogEntry, len(entries))
		copy(got, entries)
		err := sortEntries(got, test.orderBy)
		if err != nil {
			if test.want != nil {
				t.Errorf("%q: got %v, want nil error", test.orderBy, err)
			}
			continue
		}
		want := make([]*logpb.LogEntry, len(entries))
		for i, j := range test.want {
			want[i] = entries[j]
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%q: got %v, want %v", test.orderBy, got, want)
		}
	}
}
