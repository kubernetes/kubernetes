// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package uid

import (
	"fmt"
	"testing"
	"time"
)

func TestNew(t *testing.T) {
	tm := time.Date(2017, 1, 6, 0, 0, 0, 21, time.UTC)
	s := NewSpace("prefix", &Options{Time: tm})
	got := s.New()
	want := "prefix-20170106-21-0001"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}

	s2 := NewSpace("prefix2", &Options{Sep: '_', Time: tm})
	got = s2.New()
	want = "prefix2_20170106_21_0001"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestTimestamp(t *testing.T) {
	s := NewSpace("unique-ID", nil)
	startTime := s.Time
	uid := s.New()
	got, ok := s.Timestamp(uid)
	if !ok {
		t.Fatal("got ok = false, want true")
	}
	if !startTime.Equal(got) {
		t.Errorf("got %s, want %s", got, startTime)
	}

	got, ok = s.Timestamp("unique-ID-20160308-123-8")
	if !ok {
		t.Fatal("got false, want true")
	}
	if want := time.Date(2016, 3, 8, 0, 0, 0, 123, time.UTC); !want.Equal(got) {
		t.Errorf("got %s, want %s", got, want)
	}
	if _, ok = s.Timestamp("invalid-time-1234"); ok {
		t.Error("got true, want false")
	}
}

func TestOlder(t *testing.T) {
	s := NewSpace("uid", nil)
	// A non-matching ID returns false.
	id2 := NewSpace("different-prefix", nil).New()
	if got, want := s.Older(id2, time.Second), false; got != want {
		t.Errorf("got %t, want %t", got, want)
	}
}

func TestShorter(t *testing.T) {
	now := time.Now()
	shortSpace := NewSpace("uid", &Options{Short: true, Time: now})
	shortUID := shortSpace.New()

	want := fmt.Sprintf("uid-%d-01", now.UnixNano())
	if shortUID != want {
		t.Fatalf("expected %s, got %s", want, shortUID)
	}

	if got, ok := shortSpace.Timestamp(shortUID); !ok {
		t.Fatal("expected to be able to parse timestamp from short space, but was unable to")
	} else if got.UnixNano() != now.UnixNano() {
		t.Fatalf("expected to get %v, got %v", now, got)
	}
}
