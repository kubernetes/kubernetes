// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"reflect"
	"testing"
)

type s struct{}

func (s) String() string { return "lazy string" }

// TestReset checks whether all the fields are zeroed after reset.
func TestReset(t *testing.T) {
	tr := New("foo", "bar")
	tr.LazyLog(s{}, false)
	tr.LazyPrintf("%d", 1)
	tr.SetRecycler(func(_ interface{}) {})
	tr.SetTraceInfo(3, 4)
	tr.SetMaxEvents(100)
	tr.SetError()
	tr.Finish()

	tr.(*trace).reset()

	if !reflect.DeepEqual(tr, new(trace)) {
		t.Errorf("reset didn't clear all fields: %+v", tr)
	}
}

// TestResetLog checks whether all the fields are zeroed after reset.
func TestResetLog(t *testing.T) {
	el := NewEventLog("foo", "bar")
	el.Printf("message")
	el.Errorf("error")
	el.Finish()

	el.(*eventLog).reset()

	if !reflect.DeepEqual(el, new(eventLog)) {
		t.Errorf("reset didn't clear all fields: %+v", el)
	}
}
