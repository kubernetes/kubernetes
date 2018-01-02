// Copyright 2013 The Gorilla WebSocket Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package websocket

import (
	"bytes"
	"encoding/json"
	"io"
	"reflect"
	"testing"
)

func TestJSON(t *testing.T) {
	var buf bytes.Buffer
	c := fakeNetConn{&buf, &buf}
	wc := newConn(c, true, 1024, 1024)
	rc := newConn(c, false, 1024, 1024)

	var actual, expect struct {
		A int
		B string
	}
	expect.A = 1
	expect.B = "hello"

	if err := wc.WriteJSON(&expect); err != nil {
		t.Fatal("write", err)
	}

	if err := rc.ReadJSON(&actual); err != nil {
		t.Fatal("read", err)
	}

	if !reflect.DeepEqual(&actual, &expect) {
		t.Fatal("equal", actual, expect)
	}
}

func TestPartialJsonRead(t *testing.T) {
	var buf bytes.Buffer
	c := fakeNetConn{&buf, &buf}
	wc := newConn(c, true, 1024, 1024)
	rc := newConn(c, false, 1024, 1024)

	var v struct {
		A int
		B string
	}
	v.A = 1
	v.B = "hello"

	messageCount := 0

	// Partial JSON values.

	data, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	for i := len(data) - 1; i >= 0; i-- {
		if err := wc.WriteMessage(TextMessage, data[:i]); err != nil {
			t.Fatal(err)
		}
		messageCount++
	}

	// Whitespace.

	if err := wc.WriteMessage(TextMessage, []byte(" ")); err != nil {
		t.Fatal(err)
	}
	messageCount++

	// Close.

	if err := wc.WriteMessage(CloseMessage, FormatCloseMessage(CloseNormalClosure, "")); err != nil {
		t.Fatal(err)
	}

	for i := 0; i < messageCount; i++ {
		err := rc.ReadJSON(&v)
		if err != io.ErrUnexpectedEOF {
			t.Error("read", i, err)
		}
	}

	err = rc.ReadJSON(&v)
	if err != io.EOF {
		t.Error("final", err)
	}
}

func TestDeprecatedJSON(t *testing.T) {
	var buf bytes.Buffer
	c := fakeNetConn{&buf, &buf}
	wc := newConn(c, true, 1024, 1024)
	rc := newConn(c, false, 1024, 1024)

	var actual, expect struct {
		A int
		B string
	}
	expect.A = 1
	expect.B = "hello"

	if err := WriteJSON(wc, &expect); err != nil {
		t.Fatal("write", err)
	}

	if err := ReadJSON(rc, &actual); err != nil {
		t.Fatal("read", err)
	}

	if !reflect.DeepEqual(&actual, &expect) {
		t.Fatal("equal", actual, expect)
	}
}
