// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package h2c

import (
	"bufio"
	"bytes"
	"fmt"
	"log"
	"net/http"
	"testing"

	"golang.org/x/net/http2"
)

func TestSettingsAckSwallowWriter(t *testing.T) {
	var buf bytes.Buffer
	swallower := newSettingsAckSwallowWriter(bufio.NewWriter(&buf))
	fw := http2.NewFramer(swallower, nil)
	fw.WriteSettings(http2.Setting{http2.SettingMaxFrameSize, 2})
	fw.WriteSettingsAck()
	fw.WriteData(1, true, []byte{})
	swallower.Flush()

	fr := http2.NewFramer(nil, bufio.NewReader(&buf))

	f, err := fr.ReadFrame()
	if err != nil {
		t.Fatal(err)
	}
	if f.Header().Type != http2.FrameSettings {
		t.Fatalf("Expected first frame to be SETTINGS. Got: %v", f.Header().Type)
	}

	f, err = fr.ReadFrame()
	if err != nil {
		t.Fatal(err)
	}
	if f.Header().Type != http2.FrameData {
		t.Fatalf("Expected first frame to be DATA. Got: %v", f.Header().Type)
	}
}

func ExampleNewHandler() {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "Hello world")
	})
	h2s := &http2.Server{
		// ...
	}
	h1s := &http.Server{
		Addr:    ":8080",
		Handler: NewHandler(handler, h2s),
	}
	log.Fatal(h1s.ListenAndServe())
}
