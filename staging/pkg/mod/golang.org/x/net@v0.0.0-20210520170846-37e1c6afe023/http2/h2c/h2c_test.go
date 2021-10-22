// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package h2c

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"fmt"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
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

func TestContext(t *testing.T) {
	baseCtx := context.WithValue(context.Background(), "testkey", "testvalue")

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.ProtoMajor != 2 {
			t.Errorf("Request wasn't handled by h2c.  Got ProtoMajor=%v", r.ProtoMajor)
		}
		if r.Context().Value("testkey") != "testvalue" {
			t.Errorf("Request doesn't have expected base context: %v", r.Context())
		}
		fmt.Fprint(w, "Hello world")
	})

	h2s := &http2.Server{}
	h1s := httptest.NewUnstartedServer(NewHandler(handler, h2s))
	h1s.Config.BaseContext = func(_ net.Listener) context.Context {
		return baseCtx
	}
	h1s.Start()
	defer h1s.Close()

	client := &http.Client{
		Transport: &http2.Transport{
			AllowHTTP: true,
			DialTLS: func(network, addr string, _ *tls.Config) (net.Conn, error) {
				return net.Dial(network, addr)
			},
		},
	}

	resp, err := client.Get(h1s.URL)
	if err != nil {
		t.Fatal(err)
	}
	_, err = ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatal(err)
	}
}
