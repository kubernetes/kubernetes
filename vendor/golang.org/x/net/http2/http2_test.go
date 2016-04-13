// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
// See https://code.google.com/p/go/source/browse/CONTRIBUTORS
// Licensed under the same terms as Go itself:
// https://code.google.com/p/go/source/browse/LICENSE

package http2

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os/exec"
	"strconv"
	"strings"
	"testing"

	"golang.org/x/net/http2/hpack"
)

var knownFailing = flag.Bool("known_failing", false, "Run known-failing tests.")

func condSkipFailingTest(t *testing.T) {
	if !*knownFailing {
		t.Skip("Skipping known-failing test without --known_failing")
	}
}

func init() {
	DebugGoroutines = true
	flag.BoolVar(&VerboseLogs, "verboseh2", false, "Verbose HTTP/2 debug logging")
}

func TestSettingString(t *testing.T) {
	tests := []struct {
		s    Setting
		want string
	}{
		{Setting{SettingMaxFrameSize, 123}, "[MAX_FRAME_SIZE = 123]"},
		{Setting{1<<16 - 1, 123}, "[UNKNOWN_SETTING_65535 = 123]"},
	}
	for i, tt := range tests {
		got := fmt.Sprint(tt.s)
		if got != tt.want {
			t.Errorf("%d. for %#v, string = %q; want %q", i, tt.s, got, tt.want)
		}
	}
}

type twriter struct {
	t  testing.TB
	st *serverTester // optional
}

func (w twriter) Write(p []byte) (n int, err error) {
	if w.st != nil {
		ps := string(p)
		for _, phrase := range w.st.logFilter {
			if strings.Contains(ps, phrase) {
				return len(p), nil // no logging
			}
		}
	}
	w.t.Logf("%s", p)
	return len(p), nil
}

// like encodeHeader, but don't add implicit psuedo headers.
func encodeHeaderNoImplicit(t *testing.T, headers ...string) []byte {
	var buf bytes.Buffer
	enc := hpack.NewEncoder(&buf)
	for len(headers) > 0 {
		k, v := headers[0], headers[1]
		headers = headers[2:]
		if err := enc.WriteField(hpack.HeaderField{Name: k, Value: v}); err != nil {
			t.Fatalf("HPACK encoding error for %q/%q: %v", k, v, err)
		}
	}
	return buf.Bytes()
}

// Verify that curl has http2.
func requireCurl(t *testing.T) {
	out, err := dockerLogs(curl(t, "--version"))
	if err != nil {
		t.Skipf("failed to determine curl features; skipping test")
	}
	if !strings.Contains(string(out), "HTTP2") {
		t.Skip("curl doesn't support HTTP2; skipping test")
	}
}

func curl(t *testing.T, args ...string) (container string) {
	out, err := exec.Command("docker", append([]string{"run", "-d", "--net=host", "gohttp2/curl"}, args...)...).CombinedOutput()
	if err != nil {
		t.Skipf("Failed to run curl in docker: %v, %s", err, out)
	}
	return strings.TrimSpace(string(out))
}

type puppetCommand struct {
	fn   func(w http.ResponseWriter, r *http.Request)
	done chan<- bool
}

type handlerPuppet struct {
	ch chan puppetCommand
}

func newHandlerPuppet() *handlerPuppet {
	return &handlerPuppet{
		ch: make(chan puppetCommand),
	}
}

func (p *handlerPuppet) act(w http.ResponseWriter, r *http.Request) {
	for cmd := range p.ch {
		cmd.fn(w, r)
		cmd.done <- true
	}
}

func (p *handlerPuppet) done() { close(p.ch) }
func (p *handlerPuppet) do(fn func(http.ResponseWriter, *http.Request)) {
	done := make(chan bool)
	p.ch <- puppetCommand{fn, done}
	<-done
}
func dockerLogs(container string) ([]byte, error) {
	out, err := exec.Command("docker", "wait", container).CombinedOutput()
	if err != nil {
		return out, err
	}
	exitStatus, err := strconv.Atoi(strings.TrimSpace(string(out)))
	if err != nil {
		return out, errors.New("unexpected exit status from docker wait")
	}
	out, err = exec.Command("docker", "logs", container).CombinedOutput()
	exec.Command("docker", "rm", container).Run()
	if err == nil && exitStatus != 0 {
		err = fmt.Errorf("exit status %d: %s", exitStatus, out)
	}
	return out, err
}

func kill(container string) {
	exec.Command("docker", "kill", container).Run()
	exec.Command("docker", "rm", container).Run()
}
