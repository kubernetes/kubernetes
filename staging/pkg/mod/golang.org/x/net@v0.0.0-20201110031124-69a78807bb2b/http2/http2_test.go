// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
	"time"

	"golang.org/x/net/http2/hpack"
)

var knownFailing = flag.Bool("known_failing", false, "Run known-failing tests.")

func condSkipFailingTest(t *testing.T) {
	if !*knownFailing {
		t.Skip("Skipping known-failing test without --known_failing")
	}
}

func init() {
	inTests = true
	DebugGoroutines = true
	flag.BoolVar(&VerboseLogs, "verboseh2", VerboseLogs, "Verbose HTTP/2 debug logging")
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

// like encodeHeader, but don't add implicit pseudo headers.
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
	out, err := exec.Command("docker", append([]string{"run", "-d", "--net=host", "gohttp2/curl"}, args...)...).Output()
	if err != nil {
		t.Skipf("Failed to run curl in docker: %v, %s", err, out)
	}
	return strings.TrimSpace(string(out))
}

// Verify that h2load exists.
func requireH2load(t *testing.T) {
	out, err := dockerLogs(h2load(t, "--version"))
	if err != nil {
		t.Skipf("failed to probe h2load; skipping test: %s", out)
	}
	if !strings.Contains(string(out), "h2load nghttp2/") {
		t.Skipf("h2load not present; skipping test. (Output=%q)", out)
	}
}

func h2load(t *testing.T, args ...string) (container string) {
	out, err := exec.Command("docker", append([]string{"run", "-d", "--net=host", "--entrypoint=/usr/local/bin/h2load", "gohttp2/curl"}, args...)...).Output()
	if err != nil {
		t.Skipf("Failed to run h2load in docker: %v, %s", err, out)
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

func cleanDate(res *http.Response) {
	if d := res.Header["Date"]; len(d) == 1 {
		d[0] = "XXX"
	}
}

func TestSorterPoolAllocs(t *testing.T) {
	ss := []string{"a", "b", "c"}
	h := http.Header{
		"a": nil,
		"b": nil,
		"c": nil,
	}
	sorter := new(sorter)

	if allocs := testing.AllocsPerRun(100, func() {
		sorter.SortStrings(ss)
	}); allocs >= 1 {
		t.Logf("SortStrings allocs = %v; want <1", allocs)
	}

	if allocs := testing.AllocsPerRun(5, func() {
		if len(sorter.Keys(h)) != 3 {
			t.Fatal("wrong result")
		}
	}); allocs > 0 {
		t.Logf("Keys allocs = %v; want <1", allocs)
	}
}

// waitCondition reports whether fn eventually returned true,
// checking immediately and then every checkEvery amount,
// until waitFor has elapsed, at which point it returns false.
func waitCondition(waitFor, checkEvery time.Duration, fn func() bool) bool {
	deadline := time.Now().Add(waitFor)
	for time.Now().Before(deadline) {
		if fn() {
			return true
		}
		time.Sleep(checkEvery)
	}
	return false
}

// waitErrCondition is like waitCondition but with errors instead of bools.
func waitErrCondition(waitFor, checkEvery time.Duration, fn func() error) error {
	deadline := time.Now().Add(waitFor)
	var err error
	for time.Now().Before(deadline) {
		if err = fn(); err == nil {
			return nil
		}
		time.Sleep(checkEvery)
	}
	return err
}

func equalError(a, b error) bool {
	if a == nil {
		return b == nil
	}
	if b == nil {
		return a == nil
	}
	return a.Error() == b.Error()
}

// Tests that http2.Server.IdleTimeout is initialized from
// http.Server.{Idle,Read}Timeout. http.Server.IdleTimeout was
// added in Go 1.8.
func TestConfigureServerIdleTimeout_Go18(t *testing.T) {
	const timeout = 5 * time.Second
	const notThisOne = 1 * time.Second

	// With a zero http2.Server, verify that it copies IdleTimeout:
	{
		s1 := &http.Server{
			IdleTimeout: timeout,
			ReadTimeout: notThisOne,
		}
		s2 := &Server{}
		if err := ConfigureServer(s1, s2); err != nil {
			t.Fatal(err)
		}
		if s2.IdleTimeout != timeout {
			t.Errorf("s2.IdleTimeout = %v; want %v", s2.IdleTimeout, timeout)
		}
	}

	// And that it falls back to ReadTimeout:
	{
		s1 := &http.Server{
			ReadTimeout: timeout,
		}
		s2 := &Server{}
		if err := ConfigureServer(s1, s2); err != nil {
			t.Fatal(err)
		}
		if s2.IdleTimeout != timeout {
			t.Errorf("s2.IdleTimeout = %v; want %v", s2.IdleTimeout, timeout)
		}
	}

	// Verify that s1's IdleTimeout doesn't overwrite an existing setting:
	{
		s1 := &http.Server{
			IdleTimeout: notThisOne,
		}
		s2 := &Server{
			IdleTimeout: timeout,
		}
		if err := ConfigureServer(s1, s2); err != nil {
			t.Fatal(err)
		}
		if s2.IdleTimeout != timeout {
			t.Errorf("s2.IdleTimeout = %v; want %v", s2.IdleTimeout, timeout)
		}
	}
}
