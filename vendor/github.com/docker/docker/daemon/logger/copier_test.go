package logger

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

type TestLoggerJSON struct {
	*json.Encoder
	mu    sync.Mutex
	delay time.Duration
}

func (l *TestLoggerJSON) Log(m *Message) error {
	if l.delay > 0 {
		time.Sleep(l.delay)
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.Encode(m)
}

func (l *TestLoggerJSON) Close() error { return nil }

func (l *TestLoggerJSON) Name() string { return "json" }

func TestCopier(t *testing.T) {
	stdoutLine := "Line that thinks that it is log line from docker stdout"
	stderrLine := "Line that thinks that it is log line from docker stderr"
	stdoutTrailingLine := "stdout trailing line"
	stderrTrailingLine := "stderr trailing line"

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	for i := 0; i < 30; i++ {
		if _, err := stdout.WriteString(stdoutLine + "\n"); err != nil {
			t.Fatal(err)
		}
		if _, err := stderr.WriteString(stderrLine + "\n"); err != nil {
			t.Fatal(err)
		}
	}

	// Test remaining lines without line-endings
	if _, err := stdout.WriteString(stdoutTrailingLine); err != nil {
		t.Fatal(err)
	}
	if _, err := stderr.WriteString(stderrTrailingLine); err != nil {
		t.Fatal(err)
	}

	var jsonBuf bytes.Buffer

	jsonLog := &TestLoggerJSON{Encoder: json.NewEncoder(&jsonBuf)}

	c := NewCopier(
		map[string]io.Reader{
			"stdout": &stdout,
			"stderr": &stderr,
		},
		jsonLog)
	c.Run()
	wait := make(chan struct{})
	go func() {
		c.Wait()
		close(wait)
	}()
	select {
	case <-time.After(1 * time.Second):
		t.Fatal("Copier failed to do its work in 1 second")
	case <-wait:
	}
	dec := json.NewDecoder(&jsonBuf)
	for {
		var msg Message
		if err := dec.Decode(&msg); err != nil {
			if err == io.EOF {
				break
			}
			t.Fatal(err)
		}
		if msg.Source != "stdout" && msg.Source != "stderr" {
			t.Fatalf("Wrong Source: %q, should be %q or %q", msg.Source, "stdout", "stderr")
		}
		if msg.Source == "stdout" {
			if string(msg.Line) != stdoutLine && string(msg.Line) != stdoutTrailingLine {
				t.Fatalf("Wrong Line: %q, expected %q or %q", msg.Line, stdoutLine, stdoutTrailingLine)
			}
		}
		if msg.Source == "stderr" {
			if string(msg.Line) != stderrLine && string(msg.Line) != stderrTrailingLine {
				t.Fatalf("Wrong Line: %q, expected %q or %q", msg.Line, stderrLine, stderrTrailingLine)
			}
		}
	}
}

// TestCopierLongLines tests long lines without line breaks
func TestCopierLongLines(t *testing.T) {
	// Long lines (should be split at "bufSize")
	const bufSize = 16 * 1024
	stdoutLongLine := strings.Repeat("a", bufSize)
	stderrLongLine := strings.Repeat("b", bufSize)
	stdoutTrailingLine := "stdout trailing line"
	stderrTrailingLine := "stderr trailing line"

	var stdout bytes.Buffer
	var stderr bytes.Buffer

	for i := 0; i < 3; i++ {
		if _, err := stdout.WriteString(stdoutLongLine); err != nil {
			t.Fatal(err)
		}
		if _, err := stderr.WriteString(stderrLongLine); err != nil {
			t.Fatal(err)
		}
	}

	if _, err := stdout.WriteString(stdoutTrailingLine); err != nil {
		t.Fatal(err)
	}
	if _, err := stderr.WriteString(stderrTrailingLine); err != nil {
		t.Fatal(err)
	}

	var jsonBuf bytes.Buffer

	jsonLog := &TestLoggerJSON{Encoder: json.NewEncoder(&jsonBuf)}

	c := NewCopier(
		map[string]io.Reader{
			"stdout": &stdout,
			"stderr": &stderr,
		},
		jsonLog)
	c.Run()
	wait := make(chan struct{})
	go func() {
		c.Wait()
		close(wait)
	}()
	select {
	case <-time.After(1 * time.Second):
		t.Fatal("Copier failed to do its work in 1 second")
	case <-wait:
	}
	dec := json.NewDecoder(&jsonBuf)
	for {
		var msg Message
		if err := dec.Decode(&msg); err != nil {
			if err == io.EOF {
				break
			}
			t.Fatal(err)
		}
		if msg.Source != "stdout" && msg.Source != "stderr" {
			t.Fatalf("Wrong Source: %q, should be %q or %q", msg.Source, "stdout", "stderr")
		}
		if msg.Source == "stdout" {
			if string(msg.Line) != stdoutLongLine && string(msg.Line) != stdoutTrailingLine {
				t.Fatalf("Wrong Line: %q, expected 'stdoutLongLine' or 'stdoutTrailingLine'", msg.Line)
			}
		}
		if msg.Source == "stderr" {
			if string(msg.Line) != stderrLongLine && string(msg.Line) != stderrTrailingLine {
				t.Fatalf("Wrong Line: %q, expected 'stderrLongLine' or 'stderrTrailingLine'", msg.Line)
			}
		}
	}
}

func TestCopierSlow(t *testing.T) {
	stdoutLine := "Line that thinks that it is log line from docker stdout"
	var stdout bytes.Buffer
	for i := 0; i < 30; i++ {
		if _, err := stdout.WriteString(stdoutLine + "\n"); err != nil {
			t.Fatal(err)
		}
	}

	var jsonBuf bytes.Buffer
	//encoder := &encodeCloser{Encoder: json.NewEncoder(&jsonBuf)}
	jsonLog := &TestLoggerJSON{Encoder: json.NewEncoder(&jsonBuf), delay: 100 * time.Millisecond}

	c := NewCopier(map[string]io.Reader{"stdout": &stdout}, jsonLog)
	c.Run()
	wait := make(chan struct{})
	go func() {
		c.Wait()
		close(wait)
	}()
	<-time.After(150 * time.Millisecond)
	c.Close()
	select {
	case <-time.After(200 * time.Millisecond):
		t.Fatal("failed to exit in time after the copier is closed")
	case <-wait:
	}
}

type BenchmarkLoggerDummy struct {
}

func (l *BenchmarkLoggerDummy) Log(m *Message) error { PutMessage(m); return nil }

func (l *BenchmarkLoggerDummy) Close() error { return nil }

func (l *BenchmarkLoggerDummy) Name() string { return "dummy" }

func BenchmarkCopier64(b *testing.B) {
	benchmarkCopier(b, 1<<6)
}
func BenchmarkCopier128(b *testing.B) {
	benchmarkCopier(b, 1<<7)
}
func BenchmarkCopier256(b *testing.B) {
	benchmarkCopier(b, 1<<8)
}
func BenchmarkCopier512(b *testing.B) {
	benchmarkCopier(b, 1<<9)
}
func BenchmarkCopier1K(b *testing.B) {
	benchmarkCopier(b, 1<<10)
}
func BenchmarkCopier2K(b *testing.B) {
	benchmarkCopier(b, 1<<11)
}
func BenchmarkCopier4K(b *testing.B) {
	benchmarkCopier(b, 1<<12)
}
func BenchmarkCopier8K(b *testing.B) {
	benchmarkCopier(b, 1<<13)
}
func BenchmarkCopier16K(b *testing.B) {
	benchmarkCopier(b, 1<<14)
}
func BenchmarkCopier32K(b *testing.B) {
	benchmarkCopier(b, 1<<15)
}
func BenchmarkCopier64K(b *testing.B) {
	benchmarkCopier(b, 1<<16)
}
func BenchmarkCopier128K(b *testing.B) {
	benchmarkCopier(b, 1<<17)
}
func BenchmarkCopier256K(b *testing.B) {
	benchmarkCopier(b, 1<<18)
}

func piped(b *testing.B, iterations int, delay time.Duration, buf []byte) io.Reader {
	r, w, err := os.Pipe()
	if err != nil {
		b.Fatal(err)
		return nil
	}
	go func() {
		for i := 0; i < iterations; i++ {
			time.Sleep(delay)
			if n, err := w.Write(buf); err != nil || n != len(buf) {
				if err != nil {
					b.Fatal(err)
				}
				b.Fatal(fmt.Errorf("short write"))
			}
		}
		w.Close()
	}()
	return r
}

func benchmarkCopier(b *testing.B, length int) {
	b.StopTimer()
	buf := []byte{'A'}
	for len(buf) < length {
		buf = append(buf, buf...)
	}
	buf = append(buf[:length-1], []byte{'\n'}...)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		c := NewCopier(
			map[string]io.Reader{
				"buffer": piped(b, 10, time.Nanosecond, buf),
			},
			&BenchmarkLoggerDummy{})
		c.Run()
		c.Wait()
		c.Close()
	}
}
