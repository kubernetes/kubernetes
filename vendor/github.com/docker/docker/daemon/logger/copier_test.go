package logger

import (
	"bytes"
	"encoding/json"
	"io"
	"testing"
	"time"
)

type TestLoggerJSON struct {
	*json.Encoder
}

func (l *TestLoggerJSON) Log(m *Message) error { return l.Encode(m) }

func (l *TestLoggerJSON) Close() error { return nil }

func (l *TestLoggerJSON) Name() string { return "json" }

type TestLoggerText struct {
	*bytes.Buffer
}

func (l *TestLoggerText) Log(m *Message) error {
	_, err := l.WriteString(m.ContainerID + " " + m.Source + " " + string(m.Line) + "\n")
	return err
}

func (l *TestLoggerText) Close() error { return nil }

func (l *TestLoggerText) Name() string { return "text" }

func TestCopier(t *testing.T) {
	stdoutLine := "Line that thinks that it is log line from docker stdout"
	stderrLine := "Line that thinks that it is log line from docker stderr"
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

	var jsonBuf bytes.Buffer

	jsonLog := &TestLoggerJSON{Encoder: json.NewEncoder(&jsonBuf)}

	cid := "a7317399f3f857173c6179d44823594f8294678dea9999662e5c625b5a1c7657"
	c, err := NewCopier(cid,
		map[string]io.Reader{
			"stdout": &stdout,
			"stderr": &stderr,
		},
		jsonLog)
	if err != nil {
		t.Fatal(err)
	}
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
		if msg.ContainerID != cid {
			t.Fatalf("Wrong ContainerID: %q, expected %q", msg.ContainerID, cid)
		}
		if msg.Source == "stdout" {
			if string(msg.Line) != stdoutLine {
				t.Fatalf("Wrong Line: %q, expected %q", msg.Line, stdoutLine)
			}
		}
		if msg.Source == "stderr" {
			if string(msg.Line) != stderrLine {
				t.Fatalf("Wrong Line: %q, expected %q", msg.Line, stderrLine)
			}
		}
	}
}
