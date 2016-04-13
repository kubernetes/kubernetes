package broadcastwriter

import (
	"bytes"
	"errors"

	"testing"
)

type dummyWriter struct {
	buffer      bytes.Buffer
	failOnWrite bool
}

func (dw *dummyWriter) Write(p []byte) (n int, err error) {
	if dw.failOnWrite {
		return 0, errors.New("Fake fail")
	}
	return dw.buffer.Write(p)
}

func (dw *dummyWriter) String() string {
	return dw.buffer.String()
}

func (dw *dummyWriter) Close() error {
	return nil
}

func TestBroadcastWriter(t *testing.T) {
	writer := New()

	// Test 1: Both bufferA and bufferB should contain "foo"
	bufferA := &dummyWriter{}
	writer.AddWriter(bufferA, "")
	bufferB := &dummyWriter{}
	writer.AddWriter(bufferB, "")
	writer.Write([]byte("foo"))

	if bufferA.String() != "foo" {
		t.Errorf("Buffer contains %v", bufferA.String())
	}

	if bufferB.String() != "foo" {
		t.Errorf("Buffer contains %v", bufferB.String())
	}

	// Test2: bufferA and bufferB should contain "foobar",
	// while bufferC should only contain "bar"
	bufferC := &dummyWriter{}
	writer.AddWriter(bufferC, "")
	writer.Write([]byte("bar"))

	if bufferA.String() != "foobar" {
		t.Errorf("Buffer contains %v", bufferA.String())
	}

	if bufferB.String() != "foobar" {
		t.Errorf("Buffer contains %v", bufferB.String())
	}

	if bufferC.String() != "bar" {
		t.Errorf("Buffer contains %v", bufferC.String())
	}

	// Test3: Test eviction on failure
	bufferA.failOnWrite = true
	writer.Write([]byte("fail"))
	if bufferA.String() != "foobar" {
		t.Errorf("Buffer contains %v", bufferA.String())
	}
	if bufferC.String() != "barfail" {
		t.Errorf("Buffer contains %v", bufferC.String())
	}
	// Even though we reset the flag, no more writes should go in there
	bufferA.failOnWrite = false
	writer.Write([]byte("test"))
	if bufferA.String() != "foobar" {
		t.Errorf("Buffer contains %v", bufferA.String())
	}
	if bufferC.String() != "barfailtest" {
		t.Errorf("Buffer contains %v", bufferC.String())
	}

	writer.Clean()
}

type devNullCloser int

func (d devNullCloser) Close() error {
	return nil
}

func (d devNullCloser) Write(buf []byte) (int, error) {
	return len(buf), nil
}

// This test checks for races. It is only useful when run with the race detector.
func TestRaceBroadcastWriter(t *testing.T) {
	writer := New()
	c := make(chan bool)
	go func() {
		writer.AddWriter(devNullCloser(0), "")
		c <- true
	}()
	writer.Write([]byte("hello"))
	<-c
}

func BenchmarkBroadcastWriter(b *testing.B) {
	writer := New()
	setUpWriter := func() {
		for i := 0; i < 100; i++ {
			writer.AddWriter(devNullCloser(0), "stdout")
			writer.AddWriter(devNullCloser(0), "stderr")
			writer.AddWriter(devNullCloser(0), "")
		}
	}
	testLine := "Line that thinks that it is log line from docker"
	var buf bytes.Buffer
	for i := 0; i < 100; i++ {
		buf.Write([]byte(testLine + "\n"))
	}
	// line without eol
	buf.Write([]byte(testLine))
	testText := buf.Bytes()
	b.SetBytes(int64(5 * len(testText)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		setUpWriter()
		b.StartTimer()

		for j := 0; j < 5; j++ {
			if _, err := writer.Write(testText); err != nil {
				b.Fatal(err)
			}
		}

		b.StopTimer()
		writer.Clean()
		b.StartTimer()
	}
}

func BenchmarkBroadcastWriterWithoutStdoutStderr(b *testing.B) {
	writer := New()
	setUpWriter := func() {
		for i := 0; i < 100; i++ {
			writer.AddWriter(devNullCloser(0), "")
		}
	}
	testLine := "Line that thinks that it is log line from docker"
	var buf bytes.Buffer
	for i := 0; i < 100; i++ {
		buf.Write([]byte(testLine + "\n"))
	}
	// line without eol
	buf.Write([]byte(testLine))
	testText := buf.Bytes()
	b.SetBytes(int64(5 * len(testText)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		setUpWriter()

		for j := 0; j < 5; j++ {
			if _, err := writer.Write(testText); err != nil {
				b.Fatal(err)
			}
		}

		writer.Clean()
	}
}
