package broadcaster

import (
	"bytes"
	"errors"
	"strings"

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

func TestUnbuffered(t *testing.T) {
	writer := new(Unbuffered)

	// Test 1: Both bufferA and bufferB should contain "foo"
	bufferA := &dummyWriter{}
	writer.Add(bufferA)
	bufferB := &dummyWriter{}
	writer.Add(bufferB)
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
	writer.Add(bufferC)
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

	// Test4: Test eviction on multiple simultaneous failures
	bufferB.failOnWrite = true
	bufferC.failOnWrite = true
	bufferD := &dummyWriter{}
	writer.Add(bufferD)
	writer.Write([]byte("yo"))
	writer.Write([]byte("ink"))
	if strings.Contains(bufferB.String(), "yoink") {
		t.Errorf("bufferB received write. contents: %q", bufferB)
	}
	if strings.Contains(bufferC.String(), "yoink") {
		t.Errorf("bufferC received write. contents: %q", bufferC)
	}
	if g, w := bufferD.String(), "yoink"; g != w {
		t.Errorf("bufferD = %q, want %q", g, w)
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
func TestRaceUnbuffered(t *testing.T) {
	writer := new(Unbuffered)
	c := make(chan bool)
	go func() {
		writer.Add(devNullCloser(0))
		c <- true
	}()
	writer.Write([]byte("hello"))
	<-c
}

func BenchmarkUnbuffered(b *testing.B) {
	writer := new(Unbuffered)
	setUpWriter := func() {
		for i := 0; i < 100; i++ {
			writer.Add(devNullCloser(0))
			writer.Add(devNullCloser(0))
			writer.Add(devNullCloser(0))
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
