package request

import (
	"bytes"
	"io"
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/internal/sdkio"
)

func TestOffsetReaderRead(t *testing.T) {
	buf := []byte("testData")
	reader := &offsetReader{buf: bytes.NewReader(buf)}

	tempBuf := make([]byte, len(buf))

	n, err := reader.Read(tempBuf)

	if e, a := n, len(buf); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if e, a := buf, tempBuf; !bytes.Equal(e, a) {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestOffsetReaderSeek(t *testing.T) {
	buf := []byte("testData")
	reader, err := newOffsetReader(bytes.NewReader(buf), 0)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	orig, err := reader.Seek(0, sdkio.SeekCurrent)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if e, a := int64(0), orig; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

	n, err := reader.Seek(0, sdkio.SeekEnd)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if e, a := int64(len(buf)), n; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

	n, err = reader.Seek(orig, sdkio.SeekStart)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if e, a := int64(0), n; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestOffsetReaderClose(t *testing.T) {
	buf := []byte("testData")
	reader := &offsetReader{buf: bytes.NewReader(buf)}

	err := reader.Close()
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	tempBuf := make([]byte, len(buf))
	n, err := reader.Read(tempBuf)
	if e, a := n, 0; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := err, io.EOF; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestOffsetReaderCloseAndCopy(t *testing.T) {
	buf := []byte("testData")
	tempBuf := make([]byte, len(buf))
	reader := &offsetReader{buf: bytes.NewReader(buf)}

	newReader, err := reader.CloseAndCopy(0)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	n, err := reader.Read(tempBuf)
	if e, a := n, 0; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := err, io.EOF; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

	n, err = newReader.Read(tempBuf)
	if e, a := n, len(buf); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if e, a := buf, tempBuf; !bytes.Equal(e, a) {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestOffsetReaderCloseAndCopyOffset(t *testing.T) {
	buf := []byte("testData")
	tempBuf := make([]byte, len(buf))
	reader := &offsetReader{buf: bytes.NewReader(buf)}

	newReader, err := reader.CloseAndCopy(4)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	n, err := newReader.Read(tempBuf)
	if e, a := n, len(buf)-4; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	expected := []byte{'D', 'a', 't', 'a', 0, 0, 0, 0}
	if e, a := expected, tempBuf; !bytes.Equal(e, a) {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestOffsetReaderRace(t *testing.T) {
	wg := sync.WaitGroup{}

	f := func(reader *offsetReader) {
		defer wg.Done()
		var err error
		buf := make([]byte, 1)
		_, err = reader.Read(buf)
		for err != io.EOF {
			_, err = reader.Read(buf)
		}

	}

	closeFn := func(reader *offsetReader) {
		defer wg.Done()
		time.Sleep(time.Duration(rand.Intn(20)+1) * time.Millisecond)
		reader.Close()
	}
	for i := 0; i < 50; i++ {
		reader := &offsetReader{buf: bytes.NewReader(make([]byte, 1024*1024))}
		wg.Add(1)
		go f(reader)
		wg.Add(1)
		go closeFn(reader)
	}
	wg.Wait()
}

func BenchmarkOffsetReader(b *testing.B) {
	bufSize := 1024 * 1024 * 100
	buf := make([]byte, bufSize)
	reader := &offsetReader{buf: bytes.NewReader(buf)}

	tempBuf := make([]byte, 1024)

	for i := 0; i < b.N; i++ {
		reader.Read(tempBuf)
	}
}

func BenchmarkBytesReader(b *testing.B) {
	bufSize := 1024 * 1024 * 100
	buf := make([]byte, bufSize)
	reader := bytes.NewReader(buf)

	tempBuf := make([]byte, 1024)

	for i := 0; i < b.N; i++ {
		reader.Read(tempBuf)
	}
}
