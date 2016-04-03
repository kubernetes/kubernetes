package progressreader

import (
	"bufio"
	"bytes"
	"io"
	"io/ioutil"
	"testing"

	"github.com/docker/docker/pkg/streamformatter"
)

func TestOutputOnPrematureClose(t *testing.T) {
	var outBuf bytes.Buffer
	content := []byte("TESTING")
	reader := ioutil.NopCloser(bytes.NewReader(content))
	writer := bufio.NewWriter(&outBuf)

	prCfg := Config{
		In:        reader,
		Out:       writer,
		Formatter: streamformatter.NewStreamFormatter(),
		Size:      len(content),
		NewLines:  true,
		ID:        "Test",
		Action:    "Read",
	}
	pr := New(prCfg)

	part := make([]byte, 4, 4)
	_, err := io.ReadFull(pr, part)
	if err != nil {
		pr.Close()
		t.Fatal(err)
	}

	if err := writer.Flush(); err != nil {
		pr.Close()
		t.Fatal(err)
	}

	tlen := outBuf.Len()
	pr.Close()
	if err := writer.Flush(); err != nil {
		t.Fatal(err)
	}

	if outBuf.Len() == tlen {
		t.Fatalf("Expected some output when closing prematurely")
	}
}

func TestCompleteSilently(t *testing.T) {
	var outBuf bytes.Buffer
	content := []byte("TESTING")
	reader := ioutil.NopCloser(bytes.NewReader(content))
	writer := bufio.NewWriter(&outBuf)

	prCfg := Config{
		In:        reader,
		Out:       writer,
		Formatter: streamformatter.NewStreamFormatter(),
		Size:      len(content),
		NewLines:  true,
		ID:        "Test",
		Action:    "Read",
	}
	pr := New(prCfg)

	out, err := ioutil.ReadAll(pr)
	if err != nil {
		pr.Close()
		t.Fatal(err)
	}
	if string(out) != "TESTING" {
		pr.Close()
		t.Fatalf("Unexpected output %q from reader", string(out))
	}

	if err := writer.Flush(); err != nil {
		pr.Close()
		t.Fatal(err)
	}

	tlen := outBuf.Len()
	pr.Close()
	if err := writer.Flush(); err != nil {
		t.Fatal(err)
	}

	if outBuf.Len() > tlen {
		t.Fatalf("Should have closed silently when read is complete")
	}
}
