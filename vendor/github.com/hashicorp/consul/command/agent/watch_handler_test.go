package agent

import (
	"io/ioutil"
	"os"
	"sync"
	"testing"
)

func TestVerifyWatchHandler(t *testing.T) {
	if err := verifyWatchHandler(nil); err == nil {
		t.Fatalf("should err")
	}
	if err := verifyWatchHandler(123); err == nil {
		t.Fatalf("should err")
	}
	if err := verifyWatchHandler([]string{"foo"}); err == nil {
		t.Fatalf("should err")
	}
	if err := verifyWatchHandler("foo"); err != nil {
		t.Fatalf("err: %v", err)
	}
}

func TestMakeWatchHandler(t *testing.T) {
	defer os.Remove("handler_out")
	defer os.Remove("handler_index_out")
	script := "echo $CONSUL_INDEX >> handler_index_out && cat >> handler_out"
	handler := makeWatchHandler(os.Stderr, script, &sync.RWMutex{})
	handler(100, []string{"foo", "bar", "baz"})
	raw, err := ioutil.ReadFile("handler_out")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if string(raw) != "[\"foo\",\"bar\",\"baz\"]\n" {
		t.Fatalf("bad: %s", raw)
	}
	raw, err = ioutil.ReadFile("handler_index_out")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if string(raw) != "100\n" {
		t.Fatalf("bad: %s", raw)
	}
}
