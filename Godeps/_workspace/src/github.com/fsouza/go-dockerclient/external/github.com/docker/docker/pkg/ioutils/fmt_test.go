package ioutils

import "testing"

func TestFprintfIfNotEmpty(t *testing.T) {
	wc := NewWriteCounter(&NopWriter{})
	n, _ := FprintfIfNotEmpty(wc, "foo%s", "")

	if wc.Count != 0 || n != 0 {
		t.Errorf("Wrong count: %v vs. %v vs. 0", wc.Count, n)
	}

	n, _ = FprintfIfNotEmpty(wc, "foo%s", "bar")
	if wc.Count != 6 || n != 6 {
		t.Errorf("Wrong count: %v vs. %v vs. 6", wc.Count, n)
	}
}
