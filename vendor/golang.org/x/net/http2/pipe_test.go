// Copyright 2014 The Go Authors.
// See https://code.google.com/p/go/source/browse/CONTRIBUTORS
// Licensed under the same terms as Go itself:
// https://code.google.com/p/go/source/browse/LICENSE

package http2

import (
	"errors"
	"testing"
)

func TestPipeClose(t *testing.T) {
	var p pipe
	p.c.L = &p.m
	a := errors.New("a")
	b := errors.New("b")
	p.Close(a)
	p.Close(b)
	_, err := p.Read(make([]byte, 1))
	if err != a {
		t.Errorf("err = %v want %v", err, a)
	}
}
