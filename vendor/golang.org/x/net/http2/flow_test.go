// Copyright 2014 The Go Authors.
// See https://code.google.com/p/go/source/browse/CONTRIBUTORS
// Licensed under the same terms as Go itself:
// https://code.google.com/p/go/source/browse/LICENSE

package http2

import "testing"

func TestFlow(t *testing.T) {
	var st flow
	var conn flow
	st.add(3)
	conn.add(2)

	if got, want := st.available(), int32(3); got != want {
		t.Errorf("available = %d; want %d", got, want)
	}
	st.setConnFlow(&conn)
	if got, want := st.available(), int32(2); got != want {
		t.Errorf("after parent setup, available = %d; want %d", got, want)
	}

	st.take(2)
	if got, want := conn.available(), int32(0); got != want {
		t.Errorf("after taking 2, conn = %d; want %d", got, want)
	}
	if got, want := st.available(), int32(0); got != want {
		t.Errorf("after taking 2, stream = %d; want %d", got, want)
	}
}

func TestFlowAdd(t *testing.T) {
	var f flow
	if !f.add(1) {
		t.Fatal("failed to add 1")
	}
	if !f.add(-1) {
		t.Fatal("failed to add -1")
	}
	if got, want := f.available(), int32(0); got != want {
		t.Fatalf("size = %d; want %d", got, want)
	}
	if !f.add(1<<31 - 1) {
		t.Fatal("failed to add 2^31-1")
	}
	if got, want := f.available(), int32(1<<31-1); got != want {
		t.Fatalf("size = %d; want %d", got, want)
	}
	if f.add(1) {
		t.Fatal("adding 1 to max shouldn't be allowed")
	}

}
