package state

import (
	"testing"
	"time"
)

func TestTombstoneGC_invalid(t *testing.T) {
	_, err := NewTombstoneGC(0, 0)
	if err == nil {
		t.Fatalf("should fail")
	}

	_, err = NewTombstoneGC(time.Second, 0)
	if err == nil {
		t.Fatalf("should fail")
	}

	_, err = NewTombstoneGC(0, time.Second)
	if err == nil {
		t.Fatalf("should fail")
	}
}

func TestTombstoneGC(t *testing.T) {
	ttl := 20 * time.Millisecond
	gran := 5 * time.Millisecond
	gc, err := NewTombstoneGC(ttl, gran)
	if err != nil {
		t.Fatalf("should fail")
	}
	gc.SetEnabled(true)

	if gc.PendingExpiration() {
		t.Fatalf("should not be pending")
	}

	start := time.Now()
	gc.Hint(100)

	time.Sleep(2 * gran)
	start2 := time.Now()
	gc.Hint(120)
	gc.Hint(125)

	if !gc.PendingExpiration() {
		t.Fatalf("should be pending")
	}

	select {
	case index := <-gc.ExpireCh():
		end := time.Now()
		if end.Sub(start) < ttl {
			t.Fatalf("expired early")
		}
		if index != 100 {
			t.Fatalf("bad index: %d", index)
		}

	case <-time.After(ttl * 2):
		t.Fatalf("should get expiration")
	}

	select {
	case index := <-gc.ExpireCh():
		end := time.Now()
		if end.Sub(start2) < ttl {
			t.Fatalf("expired early")
		}
		if index != 125 {
			t.Fatalf("bad index: %d", index)
		}

	case <-time.After(ttl * 2):
		t.Fatalf("should get expiration")
	}
}

func TestTombstoneGC_Expire(t *testing.T) {
	ttl := 10 * time.Millisecond
	gran := 5 * time.Millisecond
	gc, err := NewTombstoneGC(ttl, gran)
	if err != nil {
		t.Fatalf("should fail")
	}
	gc.SetEnabled(true)

	if gc.PendingExpiration() {
		t.Fatalf("should not be pending")
	}

	gc.Hint(100)
	gc.SetEnabled(false)

	if gc.PendingExpiration() {
		t.Fatalf("should not be pending")
	}

	select {
	case <-gc.ExpireCh():
		t.Fatalf("should be reset")
	case <-time.After(20 * time.Millisecond):
	}
}
