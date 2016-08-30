package state

import (
	"testing"
	"time"
)

func TestDelay(t *testing.T) {
	d := NewDelay()

	// An unknown key should have a time in the past.
	if exp := d.GetExpiration("nope"); !exp.Before(time.Now()) {
		t.Fatalf("bad: %v", exp)
	}

	// Add a key and set a short expiration.
	now := time.Now()
	delay := 250 * time.Millisecond
	d.SetExpiration("bye", now, delay)
	if exp := d.GetExpiration("bye"); !exp.After(now) {
		t.Fatalf("bad: %v", exp)
	}

	// Wait for the key to expire and check again.
	time.Sleep(2 * delay)
	if exp := d.GetExpiration("bye"); !exp.Before(now) {
		t.Fatalf("bad: %v", exp)
	}
}
