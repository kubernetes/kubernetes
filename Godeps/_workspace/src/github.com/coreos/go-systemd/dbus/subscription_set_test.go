package dbus

import (
	"testing"
	"time"
)

// TestSubscribeUnit exercises the basics of subscription of a particular unit.
func TestSubscriptionSetUnit(t *testing.T) {
	target := "subscribe-events-set.service"

	conn, err := New()

	if err != nil {
		t.Fatal(err)
	}

	err = conn.Subscribe()
	if err != nil {
		t.Fatal(err)
	}

	subSet := conn.NewSubscriptionSet()
	evChan, errChan := subSet.Subscribe()

	subSet.Add(target)
	setupUnit(target, conn, t)
	linkUnit(target, conn, t)

	job, err := conn.StartUnit(target, "replace")
	if err != nil {
		t.Fatal(err)
	}

	if job != "done" {
		t.Fatal("Couldn't start", target)
	}

	timeout := make(chan bool, 1)
	go func() {
		time.Sleep(3 * time.Second)
		close(timeout)
	}()

	for {
		select {
		case changes := <-evChan:
			tCh, ok := changes[target]

			if !ok {
				t.Fatal("Unexpected event:", changes)
			}

			if tCh.ActiveState == "active" && tCh.Name == target {
				goto success
			}
		case err = <-errChan:
			t.Fatal(err)
		case <-timeout:
			t.Fatal("Reached timeout")
		}
	}

success:
	return
}
