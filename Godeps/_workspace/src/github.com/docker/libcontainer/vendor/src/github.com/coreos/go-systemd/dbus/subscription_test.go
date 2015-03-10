package dbus

import (
	"testing"
	"time"
)

// TestSubscribe exercises the basics of subscription
func TestSubscribe(t *testing.T) {
	conn, err := New()

	if err != nil {
		t.Fatal(err)
	}

	err = conn.Subscribe()
	if err != nil {
		t.Fatal(err)
	}

	err = conn.Unsubscribe()
	if err != nil {
		t.Fatal(err)
	}
}

// TestSubscribeUnit exercises the basics of subscription of a particular unit.
func TestSubscribeUnit(t *testing.T) {
	target := "subscribe-events.service"

	conn, err := New()

	if err != nil {
		t.Fatal(err)
	}

	err = conn.Subscribe()
	if err != nil {
		t.Fatal(err)
	}

	err = conn.Unsubscribe()
	if err != nil {
		t.Fatal(err)
	}

	evChan, errChan := conn.SubscribeUnits(time.Second)

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

			// Just continue until we see our event.
			if !ok {
				continue
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


