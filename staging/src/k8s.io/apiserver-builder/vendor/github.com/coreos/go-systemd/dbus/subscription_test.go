// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

	reschan := make(chan string)
	_, err = conn.StartUnit(target, "replace", reschan)
	if err != nil {
		t.Fatal(err)
	}

	job := <-reschan
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
