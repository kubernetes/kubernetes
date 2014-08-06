// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build freebsd openbsd netbsd darwin linux

package fsnotify

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestFsnotifyFakeSymlink(t *testing.T) {
	watcher := newWatcher(t)

	// Create directory to watch
	testDir := tempMkdir(t)
	defer os.RemoveAll(testDir)

	var errorsReceived counter
	// Receive errors on the error channel on a separate goroutine
	go func() {
		for errors := range watcher.Errors {
			t.Logf("Received error: %s", errors)
			errorsReceived.increment()
		}
	}()

	// Count the CREATE events received
	var createEventsReceived, otherEventsReceived counter
	go func() {
		for ev := range watcher.Events {
			t.Logf("event received: %s", ev)
			if ev.Op&Create == Create {
				createEventsReceived.increment()
			} else {
				otherEventsReceived.increment()
			}
		}
	}()

	addWatch(t, watcher, testDir)

	if err := os.Symlink(filepath.Join(testDir, "zzz"), filepath.Join(testDir, "zzznew")); err != nil {
		t.Fatalf("Failed to create bogus symlink: %s", err)
	}
	t.Logf("Created bogus symlink")

	// We expect this event to be received almost immediately, but let's wait 500 ms to be sure
	time.Sleep(500 * time.Millisecond)

	// Should not be error, just no events for broken links (watching nothing)
	if errorsReceived.value() > 0 {
		t.Fatal("fsnotify errors have been received.")
	}
	if otherEventsReceived.value() > 0 {
		t.Fatal("fsnotify other events received on the broken link")
	}

	// Except for 1 create event (for the link itself)
	if createEventsReceived.value() == 0 {
		t.Fatal("fsnotify create events were not received after 500 ms")
	}
	if createEventsReceived.value() > 1 {
		t.Fatal("fsnotify more create events received than expected")
	}

	// Try closing the fsnotify instance
	t.Log("calling Close()")
	watcher.Close()
}
