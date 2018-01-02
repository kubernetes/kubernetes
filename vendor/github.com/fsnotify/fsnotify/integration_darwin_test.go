// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fsnotify

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

// testExchangedataForWatcher tests the watcher with the exchangedata operation on OS X.
//
// This is widely used for atomic saves on OS X, e.g. TextMate and in Apple's NSDocument.
//
// See https://developer.apple.com/library/mac/documentation/Darwin/Reference/ManPages/man2/exchangedata.2.html
// Also see: https://github.com/textmate/textmate/blob/cd016be29489eba5f3c09b7b70b06da134dda550/Frameworks/io/src/swap_file_data.cc#L20
func testExchangedataForWatcher(t *testing.T, watchDir bool) {
	// Create directory to watch
	testDir1 := tempMkdir(t)

	// For the intermediate file
	testDir2 := tempMkdir(t)

	defer os.RemoveAll(testDir1)
	defer os.RemoveAll(testDir2)

	resolvedFilename := "TestFsnotifyEvents.file"

	// TextMate does:
	//
	// 1. exchangedata (intermediate, resolved)
	// 2. unlink intermediate
	//
	// Let's try to simulate that:
	resolved := filepath.Join(testDir1, resolvedFilename)
	intermediate := filepath.Join(testDir2, resolvedFilename+"~")

	// Make sure we create the file before we start watching
	createAndSyncFile(t, resolved)

	watcher := newWatcher(t)

	// Test both variants in isolation
	if watchDir {
		addWatch(t, watcher, testDir1)
	} else {
		addWatch(t, watcher, resolved)
	}

	// Receive errors on the error channel on a separate goroutine
	go func() {
		for err := range watcher.Errors {
			t.Fatalf("error received: %s", err)
		}
	}()

	// Receive events on the event channel on a separate goroutine
	eventstream := watcher.Events
	var removeReceived counter
	var createReceived counter

	done := make(chan bool)

	go func() {
		for event := range eventstream {
			// Only count relevant events
			if event.Name == filepath.Clean(resolved) {
				if event.Op&Remove == Remove {
					removeReceived.increment()
				}
				if event.Op&Create == Create {
					createReceived.increment()
				}
			}
			t.Logf("event received: %s", event)
		}
		done <- true
	}()

	// Repeat to make sure the watched file/directory "survives" the REMOVE/CREATE loop.
	for i := 1; i <= 3; i++ {
		// The intermediate file is created in a folder outside the watcher
		createAndSyncFile(t, intermediate)

		// 1. Swap
		if err := unix.Exchangedata(intermediate, resolved, 0); err != nil {
			t.Fatalf("[%d] exchangedata failed: %s", i, err)
		}

		time.Sleep(50 * time.Millisecond)

		// 2. Delete the intermediate file
		err := os.Remove(intermediate)

		if err != nil {
			t.Fatalf("[%d] remove %s failed: %s", i, intermediate, err)
		}

		time.Sleep(50 * time.Millisecond)

	}

	// We expect this event to be received almost immediately, but let's wait 500 ms to be sure
	time.Sleep(500 * time.Millisecond)

	// The events will be (CHMOD + REMOVE + CREATE) X 2. Let's focus on the last two:
	if removeReceived.value() < 3 {
		t.Fatal("fsnotify remove events have not been received after 500 ms")
	}

	if createReceived.value() < 3 {
		t.Fatal("fsnotify create events have not been received after 500 ms")
	}

	watcher.Close()
	t.Log("waiting for the event channel to become closed...")
	select {
	case <-done:
		t.Log("event channel closed")
	case <-time.After(2 * time.Second):
		t.Fatal("event stream was not closed after 2 seconds")
	}
}

// TestExchangedataInWatchedDir test exchangedata operation on file in watched dir.
func TestExchangedataInWatchedDir(t *testing.T) {
	testExchangedataForWatcher(t, true)
}

// TestExchangedataInWatchedDir test exchangedata operation on watched file.
func TestExchangedataInWatchedFile(t *testing.T) {
	testExchangedataForWatcher(t, false)
}

func createAndSyncFile(t *testing.T, filepath string) {
	f1, err := os.OpenFile(filepath, os.O_WRONLY|os.O_CREATE, 0666)
	if err != nil {
		t.Fatalf("creating %s failed: %s", filepath, err)
	}
	f1.Sync()
	f1.Close()
}
