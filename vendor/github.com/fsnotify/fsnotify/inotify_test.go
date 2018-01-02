// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

package fsnotify

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

func TestInotifyCloseRightAway(t *testing.T) {
	w, err := NewWatcher()
	if err != nil {
		t.Fatalf("Failed to create watcher")
	}

	// Close immediately; it won't even reach the first unix.Read.
	w.Close()

	// Wait for the close to complete.
	<-time.After(50 * time.Millisecond)
	isWatcherReallyClosed(t, w)
}

func TestInotifyCloseSlightlyLater(t *testing.T) {
	w, err := NewWatcher()
	if err != nil {
		t.Fatalf("Failed to create watcher")
	}

	// Wait until readEvents has reached unix.Read, and Close.
	<-time.After(50 * time.Millisecond)
	w.Close()

	// Wait for the close to complete.
	<-time.After(50 * time.Millisecond)
	isWatcherReallyClosed(t, w)
}

func TestInotifyCloseSlightlyLaterWithWatch(t *testing.T) {
	testDir := tempMkdir(t)
	defer os.RemoveAll(testDir)

	w, err := NewWatcher()
	if err != nil {
		t.Fatalf("Failed to create watcher")
	}
	w.Add(testDir)

	// Wait until readEvents has reached unix.Read, and Close.
	<-time.After(50 * time.Millisecond)
	w.Close()

	// Wait for the close to complete.
	<-time.After(50 * time.Millisecond)
	isWatcherReallyClosed(t, w)
}

func TestInotifyCloseAfterRead(t *testing.T) {
	testDir := tempMkdir(t)
	defer os.RemoveAll(testDir)

	w, err := NewWatcher()
	if err != nil {
		t.Fatalf("Failed to create watcher")
	}

	err = w.Add(testDir)
	if err != nil {
		t.Fatalf("Failed to add .")
	}

	// Generate an event.
	os.Create(filepath.Join(testDir, "somethingSOMETHINGsomethingSOMETHING"))

	// Wait for readEvents to read the event, then close the watcher.
	<-time.After(50 * time.Millisecond)
	w.Close()

	// Wait for the close to complete.
	<-time.After(50 * time.Millisecond)
	isWatcherReallyClosed(t, w)
}

func isWatcherReallyClosed(t *testing.T, w *Watcher) {
	select {
	case err, ok := <-w.Errors:
		if ok {
			t.Fatalf("w.Errors is not closed; readEvents is still alive after closing (error: %v)", err)
		}
	default:
		t.Fatalf("w.Errors would have blocked; readEvents is still alive!")
	}

	select {
	case _, ok := <-w.Events:
		if ok {
			t.Fatalf("w.Events is not closed; readEvents is still alive after closing")
		}
	default:
		t.Fatalf("w.Events would have blocked; readEvents is still alive!")
	}
}

func TestInotifyCloseCreate(t *testing.T) {
	testDir := tempMkdir(t)
	defer os.RemoveAll(testDir)

	w, err := NewWatcher()
	if err != nil {
		t.Fatalf("Failed to create watcher: %v", err)
	}
	defer w.Close()

	err = w.Add(testDir)
	if err != nil {
		t.Fatalf("Failed to add testDir: %v", err)
	}
	h, err := os.Create(filepath.Join(testDir, "testfile"))
	if err != nil {
		t.Fatalf("Failed to create file in testdir: %v", err)
	}
	h.Close()
	select {
	case _ = <-w.Events:
	case err := <-w.Errors:
		t.Fatalf("Error from watcher: %v", err)
	case <-time.After(50 * time.Millisecond):
		t.Fatalf("Took too long to wait for event")
	}

	// At this point, we've received one event, so the goroutine is ready.
	// It's also blocking on unix.Read.
	// Now we try to swap the file descriptor under its nose.
	w.Close()
	w, err = NewWatcher()
	defer w.Close()
	if err != nil {
		t.Fatalf("Failed to create second watcher: %v", err)
	}

	<-time.After(50 * time.Millisecond)
	err = w.Add(testDir)
	if err != nil {
		t.Fatalf("Error adding testDir again: %v", err)
	}
}

func TestInotifyStress(t *testing.T) {
	testDir := tempMkdir(t)
	defer os.RemoveAll(testDir)
	testFile := filepath.Join(testDir, "testfile")

	w, err := NewWatcher()
	if err != nil {
		t.Fatalf("Failed to create watcher: %v", err)
	}
	defer w.Close()

	killchan := make(chan struct{})
	defer close(killchan)

	err = w.Add(testDir)
	if err != nil {
		t.Fatalf("Failed to add testDir: %v", err)
	}

	proc, err := os.FindProcess(os.Getpid())
	if err != nil {
		t.Fatalf("Error finding process: %v", err)
	}

	go func() {
		for {
			select {
			case <-time.After(5 * time.Millisecond):
				err := proc.Signal(unix.SIGUSR1)
				if err != nil {
					t.Fatalf("Signal failed: %v", err)
				}
			case <-killchan:
				return
			}
		}
	}()

	go func() {
		for {
			select {
			case <-time.After(11 * time.Millisecond):
				err := w.poller.wake()
				if err != nil {
					t.Fatalf("Wake failed: %v", err)
				}
			case <-killchan:
				return
			}
		}
	}()

	go func() {
		for {
			select {
			case <-killchan:
				return
			default:
				handle, err := os.Create(testFile)
				if err != nil {
					t.Fatalf("Create failed: %v", err)
				}
				handle.Close()
				time.Sleep(time.Millisecond)
				err = os.Remove(testFile)
				if err != nil {
					t.Fatalf("Remove failed: %v", err)
				}
			}
		}
	}()

	creates := 0
	removes := 0
	after := time.After(5 * time.Second)
	for {
		select {
		case <-after:
			if creates-removes > 1 || creates-removes < -1 {
				t.Fatalf("Creates and removes should not be off by more than one: %d creates, %d removes", creates, removes)
			}
			if creates < 50 {
				t.Fatalf("Expected at least 50 creates, got %d", creates)
			}
			return
		case err := <-w.Errors:
			t.Fatalf("Got an error from watcher: %v", err)
		case evt := <-w.Events:
			if evt.Name != testFile {
				t.Fatalf("Got an event for an unknown file: %s", evt.Name)
			}
			if evt.Op == Create {
				creates++
			}
			if evt.Op == Remove {
				removes++
			}
		}
	}
}

func TestInotifyRemoveTwice(t *testing.T) {
	testDir := tempMkdir(t)
	defer os.RemoveAll(testDir)
	testFile := filepath.Join(testDir, "testfile")

	handle, err := os.Create(testFile)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}
	handle.Close()

	w, err := NewWatcher()
	if err != nil {
		t.Fatalf("Failed to create watcher: %v", err)
	}
	defer w.Close()

	err = w.Add(testFile)
	if err != nil {
		t.Fatalf("Failed to add testFile: %v", err)
	}

	err = os.Remove(testFile)
	if err != nil {
		t.Fatalf("Failed to remove testFile: %v", err)
	}

	err = w.Remove(testFile)
	if err == nil {
		t.Fatalf("no error on removing invalid file")
	}
	s1 := fmt.Sprintf("%s", err)

	err = w.Remove(testFile)
	if err == nil {
		t.Fatalf("no error on removing invalid file")
	}
	s2 := fmt.Sprintf("%s", err)

	if s1 != s2 {
		t.Fatalf("receive different error - %s / %s", s1, s2)
	}
}

func TestInotifyInnerMapLength(t *testing.T) {
	testDir := tempMkdir(t)
	defer os.RemoveAll(testDir)
	testFile := filepath.Join(testDir, "testfile")

	handle, err := os.Create(testFile)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}
	handle.Close()

	w, err := NewWatcher()
	if err != nil {
		t.Fatalf("Failed to create watcher: %v", err)
	}
	defer w.Close()

	err = w.Add(testFile)
	if err != nil {
		t.Fatalf("Failed to add testFile: %v", err)
	}
	go func() {
		for err := range w.Errors {
			t.Fatalf("error received: %s", err)
		}
	}()

	err = os.Remove(testFile)
	if err != nil {
		t.Fatalf("Failed to remove testFile: %v", err)
	}
	_ = <-w.Events                      // consume Remove event
	<-time.After(50 * time.Millisecond) // wait IN_IGNORE propagated

	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.watches) != 0 {
		t.Fatalf("Expected watches len is 0, but got: %d, %v", len(w.watches), w.watches)
	}
	if len(w.paths) != 0 {
		t.Fatalf("Expected paths len is 0, but got: %d, %v", len(w.paths), w.paths)
	}
}
