//go:build linux

/*
Copyright 2026 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package dynamiccertificates

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

// TestDynamicFileCAContentPollReloadsAfterBindMountHidesInode covers the miss
// class from atomic replace / overlay / bind-mount: fsnotify stays on the old
// inode while os.ReadFile of the path returns new bytes. Deleting the file is
// not a valid stand-in — loadCABundle errors and the workqueue retries, which
// can reload without FileRefreshDuration.
func TestDynamicFileCAContentPollReloadsAfterBindMountHidesInode(t *testing.T) {
	orig := FileRefreshDuration
	FileRefreshDuration = 50 * time.Millisecond
	t.Cleanup(func() { FileRefreshDuration = orig })

	dir := t.TempDir()
	filename := filepath.Join(dir, "ca.crt")
	shadow := filepath.Join(dir, "ca.shadow")
	ca1 := mustCreateCA(t, "ca-one")
	ca2 := mustCreateCA(t, "ca-two")
	if err := os.WriteFile(filename, ca1, 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(shadow, ca1, 0644); err != nil {
		t.Fatal(err)
	}

	c, err := NewDynamicCAContentFromFile("test", filename)
	if err != nil {
		t.Fatal(err)
	}
	if got := c.CurrentCABundleContent(); !bytes.Equal(got, ca1) {
		t.Fatalf("initial bundle mismatch")
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go c.Run(ctx, 1)
	time.Sleep(150 * time.Millisecond)

	if err := syscall.Mount(shadow, filename, "", syscall.MS_BIND, ""); err != nil {
		t.Skipf("bind mount not permitted in this environment: %v", err)
	}
	t.Cleanup(func() { _ = syscall.Unmount(filename, syscall.MNT_DETACH) })

	// Append on the new inode. The watch still holds the hidden original inode,
	// so fsnotify should not fire; FileRefreshDuration must reload from the path.
	f, err := os.OpenFile(shadow, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(ca2); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	want, err := os.ReadFile(filename)
	if err != nil {
		t.Fatal(err)
	}
	waitForCABundle(t, c, want, 2*time.Second)
}
