/*
Copyright The Kubernetes Authors.

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

package etcd

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func TestBackupDataDirectory(t *testing.T) {
	src := filepath.Join(t.TempDir(), "etcd")
	mustMkdir(t, filepath.Join(src, "member", "wal"), 0700)
	mustMkdir(t, filepath.Join(src, "member", "snap"), 0700)
	mustWriteFile(t, filepath.Join(src, "member", "wal", "0000000000000000-0000000000000000.wal"), []byte("wal"), 0600)
	mustWriteFile(t, filepath.Join(src, "member", "wal", "0.tmp"), []byte("prealloc"), 0600)
	mustWriteFile(t, filepath.Join(src, "member", "wal", "1.tmp"), []byte("prealloc"), 0600)
	mustWriteFile(t, filepath.Join(src, "member", "snap", "db"), []byte("db"), 0600)

	dst := t.TempDir()
	if err := BackupDataDirectory(src, dst); err != nil {
		t.Fatalf("BackupDataDirectory() error = %v", err)
	}

	// Layout matches `cp -r src dst` when dst exists: dst/basename(src)/...
	copiedRoot := filepath.Join(dst, "etcd")
	assertFileExists(t, filepath.Join(copiedRoot, "member", "wal", "0000000000000000-0000000000000000.wal"))
	assertFileExists(t, filepath.Join(copiedRoot, "member", "snap", "db"))
	assertFileNotExists(t, filepath.Join(copiedRoot, "member", "wal", "0.tmp"))
	assertFileNotExists(t, filepath.Join(copiedRoot, "member", "wal", "1.tmp"))
}

func TestBackupDataDirectoryToleratesWALTmpChurn(t *testing.T) {
	src := filepath.Join(t.TempDir(), "etcd")
	walDir := filepath.Join(src, "member", "wal")
	mustMkdir(t, walDir, 0700)
	mustWriteFile(t, filepath.Join(walDir, "0000000000000000-0000000000000000.wal"), []byte("wal"), 0600)

	// Simulate etcd filePipeline: create 0.tmp / 1.tmp and rename/remove them
	// while the backup walks member/wal. Plain `cp -r` fails with
	// "cannot stat '.../N.tmp'" under this churn.
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		tmp0 := filepath.Join(walDir, "0.tmp")
		tmp1 := filepath.Join(walDir, "1.tmp")
		for {
			select {
			case <-stop:
				return
			default:
				_ = os.WriteFile(tmp0, []byte("prealloc"), 0600)
				_ = os.Rename(tmp0, tmp1)
				_ = os.Remove(tmp1)
			}
		}
	}()
	t.Cleanup(func() {
		close(stop)
		wg.Wait()
	})

	for i := range 30 {
		dst := t.TempDir()
		if err := BackupDataDirectory(src, dst); err != nil {
			t.Fatalf("BackupDataDirectory() iteration %d: %v", i, err)
		}
		assertFileExists(t, filepath.Join(dst, "etcd", "member", "wal", "0000000000000000-0000000000000000.wal"))
		assertFileNotExists(t, filepath.Join(dst, "etcd", "member", "wal", "0.tmp"))
		assertFileNotExists(t, filepath.Join(dst, "etcd", "member", "wal", "1.tmp"))
	}
}

func mustMkdir(t *testing.T, path string, mode os.FileMode) {
	t.Helper()
	if err := os.MkdirAll(path, mode); err != nil {
		t.Fatal(err)
	}
}

func mustWriteFile(t *testing.T, path string, data []byte, mode os.FileMode) {
	t.Helper()
	if err := os.WriteFile(path, data, mode); err != nil {
		t.Fatal(err)
	}
}

func assertFileExists(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("expected %q to exist: %v", path, err)
	}
}

func assertFileNotExists(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("expected %q not to exist, got err=%v", path, err)
	}
}
