/*
Copyright 2025 The Kubernetes Authors.

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

package logrotation

import (
	"bytes"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestLogrotationWrite(t *testing.T) {
	tests := []struct {
		name             string
		flushInterval    time.Duration
		maxSize          int64
		maxAge           time.Duration
		rotationExpected bool
		cleanupExpected  bool
	}{
		{
			name:             "no rotation",
			flushInterval:    0,
			maxSize:          0,
			maxAge:           10 * time.Hour,
			rotationExpected: false,
			cleanupExpected:  false,
		},
		{
			name:             "Rotation with flushInterval, maxSize, no cleanup",
			flushInterval:    5 * time.Second,
			maxSize:          int64(1024 * 1024),
			maxAge:           time.Duration(0),
			rotationExpected: true,
			cleanupExpected:  false,
		},
		{
			name:             "Rotation with maxSize and cleanup with maxAge",
			flushInterval:    0,
			maxSize:          int64(1024 * 1024),
			maxAge:           24 * time.Hour,
			rotationExpected: true,
			cleanupExpected:  true,
		},
		{
			name:             "Rotation with flushInterval, maxSize and cleanup with maxAge",
			flushInterval:    10 * time.Second,
			maxSize:          int64(1024 * 1024),
			maxAge:           24 * time.Hour,
			rotationExpected: true,
			cleanupExpected:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir, err := os.MkdirTemp("", "logrotation_test")
			if err != nil {
				t.Fatalf("Failed to create temp dir: %v", err)
			}
			t.Cleanup(func() {
				if err = os.RemoveAll(tmpDir); err != nil {
					t.Errorf("Failed to remove temp directory: %v", err)
				}
			})

			logFilePath := filepath.Join(tmpDir, "test.log")
			flushInterval := tt.flushInterval
			maxSize := tt.maxSize
			maxAge := tt.maxAge

			rotationFile, err := Open(logFilePath, flushInterval, maxSize, maxAge)
			if err != nil {
				t.Fatalf("Failed to open RotationFile: %v", err)
			}

			t.Cleanup(func() {
				if err := rotationFile.Close(); err != nil {
					t.Errorf("Failed to close rotationFile: %v", err)
				}
			})

			testData := []byte("This is a test log entry.")
			n, err := rotationFile.Write(testData)
			if err != nil {
				t.Fatalf("Failed to write to RotationFile: %v", err)
			}
			if n != len(testData) {
				t.Errorf("Expected to write %d bytes, but wrote %d bytes", len(testData), n)
			}

			// Check if data is written to the file
			content, err := os.ReadFile(logFilePath)
			if err != nil {
				t.Fatalf("Failed to read log file: %v", err)
			}
			if string(content) != string(testData) {
				t.Errorf("Expected log file content to be %q, but got %q", string(testData), string(content))
			}

			// Write more data to trigger rotation
			largeData := make([]byte, 1024*1024) // 1 MB
			n, err = rotationFile.Write(largeData)
			if err != nil {
				t.Fatalf("Failed to write large data to RotationFile: %v", err)
			}
			if n != len(largeData) {
				t.Errorf("Expected to write %d bytes, but wrote %d bytes", len(largeData), n)
			}

			// Check if rotation happened
			rotatedFiles, err := filepath.Glob(filepath.Join(tmpDir, "test-*.log"))
			if err != nil {
				t.Fatalf("Failed to list rotated files: %v", err)
			}

			if tt.rotationExpected {
				numberFiles := 1
				if len(rotatedFiles) != numberFiles {
					t.Errorf("Expected %d rotated log files, but found %d", numberFiles, len(rotatedFiles))
				}

				// Check if new log file is created
				newContent, err := os.ReadFile(logFilePath)
				if err != nil {
					t.Fatalf("Failed to read new log file: %v", err)
				}
				// Rotation after the write, new log file should be empty
				if len(newContent) != 0 {
					t.Errorf("Expected new log file content to be 0 bytes, but got %d bytes", len(newContent))
				}

				// Check if content is append to the rotated log file
				content, err := os.ReadFile(rotatedFiles[0])
				if err != nil {
					t.Fatalf("Failed to read log file: %v", err)
				}
				if len(content) != len(testData)+len(largeData) {
					t.Errorf("Expected log file content to be %d bytes, but got %d bytes", len(testData)+len(largeData), len(content))
				}
			} else {
				numberFiles := 0
				if len(rotatedFiles) != numberFiles {
					t.Errorf("Expected %d rotated log files, but found %d", numberFiles, len(rotatedFiles))
				}

				// Check if content is append to the existing log file
				content, err := os.ReadFile(logFilePath)
				if err != nil {
					t.Fatalf("Failed to read log file: %v", err)
				}
				if len(content) != len(testData)+len(largeData) {
					t.Errorf("Expected log file content to be %d bytes, but got %d bytes", len(testData)+len(largeData), len(content))
				}
			}

			// Check if rotated file was able to be cleaned up
			if len(rotatedFiles) != 0 {
				err = os.Chtimes(rotatedFiles[0], time.Now(), time.Now().AddDate(0, 0, -10))
				if err != nil {
					t.Fatalf("Failed to change access time of rotated file: %v", err)
				}
			}

			// Trigger another rotation and check if the old rotated file has been cleaned up
			time.Sleep(1 * time.Second)
			largeData2 := make([]byte, 1024*1025)
			_, err = rotationFile.Write(largeData2)
			if err != nil {
				t.Fatalf("Failed to write to RotationFile: %v", err)
			}
			rotatedFiles2, err := filepath.Glob(filepath.Join(tmpDir, "test-*.log"))
			if err != nil {
				t.Fatalf("Failed to list rotated files: %v", err)
			}

			// Read the content of the log file
			content, err = os.ReadFile(logFilePath)
			if err != nil {
				t.Fatalf("Failed to read log file: %v", err)
			}

			if !tt.rotationExpected {
				if len(rotatedFiles2) != 0 {
					t.Errorf("Expected rotated log files to be 0, but found %d", len(rotatedFiles))
				}
				if len(content) != len(testData)+len(largeData)+len(largeData2) {
					t.Errorf("Expected log file content to be %d bytes, but got %d bytes", len(testData)+len(largeData)+len(largeData2), len(content))
				}
			} else {
				if tt.cleanupExpected {
					// clean up is performed in a async goroutine, wait for it to finish
					end := time.Now().Add(5 * time.Second)
					for time.Now().Before(end) {
						rotatedFiles2, err = filepath.Glob(filepath.Join(tmpDir, "test-*.log"))
						if err != nil {
							t.Fatalf("Failed to list rotated files: %v", err)
						}
						if len(rotatedFiles2) == 1 {
							break
						}
						time.Sleep(10 * time.Millisecond)
					}
					rotatedFiles2, err = filepath.Glob(filepath.Join(tmpDir, "test-*.log"))
					if err != nil {
						t.Fatalf("Failed to list rotated files: %v", err)
					}
					if len(rotatedFiles2) != 1 {
						t.Errorf("Expected rotated log files to be 1, but found %d", len(rotatedFiles2))
					}
					if rotatedFiles[0] == rotatedFiles2[0] {
						t.Errorf("Expected rotated log files to be different, but found same")
					}
				} else if len(rotatedFiles2) != 2 {
					t.Errorf("Expected rotated log files to be 2, but found %d", len(rotatedFiles2))
				}
				// Rotation after the write, new log file should be empty
				if len(content) != 0 {
					t.Errorf("Expected new log file content to be 0 bytes, but got %d bytes", len(content))
				}
			}
		})
	}
}

func TestRotationWithinSameSecondKeepsBackups(t *testing.T) {
	tmpDir := t.TempDir()
	logFilePath := filepath.Join(tmpDir, "test.log")

	rotationFile, err := Open(logFilePath, 0, 10, 0)
	if err != nil {
		t.Fatalf("Failed to open RotationFile: %v", err)
	}
	t.Cleanup(func() {
		if err := rotationFile.Close(); err != nil {
			t.Errorf("Failed to close rotationFile: %v", err)
		}
	})

	// Each write exceeds maxSize, so each one rotates, typically within the same second.
	lines := []string{"AAAAAAAAAA\n", "BBBBBBBBBB\n", "CCCCCCCCCC\n"}
	for _, line := range lines {
		if _, err := rotationFile.Write([]byte(line)); err != nil {
			t.Fatalf("Failed to write to RotationFile: %v", err)
		}
	}

	rotatedFiles, err := filepath.Glob(filepath.Join(tmpDir, "test-*.log"))
	if err != nil {
		t.Fatalf("Failed to list rotated files: %v", err)
	}
	if len(rotatedFiles) != len(lines) {
		t.Errorf("Expected %d rotated log files, but found %d: %v", len(lines), len(rotatedFiles), rotatedFiles)
	}
	var combined strings.Builder
	for _, f := range rotatedFiles {
		content, err := os.ReadFile(f)
		if err != nil {
			t.Fatalf("Failed to read rotated log file: %v", err)
		}
		combined.Write(content)
	}
	for _, line := range lines {
		if !strings.Contains(combined.String(), line) {
			t.Errorf("Expected rotated log files to contain %q, got %q", line, combined.String())
		}
	}
}

func TestCleanRemovesSuffixedBackups(t *testing.T) {
	tmpDir := t.TempDir()
	pathWithoutExt := filepath.Join(tmpDir, "test")

	old := time.Now().Add(-time.Hour)
	for _, name := range []string{"test-20250101-000000.log", "test-20250101-000000.1.log"} {
		path := filepath.Join(tmpDir, name)
		if err := os.WriteFile(path, []byte("old"), 0644); err != nil {
			t.Fatalf("Failed to create backup: %v", err)
		}
		if err := os.Chtimes(path, old, old); err != nil {
			t.Fatalf("Failed to change modification time: %v", err)
		}
	}

	w := &rotationFile{maxAge: time.Minute}
	if err := w.clean(pathWithoutExt, ".log"); err != nil {
		t.Fatalf("clean failed: %v", err)
	}

	remaining, err := filepath.Glob(filepath.Join(tmpDir, "test-*.log"))
	if err != nil {
		t.Fatalf("Failed to list rotated files: %v", err)
	}
	if len(remaining) != 0 {
		t.Errorf("Expected all expired backups to be removed, but found %v", remaining)
	}
}

func TestWriteSurvivesFailedRotation(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("directory permissions do not prevent rename on Windows")
	}
	if os.Geteuid() == 0 {
		t.Skip("root bypasses directory permissions")
	}

	tmpDir := t.TempDir()
	logFilePath := filepath.Join(tmpDir, "test.log")

	w, err := Open(logFilePath, 0, 10, 0)
	if err != nil {
		t.Fatalf("Failed to open RotationFile: %v", err)
	}
	t.Cleanup(func() {
		if err := w.Close(); err != nil {
			t.Errorf("Failed to close rotationFile: %v", err)
		}
	})

	// Make the rename in rotate() fail.
	if err := os.Chmod(tmpDir, 0555); err != nil {
		t.Fatalf("Failed to make log directory read-only: %v", err)
	}
	t.Cleanup(func() { _ = os.Chmod(tmpDir, 0755) })

	first := []byte("AAAAAAAAAAAA\n")
	n, err := w.Write(first)
	if err != nil {
		t.Fatalf("Expected a failed rotation not to fail the write, got: %v", err)
	}
	if n != len(first) {
		t.Errorf("Expected to write %d bytes, but wrote %d bytes", len(first), n)
	}

	if err := os.Chmod(tmpDir, 0755); err != nil {
		t.Fatalf("Failed to restore log directory permissions: %v", err)
	}

	// The retry is rate limited, so move the recorded failure out of the
	// interval instead of waiting for it.
	w.(*rotationFile).rotationFailedAt = time.Now().Add(-rotationRetryInterval)

	second := []byte("BBBBBBBBBBBB\n")
	if _, err := w.Write(second); err != nil {
		t.Fatalf("Expected writes to recover after the rotation failure cleared, got: %v", err)
	}

	rotatedFiles, err := filepath.Glob(filepath.Join(tmpDir, "test-*.log"))
	if err != nil {
		t.Fatalf("Failed to list rotated files: %v", err)
	}
	if len(rotatedFiles) != 1 {
		t.Fatalf("Expected 1 rotated log file once the rotation succeeded, but found %d", len(rotatedFiles))
	}
	content, err := os.ReadFile(rotatedFiles[0])
	if err != nil {
		t.Fatalf("Failed to read rotated log file: %v", err)
	}
	if want := string(first) + string(second); string(content) != want {
		t.Errorf("Expected rotated log file content %q, got %q", want, string(content))
	}
}

func TestFailedRotationIsNotRetriedOnEveryWrite(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("directory permissions do not prevent rename on Windows")
	}
	if os.Geteuid() == 0 {
		t.Skip("root bypasses directory permissions")
	}

	tmpDir := t.TempDir()
	logFilePath := filepath.Join(tmpDir, "test.log")

	w, err := Open(logFilePath, 0, 10, 0)
	if err != nil {
		t.Fatalf("Failed to open RotationFile: %v", err)
	}
	t.Cleanup(func() {
		if err := w.Close(); err != nil {
			t.Errorf("Failed to close rotationFile: %v", err)
		}
	})
	rf := w.(*rotationFile)

	// Make the rename in rotate() fail.
	if err := os.Chmod(tmpDir, 0555); err != nil {
		t.Fatalf("Failed to make log directory read-only: %v", err)
	}
	t.Cleanup(func() { _ = os.Chmod(tmpDir, 0755) })

	if _, err := w.Write([]byte("AAAAAAAAAAAA\n")); err != nil {
		t.Fatalf("Expected a failed rotation not to fail the write, got: %v", err)
	}
	failedAt := rf.rotationFailedAt
	if failedAt.IsZero() {
		t.Fatal("Expected the failed rotation to be recorded")
	}

	// currentSize stays at or above maxSize until a rotation succeeds, so
	// without the interval this write would close, rename and reopen again.
	fileBefore := rf.file
	if _, err := w.Write([]byte("BBBBBBBBBBBB\n")); err != nil {
		t.Fatalf("Expected the write to succeed while the retry is held off, got: %v", err)
	}
	if !rf.rotationFailedAt.Equal(failedAt) {
		t.Errorf("Expected no rotation attempt within %v of the failure, but one was made", rotationRetryInterval)
	}
	if rf.file != fileBefore {
		t.Error("Expected the log file not to be reopened within the retry interval")
	}

	// Once the interval has passed the next write retries, and succeeds here.
	if err := os.Chmod(tmpDir, 0755); err != nil {
		t.Fatalf("Failed to restore log directory permissions: %v", err)
	}
	rf.rotationFailedAt = time.Now().Add(-rotationRetryInterval)
	if _, err := w.Write([]byte("CCCCCCCCCCCC\n")); err != nil {
		t.Fatalf("Expected the write to succeed once the retry interval passed, got: %v", err)
	}
	if !rf.rotationFailedAt.IsZero() {
		t.Error("Expected the recorded failure to be cleared by a successful rotation")
	}

	rotatedFiles, err := filepath.Glob(filepath.Join(tmpDir, "test-*.log"))
	if err != nil {
		t.Fatalf("Failed to list rotated files: %v", err)
	}
	if len(rotatedFiles) != 1 {
		t.Fatalf("Expected 1 rotated log file once the rotation succeeded, but found %d", len(rotatedFiles))
	}
}

func TestWriteSurvivesFailedWriteAndSync(t *testing.T) {
	tmpDir := t.TempDir()
	logFilePath := filepath.Join(tmpDir, "test.log")

	// A flush interval this short makes every write sync.
	w, err := Open(logFilePath, time.Nanosecond, 0, 0)
	if err != nil {
		t.Fatalf("Failed to open RotationFile: %v", err)
	}
	rf := w.(*rotationFile)
	t.Cleanup(func() {
		if err := w.Close(); err != nil {
			t.Errorf("Failed to close rotationFile: %v", err)
		}
	})

	// Closing the file makes both the write and the sync fail, standing in for
	// the ENOSPC that a full filesystem returns.
	if err := rf.file.Close(); err != nil {
		t.Fatalf("Failed to close the underlying log file: %v", err)
	}

	dropped := []byte("AAAAAAAAAAAA\n")
	n, err := w.Write(dropped)
	if err != nil {
		t.Fatalf("Expected a failed write not to be reported, got: %v", err)
	}
	// io.Copy turns a short write into an error, which kills the wrapped
	// command just as returning one would.
	if n != len(dropped) {
		t.Errorf("Expected Write to report %d bytes, but it reported %d", len(dropped), n)
	}
	if rf.writeFailedAt.IsZero() {
		t.Error("Expected the failed write to be recorded")
	}
	if rf.syncFailedAt.IsZero() {
		t.Error("Expected the failed sync to be recorded")
	}

	// Reopening the file stands in for the disk space coming back.
	file, err := os.OpenFile(logFilePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		t.Fatalf("Failed to reopen the log file: %v", err)
	}
	rf.file = file

	kept := []byte("BBBBBBBBBBBB\n")
	if _, err := w.Write(kept); err != nil {
		t.Fatalf("Expected writes to recover, got: %v", err)
	}
	if !rf.writeFailedAt.IsZero() {
		t.Error("Expected a successful write to clear the recorded failure")
	}
	if !rf.syncFailedAt.IsZero() {
		t.Error("Expected a successful sync to clear the recorded failure")
	}

	content, err := os.ReadFile(logFilePath)
	if err != nil {
		t.Fatalf("Failed to read log file: %v", err)
	}
	if string(content) != string(kept) {
		t.Errorf("Expected log file content %q, got %q", string(kept), string(content))
	}
}

func TestFailedWriteDoesNotCountTowardRotation(t *testing.T) {
	tmpDir := t.TempDir()
	logFilePath := filepath.Join(tmpDir, "test.log")

	w, err := Open(logFilePath, 0, 10, 0)
	if err != nil {
		t.Fatalf("Failed to open RotationFile: %v", err)
	}
	rf := w.(*rotationFile)

	if err := rf.file.Close(); err != nil {
		t.Fatalf("Failed to close the underlying log file: %v", err)
	}

	// More than maxSize, but none of it reached the file.
	if _, err := w.Write([]byte("AAAAAAAAAAAA\n")); err != nil {
		t.Fatalf("Expected a failed write not to be reported, got: %v", err)
	}
	if rf.currentSize != 0 {
		t.Errorf("Expected a failed write not to count toward the size limit, but currentSize is %d", rf.currentSize)
	}

	rotatedFiles, err := filepath.Glob(filepath.Join(tmpDir, "test-*.log"))
	if err != nil {
		t.Fatalf("Failed to list rotated files: %v", err)
	}
	if len(rotatedFiles) != 0 {
		t.Errorf("Expected no rotation while writes are failing, but found %d backup files", len(rotatedFiles))
	}
}

func TestRepeatedWriteFailureIsLoggedOnce(t *testing.T) {
	var logged bytes.Buffer
	log.SetOutput(&logged)
	t.Cleanup(func() { log.SetOutput(os.Stderr) })

	tmpDir := t.TempDir()
	logFilePath := filepath.Join(tmpDir, "test.log")

	w, err := Open(logFilePath, 0, 0, 0)
	if err != nil {
		t.Fatalf("Failed to open RotationFile: %v", err)
	}
	rf := w.(*rotationFile)

	if err := rf.file.Close(); err != nil {
		t.Fatalf("Failed to close the underlying log file: %v", err)
	}

	for range 3 {
		if _, err := w.Write([]byte("AAAA\n")); err != nil {
			t.Fatalf("Expected a failed write not to be reported, got: %v", err)
		}
	}

	// The message lands on the filesystem that just failed, so one line per
	// failed log line would make a full filesystem worse.
	if got := strings.Count(logged.String(), "Failed to write to log file"); got != 1 {
		t.Errorf("Expected the run of failed writes to be logged once, but it was logged %d times", got)
	}
}

func TestWriteSurvivesLosingTheLogFile(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("directory permissions do not prevent creating a file on Windows")
	}
	if os.Geteuid() == 0 {
		t.Skip("root bypasses directory permissions")
	}

	tmpDir := t.TempDir()
	logFilePath := filepath.Join(tmpDir, "test.log")

	w, err := Open(logFilePath, 0, 10, 0)
	if err != nil {
		t.Fatalf("Failed to open RotationFile: %v", err)
	}
	rf := w.(*rotationFile)
	t.Cleanup(func() { _ = os.Chmod(tmpDir, 0755) })

	// Removing the file makes the rename in rotate() fail, and a read-only
	// directory stops it from creating a replacement, so rotate() ends up
	// with no file at all.
	if err := os.Remove(logFilePath); err != nil {
		t.Fatalf("Failed to remove the log file: %v", err)
	}
	if err := os.Chmod(tmpDir, 0555); err != nil {
		t.Fatalf("Failed to make log directory read-only: %v", err)
	}

	if _, err := w.Write([]byte("AAAAAAAAAAAA\n")); err != nil {
		t.Fatalf("Expected a failed rotation not to be reported, got: %v", err)
	}
	if rf.file != nil {
		t.Fatal("Expected the failed rotation to leave no log file")
	}

	// Writing without a file must not panic or report an error either.
	if _, err := w.Write([]byte("BBBBBBBBBBBB\n")); err != nil {
		t.Fatalf("Expected a write without a log file not to be reported, got: %v", err)
	}

	// Once the directory is writable again, the next retry reopens the file.
	if err := os.Chmod(tmpDir, 0755); err != nil {
		t.Fatalf("Failed to restore log directory permissions: %v", err)
	}
	rf.rotationFailedAt = time.Now().Add(-rotationRetryInterval)
	if _, err := w.Write([]byte("CCCCCCCCCCCC\n")); err != nil {
		t.Fatalf("Expected the retry not to be reported, got: %v", err)
	}
	if rf.file == nil {
		t.Fatal("Expected the retry to reopen the log file")
	}
	t.Cleanup(func() {
		if err := w.Close(); err != nil {
			t.Errorf("Failed to close rotationFile: %v", err)
		}
	})

	recovered := []byte("DDDDDDDDDDDD\n")
	if _, err := w.Write(recovered); err != nil {
		t.Fatalf("Expected writes to recover, got: %v", err)
	}
	if !rf.writeFailedAt.IsZero() {
		t.Error("Expected a successful write to clear the recorded failure")
	}
	content, err := os.ReadFile(logFilePath)
	if err != nil {
		t.Fatalf("Failed to read log file: %v", err)
	}
	if string(content) != string(recovered) {
		t.Errorf("Expected log file content %q, got %q", string(recovered), string(content))
	}
}
