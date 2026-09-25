//go:build windows

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

package logs

import (
	"bytes"
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	apitesting "k8s.io/cri-api/pkg/apis/testing"
)

// mountPointFixture creates a real log directory and a directory junction
// (reparse point) named "var" pointing at it. It returns the real directory
// and the junction path, and skips the test if a junction cannot be created.
func mountPointFixture(t *testing.T) (targetDir, junction string) {
	t.Helper()
	dir := t.TempDir()
	targetDir = filepath.Join(dir, "real")
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		t.Fatalf("failed to create target dir: %v", err)
	}
	// Create a junction (directory reparse point) named "var" pointing at
	// the real log directory. mklink /J does not require elevated privileges.
	junction = filepath.Join(dir, "var")
	if err := os.MkdirAll(junction, 0755); err != nil {
		t.Fatalf("failed to create junction placeholder: %v", err)
	}
	if err := os.Remove(junction); err != nil {
		t.Fatalf("failed to remove junction placeholder: %v", err)
	}
	if out, err := exec.Command("cmd", "/c", "mklink", "/J", junction, targetDir).CombinedOutput(); err != nil {
		t.Skipf("unable to create directory junction, skipping: %v output=%q", err, string(out))
	}
	return targetDir, junction
}

// requireFallbackPrecondition asserts that the mount-point path cannot be
// resolved by filepath.EvalSymlinks yet is reachable by os.Stat. Without this
// guard a test could pass without ever exercising the evalSymlinks fallback,
// because a junction may be resolvable on some Windows versions.
func requireFallbackPrecondition(t *testing.T, logPath string) {
	t.Helper()
	if resolved, err := filepath.EvalSymlinks(logPath); err == nil {
		t.Skipf("filepath.EvalSymlinks resolved %q to %q; the fallback branch is not exercised on this platform", logPath, resolved)
	}
	if _, err := os.Stat(logPath); err != nil {
		t.Fatalf("precondition failed: os.Stat(%q) must succeed, got %v", logPath, err)
	}
}

// writeLogLine appends a single Docker JSON log line to path, creating the
// file if needed.
func writeLogLine(t *testing.T, path, line string) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		t.Fatalf("failed to open log file %q: %v", path, err)
	}
	defer f.Close()
	entry := `{"log":"` + line + `\n","stream":"stdout","time":"` + time.Now().UTC().Format(RFC3339NanoLenient) + `"}` + "\n"
	if _, err := f.WriteString(entry); err != nil {
		t.Fatalf("failed to write log line to %q: %v", path, err)
	}
}

// lockedBuffer is a concurrency-safe io.Writer because ReadLogs writes from a
// separate goroutine while following.
type lockedBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *lockedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *lockedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

// eventuallyContains waits until buf contains want, failing the test otherwise.
func eventuallyContains(t *testing.T, buf *lockedBuffer, want string) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if strings.Contains(buf.String(), want) {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %q in output; got %q", want, buf.String())
}

func runningContainerRuntime(containerID string) *apitesting.FakeRuntimeService {
	return &apitesting.FakeRuntimeService{
		Containers: map[string]*apitesting.FakeContainer{
			containerID: {
				ContainerStatus: runtimeapi.ContainerStatus{
					State: runtimeapi.ContainerState_CONTAINER_RUNNING,
				},
			},
		},
	}
}

// TestReadLogsMountPoint verifies that container logs can be read when the
// log path crosses a Windows mount/reparse point, e.g. when C:\var itself is
// a volume mounted from a secondary disk. filepath.EvalSymlinks cannot
// traverse such reparse points, so reading falls back to the original path
// when that path is otherwise accessible.
func TestReadLogsMountPoint(t *testing.T) {
	targetDir, junction := mountPointFixture(t)
	logFile := filepath.Join(targetDir, "0.log")
	writeLogLine(t, logFile, "line1")

	// The log path as reported by the runtime when accessed through the reparse point.
	logPath := filepath.Join(junction, "0.log")
	requireFallbackPrecondition(t, logPath)

	containerID := "fake-container-id"
	fake := runningContainerRuntime(containerID)

	stdoutBuf := &bytes.Buffer{}
	stderrBuf := &bytes.Buffer{}
	if err := ReadLogs(context.Background(), logPath, containerID, &LogOptions{}, fake, stdoutBuf, stderrBuf); err != nil {
		t.Fatalf("ReadLogs failed through reparse point: %v", err)
	}
	if got := stdoutBuf.String(); got != "line1\n" {
		t.Fatalf("expected log content, got %q", got)
	}
}

// TestReadLogsMountPointFollow verifies follow mode through the mount-point
// path: fsnotify must be able to watch the reparse-point directory and stream
// lines appended after the initial read.
func TestReadLogsMountPointFollow(t *testing.T) {
	targetDir, junction := mountPointFixture(t)
	logFile := filepath.Join(targetDir, "0.log")
	writeLogLine(t, logFile, "line1")

	logPath := filepath.Join(junction, "0.log")
	requireFallbackPrecondition(t, logPath)

	containerID := "fake-container-id"
	fake := runningContainerRuntime(containerID)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stdoutBuf := &lockedBuffer{}
	stderrBuf := &lockedBuffer{}
	done := make(chan error, 1)
	go func() {
		done <- ReadLogs(ctx, logPath, containerID, &LogOptions{Follow: true}, fake, stdoutBuf, stderrBuf)
	}()

	eventuallyContains(t, stdoutBuf, "line1\n")

	writeLogLine(t, logFile, "line2")
	eventuallyContains(t, stdoutBuf, "line2\n")

	cancel()
	select {
	case err := <-done:
		// Follow returns a context-cancelled error once the caller stops.
		if err != nil && !strings.Contains(err.Error(), "context cancelled") {
			t.Fatalf("ReadLogs returned unexpected error: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("ReadLogs did not return after context cancellation")
	}
}

// TestReadLogsMountPointFollowRotation verifies follow mode survives log
// rotation (rename + create) when the log path crosses a mount-point.
func TestReadLogsMountPointFollowRotation(t *testing.T) {
	targetDir, junction := mountPointFixture(t)
	logFile := filepath.Join(targetDir, "0.log")
	writeLogLine(t, logFile, "line1")

	logPath := filepath.Join(junction, "0.log")
	requireFallbackPrecondition(t, logPath)

	containerID := "fake-container-id"
	fake := runningContainerRuntime(containerID)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stdoutBuf := &lockedBuffer{}
	stderrBuf := &lockedBuffer{}
	done := make(chan error, 1)
	go func() {
		done <- ReadLogs(ctx, logPath, containerID, &LogOptions{Follow: true}, fake, stdoutBuf, stderrBuf)
	}()

	eventuallyContains(t, stdoutBuf, "line1\n")

	// Give ReadLogs a moment to install the fsnotify watcher on the
	// reparse-point directory before generating rotation events. Otherwise the
	// Create event can be missed and the fresh file is never reopened.
	time.Sleep(100 * time.Millisecond)

	// Rotate the log: rename it away and create a fresh file at the same path.
	if err := os.Rename(logFile, logFile+".rotated"); err != nil {
		t.Fatalf("failed to rotate log %q: %v", logFile, err)
	}
	writeLogLine(t, logFile, "line2")
	eventuallyContains(t, stdoutBuf, "line2\n")

	cancel()
	select {
	case err := <-done:
		if err != nil && !strings.Contains(err.Error(), "context cancelled") {
			t.Fatalf("ReadLogs returned unexpected error: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("ReadLogs did not return after context cancellation")
	}
}

// TestEvalSymlinks exercises the error contract of the Windows evalSymlinks
// helper directly so the fallback behavior is covered even on machines where a
// volume-mount reparse point (the scenario TestReadLogsMountPoint targets) is
// not available.
func TestEvalSymlinks(t *testing.T) {
	p := filepath.Join(t.TempDir(), "real")
	if err := os.MkdirAll(p, 0755); err != nil {
		t.Fatalf("failed to create dir: %v", err)
	}

	// A resolvable path is returned without error.
	resolved, err := evalSymlinks(p)
	if err != nil {
		t.Errorf("evalSymlinks(%q) unexpected error: %v", p, err)
	}
	if resolved == "" {
		t.Errorf("evalSymlinks(%q) resolved to empty path", p)
	}

	// A path that EvalSymlinks cannot resolve and os.Stat also cannot reach
	// must surface an error (the else branch of the fallback), so that the
	// ReadLogs contract for a genuinely missing log is preserved.
	missing := filepath.Join(p, "does-not-exist-1234")
	if _, err := evalSymlinks(missing); err == nil {
		t.Errorf("evalSymlinks(%q) expected error for missing path, got nil", missing)
	}
}

// TestEvalSymlinksFallback deterministically covers the branch where
// filepath.EvalSymlinks cannot resolve an otherwise accessible path (the
// volume-mount-point case): the original path must be returned unchanged.
func TestEvalSymlinksFallback(t *testing.T) {
	p := filepath.Join(t.TempDir(), "accessible")
	if err := os.MkdirAll(p, 0755); err != nil {
		t.Fatalf("failed to create dir: %v", err)
	}

	original := evalSymlinksFunc
	t.Cleanup(func() { evalSymlinksFunc = original })
	evalSymlinksFunc = func(string) (string, error) {
		return "", errors.New("cannot traverse reparse point")
	}

	got, err := evalSymlinks(p)
	if err != nil {
		t.Fatalf("evalSymlinks(%q) must fall back to the original path, got error: %v", p, err)
	}
	if got != p {
		t.Errorf("evalSymlinks(%q) = %q, want the original path unchanged", p, got)
	}
}
