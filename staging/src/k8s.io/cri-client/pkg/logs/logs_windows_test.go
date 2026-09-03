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
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	apitesting "k8s.io/cri-api/pkg/apis/testing"
)

// TestReadLogsMountPoint verifies that container logs can be read when the
// log path crosses a Windows mount/reparse point, e.g. when C:\var itself is
// a volume mounted from a secondary disk. filepath.EvalSymlinks cannot
// traverse such reparse points, so reading falls back to the original path
// when that path is otherwise accessible.
func TestReadLogsMountPoint(t *testing.T) {
	// The log file lives in a real directory that is mounted over a var
	// directory via a junction (a reparse point).
	dir := t.TempDir()
	targetDir := filepath.Join(dir, "real")
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		t.Fatalf("failed to create target dir: %v", err)
	}
	logFile := filepath.Join(targetDir, "0.log")
	logContent := `{"log":"line1\n","stream":"stdout","time":"2020-09-27T11:18:01.00000000Z"}` + "\n"
	if err := os.WriteFile(logFile, []byte(logContent), 0644); err != nil {
		t.Fatalf("failed to write log file: %v", err)
	}

	// Create a junction (directory reparse point) named "var" pointing at
	// the real log directory. mklink /J does not require elevated privileges.
	junction := filepath.Join(dir, "var")
	if err := os.MkdirAll(junction, 0755); err != nil {
		t.Fatalf("failed to create junction placeholder: %v", err)
	}
	if err := os.Remove(junction); err != nil {
		t.Fatalf("failed to remove junction placeholder: %v", err)
	}
	if out, err := exec.Command("cmd", "/c", "mklink", "/J", junction, targetDir).CombinedOutput(); err != nil {
		t.Skipf("unable to create directory junction, skipping: %v output=%q", err, string(out))
	}

	// The log path as reported by the runtime when accessed through the reparse point.
	logPath := filepath.Join(junction, "0.log")

	containerID := "fake-container-id"
	fake := &apitesting.FakeRuntimeService{
		Containers: map[string]*apitesting.FakeContainer{
			containerID: {
				ContainerStatus: runtimeapi.ContainerStatus{
					State: runtimeapi.ContainerState_CONTAINER_EXITED,
				},
			},
		},
	}

	stdoutBuf := &bytes.Buffer{}
	stderrBuf := &bytes.Buffer{}
	if err := ReadLogs(context.Background(), logPath, containerID, &LogOptions{}, fake, stdoutBuf, stderrBuf); err != nil {
		t.Fatalf("ReadLogs failed through reparse point: %v", err)
	}
	if got := stdoutBuf.String(); got != "line1\n" {
		t.Fatalf("expected log content, got %q", got)
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
