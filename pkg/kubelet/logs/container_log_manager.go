/*
Copyright 2018 The Kubernetes Authors.

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
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"k8s.io/klog"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/wait"
	internalapi "k8s.io/kubernetes/pkg/kubelet/apis/cri"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/runtime/v1alpha2"
)

const (
	// logMonitorPeriod is the period container log manager monitors
	// container logs and performs log rotation.
	logMonitorPeriod = 10 * time.Second
	// timestampFormat is format of the timestamp suffix for rotated log.
	// See https://golang.org/pkg/time/#Time.Format.
	timestampFormat = "20060102-150405"
	// compressSuffix is the suffix for compressed log.
	compressSuffix = ".gz"
	// tmpSuffix is the suffix for temporary file.
	tmpSuffix = ".tmp"
)

// ContainerLogManager manages lifecycle of all container logs.
//
// Implementation is thread-safe.
type ContainerLogManager interface {
	// TODO(random-liu): Add RotateLogs function and call it under disk pressure.
	// Start container log manager.
	Start()
}

// LogRotatePolicy is a policy for container log rotation. The policy applies to all
// containers managed by kubelet.
type LogRotatePolicy struct {
	// MaxSize in bytes of the container log file before it is rotated. Negative
	// number means to disable container log rotation.
	MaxSize int64
	// MaxFiles is the maximum number of log files that can be present.
	// If rotating the logs creates excess files, the oldest file is removed.
	MaxFiles int
}

// GetAllLogs gets all inuse (rotated/compressed) logs for a specific container log.
// Returned logs are sorted in oldest to newest order.
// TODO(#59902): Leverage this function to support log rotation in `kubectl logs`.
func GetAllLogs(log string) ([]string, error) {
	// pattern is used to match all rotated files.
	pattern := fmt.Sprintf("%s.*", log)
	logs, err := filepath.Glob(pattern)
	if err != nil {
		return nil, fmt.Errorf("failed to list all log files with pattern %q: %v", pattern, err)
	}
	inuse, _ := filterUnusedLogs(logs)
	sort.Strings(inuse)
	return append(inuse, log), nil
}

// compressReadCloser wraps gzip.Reader with a function to close file handler.
type compressReadCloser struct {
	f *os.File
	*gzip.Reader
}

func (rc *compressReadCloser) Close() error {
	ferr := rc.f.Close()
	rerr := rc.Reader.Close()
	if ferr != nil {
		return ferr
	}
	if rerr != nil {
		return rerr
	}
	return nil
}

// UncompressLog compresses a compressed log and return a readcloser for the
// stream of the uncompressed content.
// TODO(#59902): Leverage this function to support log rotation in `kubectl logs`.
func UncompressLog(log string) (_ io.ReadCloser, retErr error) {
	if !strings.HasSuffix(log, compressSuffix) {
		return nil, fmt.Errorf("log is not compressed")
	}
	f, err := os.Open(log)
	if err != nil {
		return nil, fmt.Errorf("failed to open log: %v", err)
	}
	defer func() {
		if retErr != nil {
			f.Close()
		}
	}()
	r, err := gzip.NewReader(f)
	if err != nil {
		return nil, fmt.Errorf("failed to create gzip reader: %v", err)
	}
	return &compressReadCloser{f: f, Reader: r}, nil
}

// parseMaxSize parses quantity string to int64 max size in bytes.
func parseMaxSize(size string) (int64, error) {
	quantity, err := resource.ParseQuantity(size)
	if err != nil {
		return 0, err
	}
	maxSize, ok := quantity.AsInt64()
	if !ok {
		return 0, fmt.Errorf("invalid max log size")
	}
	if maxSize < 0 {
		return 0, fmt.Errorf("negative max log size %d", maxSize)
	}
	return maxSize, nil
}

type containerLogManager struct {
	runtimeService internalapi.RuntimeService
	policy         LogRotatePolicy
	clock          clock.Clock
}

// NewContainerLogManager creates a new container log manager.
func NewContainerLogManager(runtimeService internalapi.RuntimeService, maxSize string, maxFiles int) (ContainerLogManager, error) {
	if maxFiles <= 1 {
		return nil, fmt.Errorf("invalid MaxFiles %d, must be > 1", maxFiles)
	}
	parsedMaxSize, err := parseMaxSize(maxSize)
	if err != nil {
		return nil, fmt.Errorf("failed to parse container log max size %q: %v", maxSize, err)
	}
	// policy LogRotatePolicy
	return &containerLogManager{
		runtimeService: runtimeService,
		policy: LogRotatePolicy{
			MaxSize:  parsedMaxSize,
			MaxFiles: maxFiles,
		},
		clock: clock.RealClock{},
	}, nil
}

// Start the container log manager.
func (c *containerLogManager) Start() {
	// Start a goroutine periodically does container log rotation.
	go wait.Forever(func() {
		if err := c.rotateLogs(); err != nil {
			klog.Errorf("Failed to rotate container logs: %v", err)
		}
	}, logMonitorPeriod)
}

func (c *containerLogManager) rotateLogs() error {
	// TODO(#59998): Use kubelet pod cache.
	containers, err := c.runtimeService.ListContainers(&runtimeapi.ContainerFilter{})
	if err != nil {
		return fmt.Errorf("failed to list containers: %v", err)
	}
	// NOTE(random-liu): Figure out whether we need to rotate container logs in parallel.
	for _, container := range containers {
		// Only rotate logs for running containers. Non-running containers won't
		// generate new output, it doesn't make sense to keep an empty latest log.
		if container.GetState() != runtimeapi.ContainerState_CONTAINER_RUNNING {
			continue
		}
		id := container.GetId()
		// Note that we should not block log rotate for an error of a single container.
		status, err := c.runtimeService.ContainerStatus(id)
		if err != nil {
			klog.Errorf("Failed to get container status for %q: %v", id, err)
			continue
		}
		path := status.GetLogPath()
		info, err := os.Stat(path)
		if err != nil {
			if !os.IsNotExist(err) {
				klog.Errorf("Failed to stat container log %q: %v", path, err)
				continue
			}
			// In rotateLatestLog, there are several cases that we may
			// lose original container log after ReopenContainerLog fails.
			// We try to recover it by reopening container log.
			if err := c.runtimeService.ReopenContainerLog(id); err != nil {
				klog.Errorf("Container %q log %q doesn't exist, reopen container log failed: %v", id, path, err)
				continue
			}
			// The container log should be recovered.
			info, err = os.Stat(path)
			if err != nil {
				klog.Errorf("Failed to stat container log %q after reopen: %v", path, err)
				continue
			}
		}
		if info.Size() < c.policy.MaxSize {
			continue
		}
		// Perform log rotation.
		if err := c.rotateLog(id, path); err != nil {
			klog.Errorf("Failed to rotate log %q for container %q: %v", path, id, err)
			continue
		}
	}
	return nil
}

func (c *containerLogManager) rotateLog(id, log string) error {
	// pattern is used to match all rotated files.
	pattern := fmt.Sprintf("%s.*", log)
	logs, err := filepath.Glob(pattern)
	if err != nil {
		return fmt.Errorf("failed to list all log files with pattern %q: %v", pattern, err)
	}

	logs, err = c.cleanupUnusedLogs(logs)
	if err != nil {
		return fmt.Errorf("failed to cleanup logs: %v", err)
	}

	logs, err = c.removeExcessLogs(logs)
	if err != nil {
		return fmt.Errorf("failed to remove excess logs: %v", err)
	}

	// Compress uncompressed log files.
	for _, l := range logs {
		if strings.HasSuffix(l, compressSuffix) {
			continue
		}
		if err := c.compressLog(l); err != nil {
			return fmt.Errorf("failed to compress log %q: %v", l, err)
		}
	}

	if err := c.rotateLatestLog(id, log); err != nil {
		return fmt.Errorf("failed to rotate log %q: %v", log, err)
	}

	return nil
}

// cleanupUnusedLogs cleans up temporary or unused log files generated by previous log rotation
// failure.
func (c *containerLogManager) cleanupUnusedLogs(logs []string) ([]string, error) {
	inuse, unused := filterUnusedLogs(logs)
	for _, l := range unused {
		if err := os.Remove(l); err != nil {
			return nil, fmt.Errorf("failed to remove unused log %q: %v", l, err)
		}
	}
	return inuse, nil
}

// filterUnusedLogs splits logs into 2 groups, the 1st group is in used logs,
// the second group is unused logs.
func filterUnusedLogs(logs []string) (inuse []string, unused []string) {
	for _, l := range logs {
		if isInUse(l, logs) {
			inuse = append(inuse, l)
		} else {
			unused = append(unused, l)
		}
	}
	return inuse, unused
}

// isInUse checks whether a container log file is still inuse.
func isInUse(l string, logs []string) bool {
	// All temporary files are not in use.
	if strings.HasSuffix(l, tmpSuffix) {
		return false
	}
	// All compresed logs are in use.
	if strings.HasSuffix(l, compressSuffix) {
		return true
	}
	// Files has already been compressed are not in use.
	for _, another := range logs {
		if l+compressSuffix == another {
			return false
		}
	}
	return true
}

// removeExcessLogs removes old logs to make sure there are only at most MaxFiles log files.
func (c *containerLogManager) removeExcessLogs(logs []string) ([]string, error) {
	// Sort log files in oldest to newest order.
	sort.Strings(logs)
	// Container will create a new log file, and we'll rotate the latest log file.
	// Other than those 2 files, we can have at most MaxFiles-2 rotated log files.
	// Keep MaxFiles-2 files by removing old files.
	// We should remove from oldest to newest, so as not to break ongoing `kubectl logs`.
	maxRotatedFiles := c.policy.MaxFiles - 2
	if maxRotatedFiles < 0 {
		maxRotatedFiles = 0
	}
	i := 0
	for ; i < len(logs)-maxRotatedFiles; i++ {
		if err := os.Remove(logs[i]); err != nil {
			return nil, fmt.Errorf("failed to remove old log %q: %v", logs[i], err)
		}
	}
	logs = logs[i:]
	return logs, nil
}

// compressLog compresses a log to log.gz with gzip.
func (c *containerLogManager) compressLog(log string) error {
	r, err := os.Open(log)
	if err != nil {
		return fmt.Errorf("failed to open log %q: %v", log, err)
	}
	defer r.Close()
	tmpLog := log + tmpSuffix
	f, err := os.OpenFile(tmpLog, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return fmt.Errorf("failed to create temporary log %q: %v", tmpLog, err)
	}
	defer func() {
		// Best effort cleanup of tmpLog.
		os.Remove(tmpLog)
	}()
	defer f.Close()
	w := gzip.NewWriter(f)
	defer w.Close()
	if _, err := io.Copy(w, r); err != nil {
		return fmt.Errorf("failed to compress %q to %q: %v", log, tmpLog, err)
	}
	compressedLog := log + compressSuffix
	if err := os.Rename(tmpLog, compressedLog); err != nil {
		return fmt.Errorf("failed to rename %q to %q: %v", tmpLog, compressedLog, err)
	}
	// Remove old log file.
	if err := os.Remove(log); err != nil {
		return fmt.Errorf("failed to remove log %q after compress: %v", log, err)
	}
	return nil
}

// rotateLatestLog rotates latest log without compression, so that container can still write
// and fluentd can finish reading.
func (c *containerLogManager) rotateLatestLog(id, log string) error {
	timestamp := c.clock.Now().Format(timestampFormat)
	rotated := fmt.Sprintf("%s.%s", log, timestamp)
	if err := os.Rename(log, rotated); err != nil {
		return fmt.Errorf("failed to rotate log %q to %q: %v", log, rotated, err)
	}
	if err := c.runtimeService.ReopenContainerLog(id); err != nil {
		// Rename the rotated log back, so that we can try rotating it again
		// next round.
		// If kubelet gets restarted at this point, we'll lose original log.
		if renameErr := os.Rename(rotated, log); renameErr != nil {
			// This shouldn't happen.
			// Report an error if this happens, because we will lose original
			// log.
			klog.Errorf("Failed to rename rotated log %q back to %q: %v, reopen container log error: %v", rotated, log, renameErr, err)
		}
		return fmt.Errorf("failed to reopen container log %q: %v", id, err)
	}
	return nil
}
