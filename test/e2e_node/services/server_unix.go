//go:build unix

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

package services

import (
	"fmt"
	"syscall"

	"k8s.io/klog/v2"
)

// pause pauses the server process using SIGSTOP.
func (s *server) pause() error {
	klog.Infof("Pausing server %q", s.name)
	cmd := s.startCommand
	if cmd == nil || cmd.Process == nil {
		return fmt.Errorf("server %q is not running", s.name)
	}
	pid := cmd.Process.Pid
	if pid <= 1 {
		return fmt.Errorf("invalid PID %d for %q", pid, s.name)
	}
	return syscall.Kill(pid, syscall.SIGSTOP)
}

// resume resumes the server process using SIGCONT.
func (s *server) resume() error {
	klog.Infof("Resuming server %q", s.name)
	cmd := s.startCommand
	if cmd == nil || cmd.Process == nil {
		return fmt.Errorf("server %q is not running", s.name)
	}
	pid := cmd.Process.Pid
	if pid <= 1 {
		return fmt.Errorf("invalid PID %d for %q", pid, s.name)
	}
	return syscall.Kill(pid, syscall.SIGCONT)
}
