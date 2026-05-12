//go:build linux && usernstest

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

package testing

import (
	"os"
	"os/exec"
	"strings"
	"sync"
	"syscall"
	"testing"
)

var (
	usernsSupported bool
	usernsOnce      sync.Once
)

// UsernsSupported returns true if unprivileged user namespaces are available
// on this host. The result is cached after the first probe.
func UsernsSupported() bool {
	usernsOnce.Do(func() {
		cmd := exec.Command("sleep", "1")
		cmd.SysProcAttr = &syscall.SysProcAttr{
			Cloneflags:  syscall.CLONE_NEWUSER,
			UidMappings: []syscall.SysProcIDMap{{ContainerID: 0, HostID: os.Getuid(), Size: 1}},
			GidMappings: []syscall.SysProcIDMap{{ContainerID: 0, HostID: os.Getgid(), Size: 1}},
		}
		if err := cmd.Start(); err != nil {
			return
		}
		defer func() {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}()
		usernsSupported = true
	})
	return usernsSupported
}

// RunInUserNS re-executes the current test binary inside a new user+network
// namespace where the calling uid/gid are mapped to root. This lets tests
// create network interfaces and install nftables/iptables rules without the
// binary itself running as root on the host.
//
// If unprivileged user namespaces are unavailable the test is skipped with a
// message explaining how to enable them (e.g. on Ubuntu 24.04).
//
// extraFlags may include additional CLONE_* flags, e.g. syscall.CLONE_NEWNET.
//
// Usage:
//
//	func TestFoo(t *testing.T) {
//	    RunInUserNS(t, testFoo_Namespaced, syscall.CLONE_NEWNET)
//	}
//	func testFoo_Namespaced(t *testing.T) { /* runs as root inside netns */ }
func RunInUserNS(t *testing.T, f func(t *testing.T), extraFlags ...uintptr) {
	t.Helper()

	const envKey = "KUBE_PROXY_USERNS_SUBPROCESS"

	// Already inside the re-executed subprocess: run the real test body.
	if os.Getenv(envKey) == "1" {
		t.Run("subprocess", f)
		return
	}

	if !UsernsSupported() {
		t.Fail("unprivileged user namespaces are not available on this host " +
			"(on Ubuntu 24.04: sysctl -w kernel.apparmor_restrict_unprivileged_userns=0)")
	}

	// Re-run only the current test inside the new namespace.
	cmd := exec.Command(os.Args[0], "-test.run="+t.Name()+"$", "-test.v=true")

	// Forward the test-log file path when the infra supplies one.
	for _, arg := range os.Args[1:] {
		if strings.HasPrefix(arg, "-test.testlogfile=") {
			cmd.Args = append(cmd.Args, arg)
		}
	}

	cmd.Env = append(os.Environ(),
		envKey+"=1",
		// Ensure networking tools (ip, nft, etc.) are on PATH inside the ns.
		"PATH=/usr/local/sbin:/usr/sbin:/sbin:"+os.Getenv("PATH"),
	)
	cmd.Stdin = os.Stdin

	cloneflags := uintptr(syscall.CLONE_NEWUSER)
	for _, f := range extraFlags {
		cloneflags |= f
	}
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Cloneflags:  cloneflags,
		UidMappings: []syscall.SysProcIDMap{{ContainerID: 0, HostID: os.Getuid(), Size: 1}},
		GidMappings: []syscall.SysProcIDMap{{ContainerID: 0, HostID: os.Getgid(), Size: 1}},
	}

	out, err := cmd.CombinedOutput()
	t.Logf("\n%s", out)
	if err != nil {
		t.Fatalf("userns subprocess failed: %v", err)
	}
}
