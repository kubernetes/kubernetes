// +build linux,cgo,seccomp

package integration

import (
	"strings"
	"syscall"
	"testing"

	"github.com/opencontainers/runc/libcontainer"
	"github.com/opencontainers/runc/libcontainer/configs"
	libseccomp "github.com/seccomp/libseccomp-golang"
)

func TestSeccompDenyGetcwd(t *testing.T) {
	if testing.Short() {
		return
	}

	rootfs, err := newRootfs()
	if err != nil {
		t.Fatal(err)
	}
	defer remove(rootfs)

	config := newTemplateConfig(rootfs)
	config.Seccomp = &configs.Seccomp{
		DefaultAction: configs.Allow,
		Syscalls: []*configs.Syscall{
			{
				Name:   "getcwd",
				Action: configs.Errno,
			},
		},
	}

	container, err := newContainer(config)
	if err != nil {
		t.Fatal(err)
	}
	defer container.Destroy()

	buffers := newStdBuffers()
	pwd := &libcontainer.Process{
		Cwd:    "/",
		Args:   []string{"pwd"},
		Env:    standardEnvironment,
		Stdin:  buffers.Stdin,
		Stdout: buffers.Stdout,
		Stderr: buffers.Stderr,
	}

	err = container.Run(pwd)
	if err != nil {
		t.Fatal(err)
	}
	ps, err := pwd.Wait()
	if err == nil {
		t.Fatal("Expecting error (negative return code); instead exited cleanly!")
	}

	var exitCode int
	status := ps.Sys().(syscall.WaitStatus)
	if status.Exited() {
		exitCode = status.ExitStatus()
	} else if status.Signaled() {
		exitCode = -int(status.Signal())
	} else {
		t.Fatalf("Unrecognized exit reason!")
	}

	if exitCode == 0 {
		t.Fatalf("Getcwd should fail with negative exit code, instead got %d!", exitCode)
	}

	expected := "pwd: getcwd: Operation not permitted"
	actual := strings.Trim(buffers.Stderr.String(), "\n")
	if actual != expected {
		t.Fatalf("Expected output %s but got %s\n", expected, actual)
	}
}

func TestSeccompPermitWriteConditional(t *testing.T) {
	if testing.Short() {
		return
	}

	rootfs, err := newRootfs()
	if err != nil {
		t.Fatal(err)
	}
	defer remove(rootfs)

	config := newTemplateConfig(rootfs)
	config.Seccomp = &configs.Seccomp{
		DefaultAction: configs.Allow,
		Syscalls: []*configs.Syscall{
			{
				Name:   "write",
				Action: configs.Errno,
				Args: []*configs.Arg{
					{
						Index: 0,
						Value: 2,
						Op:    configs.EqualTo,
					},
				},
			},
		},
	}

	container, err := newContainer(config)
	if err != nil {
		t.Fatal(err)
	}
	defer container.Destroy()

	buffers := newStdBuffers()
	dmesg := &libcontainer.Process{
		Cwd:    "/",
		Args:   []string{"busybox", "ls", "/"},
		Env:    standardEnvironment,
		Stdin:  buffers.Stdin,
		Stdout: buffers.Stdout,
		Stderr: buffers.Stderr,
	}

	err = container.Run(dmesg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := dmesg.Wait(); err != nil {
		t.Fatalf("%s: %s", err, buffers.Stderr)
	}
}

func TestSeccompDenyWriteConditional(t *testing.T) {
	if testing.Short() {
		return
	}

	// Only test if library version is v2.2.1 or higher
	// Conditional filtering will always error in v2.2.0 and lower
	major, minor, micro := libseccomp.GetLibraryVersion()
	if (major == 2 && minor < 2) || (major == 2 && minor == 2 && micro < 1) {
		return
	}

	rootfs, err := newRootfs()
	if err != nil {
		t.Fatal(err)
	}
	defer remove(rootfs)

	config := newTemplateConfig(rootfs)
	config.Seccomp = &configs.Seccomp{
		DefaultAction: configs.Allow,
		Syscalls: []*configs.Syscall{
			{
				Name:   "write",
				Action: configs.Errno,
				Args: []*configs.Arg{
					{
						Index: 0,
						Value: 2,
						Op:    configs.EqualTo,
					},
				},
			},
		},
	}

	container, err := newContainer(config)
	if err != nil {
		t.Fatal(err)
	}
	defer container.Destroy()

	buffers := newStdBuffers()
	dmesg := &libcontainer.Process{
		Cwd:    "/",
		Args:   []string{"busybox", "ls", "does_not_exist"},
		Env:    standardEnvironment,
		Stdin:  buffers.Stdin,
		Stdout: buffers.Stdout,
		Stderr: buffers.Stderr,
	}

	err = container.Run(dmesg)
	if err != nil {
		t.Fatal(err)
	}

	ps, err := dmesg.Wait()
	if err == nil {
		t.Fatal("Expecting negative return, instead got 0!")
	}

	var exitCode int
	status := ps.Sys().(syscall.WaitStatus)
	if status.Exited() {
		exitCode = status.ExitStatus()
	} else if status.Signaled() {
		exitCode = -int(status.Signal())
	} else {
		t.Fatalf("Unrecognized exit reason!")
	}

	if exitCode == 0 {
		t.Fatalf("Busybox should fail with negative exit code, instead got %d!", exitCode)
	}

	// We're denying write to stderr, so we expect an empty buffer
	expected := ""
	actual := strings.Trim(buffers.Stderr.String(), "\n")
	if actual != expected {
		t.Fatalf("Expected output %s but got %s\n", expected, actual)
	}
}
