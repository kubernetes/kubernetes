package integration

import (
	"os"
	"strings"
	"testing"

	"github.com/docker/libcontainer"
)

func TestExecPS(t *testing.T) {
	if testing.Short() {
		return
	}

	rootfs, err := newRootFs()
	if err != nil {
		t.Fatal(err)
	}
	defer remove(rootfs)

	config := newTemplateConfig(rootfs)
	buffers, exitCode, err := runContainer(config, "", "ps")
	if err != nil {
		t.Fatal(err)
	}

	if exitCode != 0 {
		t.Fatalf("exit code not 0. code %d stderr %q", exitCode, buffers.Stderr)
	}

	lines := strings.Split(buffers.Stdout.String(), "\n")
	if len(lines) < 2 {
		t.Fatalf("more than one process running for output %q", buffers.Stdout.String())
	}
	expected := `1 root     ps`
	actual := strings.Trim(lines[1], "\n ")
	if actual != expected {
		t.Fatalf("expected output %q but received %q", expected, actual)
	}
}

func TestIPCPrivate(t *testing.T) {
	if testing.Short() {
		return
	}

	rootfs, err := newRootFs()
	if err != nil {
		t.Fatal(err)
	}
	defer remove(rootfs)

	l, err := os.Readlink("/proc/1/ns/ipc")
	if err != nil {
		t.Fatal(err)
	}

	config := newTemplateConfig(rootfs)
	buffers, exitCode, err := runContainer(config, "", "readlink", "/proc/self/ns/ipc")
	if err != nil {
		t.Fatal(err)
	}

	if exitCode != 0 {
		t.Fatalf("exit code not 0. code %d stderr %q", exitCode, buffers.Stderr)
	}

	if actual := strings.Trim(buffers.Stdout.String(), "\n"); actual == l {
		t.Fatalf("ipc link should be private to the conatiner but equals host %q %q", actual, l)
	}
}

func TestIPCHost(t *testing.T) {
	if testing.Short() {
		return
	}

	rootfs, err := newRootFs()
	if err != nil {
		t.Fatal(err)
	}
	defer remove(rootfs)

	l, err := os.Readlink("/proc/1/ns/ipc")
	if err != nil {
		t.Fatal(err)
	}

	config := newTemplateConfig(rootfs)
	config.Namespaces.Remove(libcontainer.NEWIPC)
	buffers, exitCode, err := runContainer(config, "", "readlink", "/proc/self/ns/ipc")
	if err != nil {
		t.Fatal(err)
	}

	if exitCode != 0 {
		t.Fatalf("exit code not 0. code %d stderr %q", exitCode, buffers.Stderr)
	}

	if actual := strings.Trim(buffers.Stdout.String(), "\n"); actual != l {
		t.Fatalf("ipc link not equal to host link %q %q", actual, l)
	}
}

func TestIPCJoinPath(t *testing.T) {
	if testing.Short() {
		return
	}

	rootfs, err := newRootFs()
	if err != nil {
		t.Fatal(err)
	}
	defer remove(rootfs)

	l, err := os.Readlink("/proc/1/ns/ipc")
	if err != nil {
		t.Fatal(err)
	}

	config := newTemplateConfig(rootfs)
	config.Namespaces.Add(libcontainer.NEWIPC, "/proc/1/ns/ipc")

	buffers, exitCode, err := runContainer(config, "", "readlink", "/proc/self/ns/ipc")
	if err != nil {
		t.Fatal(err)
	}

	if exitCode != 0 {
		t.Fatalf("exit code not 0. code %d stderr %q", exitCode, buffers.Stderr)
	}

	if actual := strings.Trim(buffers.Stdout.String(), "\n"); actual != l {
		t.Fatalf("ipc link not equal to host link %q %q", actual, l)
	}
}

func TestIPCBadPath(t *testing.T) {
	if testing.Short() {
		return
	}

	rootfs, err := newRootFs()
	if err != nil {
		t.Fatal(err)
	}
	defer remove(rootfs)

	config := newTemplateConfig(rootfs)
	config.Namespaces.Add(libcontainer.NEWIPC, "/proc/1/ns/ipcc")

	_, _, err = runContainer(config, "", "true")
	if err == nil {
		t.Fatal("container succeded with bad ipc path")
	}
}

func TestRlimit(t *testing.T) {
	if testing.Short() {
		return
	}

	rootfs, err := newRootFs()
	if err != nil {
		t.Fatal(err)
	}
	defer remove(rootfs)

	config := newTemplateConfig(rootfs)
	out, _, err := runContainer(config, "", "/bin/sh", "-c", "ulimit -n")
	if err != nil {
		t.Fatal(err)
	}
	if limit := strings.TrimSpace(out.Stdout.String()); limit != "1024" {
		t.Fatalf("expected rlimit to be 1024, got %s", limit)
	}
}
