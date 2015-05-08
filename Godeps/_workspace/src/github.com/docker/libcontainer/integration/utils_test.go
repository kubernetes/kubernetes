package integration

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"

	"github.com/docker/libcontainer"
	"github.com/docker/libcontainer/configs"
)

func newStdBuffers() *stdBuffers {
	return &stdBuffers{
		Stdin:  bytes.NewBuffer(nil),
		Stdout: bytes.NewBuffer(nil),
		Stderr: bytes.NewBuffer(nil),
	}
}

type stdBuffers struct {
	Stdin  *bytes.Buffer
	Stdout *bytes.Buffer
	Stderr *bytes.Buffer
}

func (b *stdBuffers) String() string {
	s := []string{}
	if b.Stderr != nil {
		s = append(s, b.Stderr.String())
	}
	if b.Stdout != nil {
		s = append(s, b.Stdout.String())
	}
	return strings.Join(s, "|")
}

// ok fails the test if an err is not nil.
func ok(t testing.TB, err error) {
	if err != nil {
		_, file, line, _ := runtime.Caller(1)
		t.Fatalf("%s:%d: unexpected error: %s\n\n", filepath.Base(file), line, err.Error())
	}
}

// newRootfs creates a new tmp directory and copies the busybox root filesystem
func newRootfs() (string, error) {
	dir, err := ioutil.TempDir("", "")
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		return "", err
	}
	if err := copyBusybox(dir); err != nil {
		return "", nil
	}
	return dir, nil
}

func remove(dir string) {
	os.RemoveAll(dir)
}

// copyBusybox copies the rootfs for a busybox container created for the test image
// into the new directory for the specific test
func copyBusybox(dest string) error {
	out, err := exec.Command("sh", "-c", fmt.Sprintf("cp -R /busybox/* %s/", dest)).CombinedOutput()
	if err != nil {
		return fmt.Errorf("copy error %q: %q", err, out)
	}
	return nil
}

func newContainer(config *configs.Config) (libcontainer.Container, error) {
	cgm := libcontainer.Cgroupfs
	if config.Cgroups != nil && config.Cgroups.Slice == "system.slice" {
		cgm = libcontainer.SystemdCgroups
	}

	factory, err := libcontainer.New(".",
		libcontainer.InitArgs(os.Args[0], "init", "--"),
		cgm,
	)
	if err != nil {
		return nil, err
	}
	return factory.Create("testCT", config)
}

// runContainer runs the container with the specific config and arguments
//
// buffers are returned containing the STDOUT and STDERR output for the run
// along with the exit code and any go error
func runContainer(config *configs.Config, console string, args ...string) (buffers *stdBuffers, exitCode int, err error) {
	container, err := newContainer(config)
	if err != nil {
		return nil, -1, err
	}
	defer container.Destroy()
	buffers = newStdBuffers()
	process := &libcontainer.Process{
		Args:   args,
		Env:    standardEnvironment,
		Stdin:  buffers.Stdin,
		Stdout: buffers.Stdout,
		Stderr: buffers.Stderr,
	}

	err = container.Start(process)
	if err != nil {
		return nil, -1, err
	}
	ps, err := process.Wait()
	if err != nil {
		return nil, -1, err
	}
	status := ps.Sys().(syscall.WaitStatus)
	if status.Exited() {
		exitCode = status.ExitStatus()
	} else if status.Signaled() {
		exitCode = -int(status.Signal())
	} else {
		return nil, -1, err
	}
	return
}
