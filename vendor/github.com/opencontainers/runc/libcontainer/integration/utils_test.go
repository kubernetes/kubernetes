package integration

import (
	"bytes"
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/opencontainers/runc/libcontainer"
	"github.com/opencontainers/runc/libcontainer/configs"
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

func waitProcess(p *libcontainer.Process, t *testing.T) {
	_, file, line, _ := runtime.Caller(1)
	status, err := p.Wait()

	if err != nil {
		t.Fatalf("%s:%d: unexpected error: %s\n\n", filepath.Base(file), line, err.Error())
	}

	if !status.Success() {
		t.Fatalf("%s:%d: unexpected status: %s\n\n", filepath.Base(file), line, status.String())
	}
}

func newTestRoot() (string, error) {
	dir, err := ioutil.TempDir("", "libcontainer")
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		return "", err
	}
	return dir, nil
}

func newTestBundle() (string, error) {
	dir, err := ioutil.TempDir("", "bundle")
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		return "", err
	}
	return dir, nil
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
		return "", err
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
	h := md5.New()
	h.Write([]byte(time.Now().String()))
	return newContainerWithName(hex.EncodeToString(h.Sum(nil)), config)
}

func newContainerWithName(name string, config *configs.Config) (libcontainer.Container, error) {
	f := factory
	if config.Cgroups != nil && config.Cgroups.Parent == "system.slice" {
		f = systemdFactory
	}
	return f.Create(name, config)
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
		Cwd:    "/",
		Args:   args,
		Env:    standardEnvironment,
		Stdin:  buffers.Stdin,
		Stdout: buffers.Stdout,
		Stderr: buffers.Stderr,
	}

	err = container.Run(process)
	if err != nil {
		return buffers, -1, err
	}
	ps, err := process.Wait()
	if err != nil {
		return buffers, -1, err
	}
	status := ps.Sys().(syscall.WaitStatus)
	if status.Exited() {
		exitCode = status.ExitStatus()
	} else if status.Signaled() {
		exitCode = -int(status.Signal())
	} else {
		return buffers, -1, err
	}
	return
}
