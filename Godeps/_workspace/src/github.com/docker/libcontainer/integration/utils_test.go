package integration

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/docker/libcontainer"
	"github.com/docker/libcontainer/namespaces"
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

func writeConfig(config *libcontainer.Config) error {
	f, err := os.OpenFile(filepath.Join(config.RootFs, "container.json"), os.O_CREATE|os.O_RDWR|os.O_TRUNC, 0700)
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewEncoder(f).Encode(config)
}

func loadConfig() (*libcontainer.Config, error) {
	f, err := os.Open(filepath.Join(os.Getenv("data_path"), "container.json"))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var container *libcontainer.Config
	if err := json.NewDecoder(f).Decode(&container); err != nil {
		return nil, err
	}
	return container, nil
}

// newRootFs creates a new tmp directory and copies the busybox root filesystem
func newRootFs() (string, error) {
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

// runContainer runs the container with the specific config and arguments
//
// buffers are returned containing the STDOUT and STDERR output for the run
// along with the exit code and any go error
func runContainer(config *libcontainer.Config, console string, args ...string) (buffers *stdBuffers, exitCode int, err error) {
	if err := writeConfig(config); err != nil {
		return nil, -1, err
	}

	buffers = newStdBuffers()
	exitCode, err = namespaces.Exec(config, buffers.Stdin, buffers.Stdout, buffers.Stderr,
		console, config.RootFs, args, namespaces.DefaultCreateCommand, nil)
	return
}
