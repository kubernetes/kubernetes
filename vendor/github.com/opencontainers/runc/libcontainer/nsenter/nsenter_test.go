package nsenter

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"strings"
	"syscall"
	"testing"

	"github.com/opencontainers/runc/libcontainer"
	"github.com/vishvananda/netlink/nl"
)

type pid struct {
	Pid int `json:"Pid"`
}

func TestNsenterAlivePid(t *testing.T) {
	args := []string{"nsenter-exec"}
	parent, child, err := newPipe()
	if err != nil {
		t.Fatalf("failed to create pipe %v", err)
	}

	cmd := &exec.Cmd{
		Path:       os.Args[0],
		Args:       args,
		ExtraFiles: []*os.File{child},
		Env:        []string{"_LIBCONTAINER_INITTYPE=setns", "_LIBCONTAINER_INITPIPE=3"},
	}

	if err := cmd.Start(); err != nil {
		t.Fatalf("nsenter failed to start %v", err)
	}
	r := nl.NewNetlinkRequest(int(libcontainer.InitMsg), 0)
	r.AddData(&libcontainer.Int32msg{
		Type:  libcontainer.PidAttr,
		Value: uint32(os.Getpid()),
	})
	if _, err := io.Copy(parent, bytes.NewReader(r.Serialize())); err != nil {
		t.Fatal(err)
	}
	decoder := json.NewDecoder(parent)
	var pid *pid

	if err := decoder.Decode(&pid); err != nil {
		t.Fatalf("%v", err)
	}

	if err := cmd.Wait(); err != nil {
		t.Fatalf("nsenter exits with a non-zero exit status")
	}
	p, err := os.FindProcess(pid.Pid)
	if err != nil {
		t.Fatalf("%v", err)
	}
	p.Wait()
}

func TestNsenterInvalidPid(t *testing.T) {
	args := []string{"nsenter-exec"}
	parent, child, err := newPipe()
	if err != nil {
		t.Fatalf("failed to create pipe %v", err)
	}

	cmd := &exec.Cmd{
		Path:       os.Args[0],
		Args:       args,
		ExtraFiles: []*os.File{child},
		Env:        []string{"_LIBCONTAINER_INITTYPE=setns", "_LIBCONTAINER_INITPIPE=3"},
	}

	if err := cmd.Start(); err != nil {
		t.Fatal("nsenter exits with a zero exit status")
	}
	r := nl.NewNetlinkRequest(int(libcontainer.InitMsg), 0)
	r.AddData(&libcontainer.Int32msg{
		Type:  libcontainer.PidAttr,
		Value: 0,
	})
	if _, err := io.Copy(parent, bytes.NewReader(r.Serialize())); err != nil {
		t.Fatal(err)
	}

	if err := cmd.Wait(); err == nil {
		t.Fatal("nsenter exits with a zero exit status")
	}
}

func TestNsenterDeadPid(t *testing.T) {
	deadCmd := exec.Command("true")
	if err := deadCmd.Run(); err != nil {
		t.Fatal(err)
	}
	args := []string{"nsenter-exec"}
	parent, child, err := newPipe()
	if err != nil {
		t.Fatalf("failed to create pipe %v", err)
	}

	cmd := &exec.Cmd{
		Path:       os.Args[0],
		Args:       args,
		ExtraFiles: []*os.File{child},
		Env:        []string{"_LIBCONTAINER_INITTYPE=setns", "_LIBCONTAINER_INITPIPE=3"},
	}

	if err := cmd.Start(); err != nil {
		t.Fatal("nsenter exits with a zero exit status")
	}

	r := nl.NewNetlinkRequest(int(libcontainer.InitMsg), 0)
	r.AddData(&libcontainer.Int32msg{
		Type:  libcontainer.PidAttr,
		Value: uint32(deadCmd.Process.Pid),
	})
	if _, err := io.Copy(parent, bytes.NewReader(r.Serialize())); err != nil {
		t.Fatal(err)
	}

	if err := cmd.Wait(); err == nil {
		t.Fatal("nsenter exits with a zero exit status")
	}
}

func init() {
	if strings.HasPrefix(os.Args[0], "nsenter-") {
		os.Exit(0)
	}
	return
}

func newPipe() (parent *os.File, child *os.File, err error) {
	fds, err := syscall.Socketpair(syscall.AF_LOCAL, syscall.SOCK_STREAM|syscall.SOCK_CLOEXEC, 0)
	if err != nil {
		return nil, nil, err
	}
	return os.NewFile(uintptr(fds[1]), "parent"), os.NewFile(uintptr(fds[0]), "child"), nil
}
