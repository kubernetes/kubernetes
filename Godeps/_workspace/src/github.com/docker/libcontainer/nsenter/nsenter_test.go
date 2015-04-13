package nsenter

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"
)

type pid struct {
	Pid int `json:"Pid"`
}

func TestNsenterAlivePid(t *testing.T) {
	args := []string{"nsenter-exec"}
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("failed to create pipe %v", err)
	}

	cmd := &exec.Cmd{
		Path:       os.Args[0],
		Args:       args,
		ExtraFiles: []*os.File{w},
		Env:        []string{fmt.Sprintf("_LIBCONTAINER_INITPID=%d", os.Getpid()), "_LIBCONTAINER_INITPIPE=3"},
	}

	if err := cmd.Start(); err != nil {
		t.Fatalf("nsenter failed to start %v", err)
	}
	w.Close()

	decoder := json.NewDecoder(r)
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

	cmd := &exec.Cmd{
		Path: os.Args[0],
		Args: args,
		Env:  []string{"_LIBCONTAINER_INITPID=-1"},
	}

	err := cmd.Run()
	if err == nil {
		t.Fatal("nsenter exits with a zero exit status")
	}
}

func TestNsenterDeadPid(t *testing.T) {
	dead_cmd := exec.Command("true")
	if err := dead_cmd.Run(); err != nil {
		t.Fatal(err)
	}
	args := []string{"nsenter-exec"}

	cmd := &exec.Cmd{
		Path: os.Args[0],
		Args: args,
		Env:  []string{fmt.Sprintf("_LIBCONTAINER_INITPID=%d", dead_cmd.Process.Pid)},
	}

	err := cmd.Run()
	if err == nil {
		t.Fatal("nsenter exits with a zero exit status")
	}
}

func init() {
	if strings.HasPrefix(os.Args[0], "nsenter-") {
		os.Exit(0)
	}
	return
}
