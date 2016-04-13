// +build !windows,!test_no_exec

package main

import (
	"bytes"
	"io"
	"os/exec"
	"strings"
	"time"

	"github.com/go-check/check"
	"github.com/kr/pty"
)

// regression test for #12546
func (s *DockerSuite) TestExecInteractiveStdinClose(c *check.C) {
	out, _ := dockerCmd(c, "run", "-itd", "busybox", "/bin/cat")
	contId := strings.TrimSpace(out)

	cmd := exec.Command(dockerBinary, "exec", "-i", contId, "echo", "-n", "hello")
	p, err := pty.Start(cmd)
	if err != nil {
		c.Fatal(err)
	}

	b := bytes.NewBuffer(nil)
	go io.Copy(b, p)

	ch := make(chan error)
	go func() { ch <- cmd.Wait() }()

	select {
	case err := <-ch:
		if err != nil {
			c.Errorf("cmd finished with error %v", err)
		}
		if output := b.String(); strings.TrimSpace(output) != "hello" {
			c.Fatalf("Unexpected output %s", output)
		}
	case <-time.After(1 * time.Second):
		c.Fatal("timed out running docker exec")
	}
}
