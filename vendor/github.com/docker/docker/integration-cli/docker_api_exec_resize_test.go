package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/request"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestExecResizeAPIHeightWidthNoInt(c *check.C) {
	testRequires(c, DaemonIsLinux)
	out, _ := dockerCmd(c, "run", "-d", "busybox", "top")
	cleanedContainerID := strings.TrimSpace(out)

	endpoint := "/exec/" + cleanedContainerID + "/resize?h=foo&w=bar"
	status, _, err := request.SockRequest("POST", endpoint, nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusInternalServerError)
}

// Part of #14845
func (s *DockerSuite) TestExecResizeImmediatelyAfterExecStart(c *check.C) {
	name := "exec_resize_test"
	dockerCmd(c, "run", "-d", "-i", "-t", "--name", name, "--restart", "always", "busybox", "/bin/sh")

	testExecResize := func() error {
		data := map[string]interface{}{
			"AttachStdin": true,
			"Cmd":         []string{"/bin/sh"},
		}
		uri := fmt.Sprintf("/containers/%s/exec", name)
		status, body, err := request.SockRequest("POST", uri, data, daemonHost())
		if err != nil {
			return err
		}
		if status != http.StatusCreated {
			return fmt.Errorf("POST %s is expected to return %d, got %d", uri, http.StatusCreated, status)
		}

		out := map[string]string{}
		err = json.Unmarshal(body, &out)
		if err != nil {
			return fmt.Errorf("ExecCreate returned invalid json. Error: %q", err.Error())
		}

		execID := out["Id"]
		if len(execID) < 1 {
			return fmt.Errorf("ExecCreate got invalid execID")
		}

		payload := bytes.NewBufferString(`{"Tty":true}`)
		conn, _, err := request.SockRequestHijack("POST", fmt.Sprintf("/exec/%s/start", execID), payload, "application/json", daemonHost())
		if err != nil {
			return fmt.Errorf("Failed to start the exec: %q", err.Error())
		}
		defer conn.Close()

		_, rc, err := request.Post(fmt.Sprintf("/exec/%s/resize?h=24&w=80", execID), request.ContentType("text/plain"))
		// It's probably a panic of the daemon if io.ErrUnexpectedEOF is returned.
		if err == io.ErrUnexpectedEOF {
			return fmt.Errorf("The daemon might have crashed.")
		}

		if err == nil {
			rc.Close()
		}

		// We only interested in the io.ErrUnexpectedEOF error, so we return nil otherwise.
		return nil
	}

	// The panic happens when daemon.ContainerExecStart is called but the
	// container.Exec is not called.
	// Because the panic is not 100% reproducible, we send the requests concurrently
	// to increase the probability that the problem is triggered.
	var (
		n  = 10
		ch = make(chan error, n)
		wg sync.WaitGroup
	)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := testExecResize(); err != nil {
				ch <- err
			}
		}()
	}

	wg.Wait()
	select {
	case err := <-ch:
		c.Fatal(err.Error())
	default:
	}
}
