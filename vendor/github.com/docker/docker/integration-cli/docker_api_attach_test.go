package main

import (
	"bufio"
	"bytes"
	"context"
	"io"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/client"
	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/request"
	"github.com/docker/docker/pkg/stdcopy"
	"github.com/docker/docker/pkg/testutil"
	"github.com/go-check/check"
	"golang.org/x/net/websocket"
)

func (s *DockerSuite) TestGetContainersAttachWebsocket(c *check.C) {
	testRequires(c, DaemonIsLinux)
	out, _ := dockerCmd(c, "run", "-dit", "busybox", "cat")

	rwc, err := request.SockConn(time.Duration(10*time.Second), daemonHost())
	c.Assert(err, checker.IsNil)

	cleanedContainerID := strings.TrimSpace(out)
	config, err := websocket.NewConfig(
		"/containers/"+cleanedContainerID+"/attach/ws?stream=1&stdin=1&stdout=1&stderr=1",
		"http://localhost",
	)
	c.Assert(err, checker.IsNil)

	ws, err := websocket.NewClient(config, rwc)
	c.Assert(err, checker.IsNil)
	defer ws.Close()

	expected := []byte("hello")
	actual := make([]byte, len(expected))

	outChan := make(chan error)
	go func() {
		_, err := io.ReadFull(ws, actual)
		outChan <- err
		close(outChan)
	}()

	inChan := make(chan error)
	go func() {
		_, err := ws.Write(expected)
		inChan <- err
		close(inChan)
	}()

	select {
	case err := <-inChan:
		c.Assert(err, checker.IsNil)
	case <-time.After(5 * time.Second):
		c.Fatal("Timeout writing to ws")
	}

	select {
	case err := <-outChan:
		c.Assert(err, checker.IsNil)
	case <-time.After(5 * time.Second):
		c.Fatal("Timeout reading from ws")
	}

	c.Assert(actual, checker.DeepEquals, expected, check.Commentf("Websocket didn't return the expected data"))
}

// regression gh14320
func (s *DockerSuite) TestPostContainersAttachContainerNotFound(c *check.C) {
	client, err := request.NewHTTPClient(daemonHost())
	c.Assert(err, checker.IsNil)
	req, err := request.New(daemonHost(), "/containers/doesnotexist/attach", request.Method(http.MethodPost))
	resp, err := client.Do(req)
	// connection will shutdown, err should be "persistent connection closed"
	c.Assert(resp.StatusCode, checker.Equals, http.StatusNotFound)
	content, err := testutil.ReadBody(resp.Body)
	c.Assert(err, checker.IsNil)
	expected := "No such container: doesnotexist\r\n"
	c.Assert(string(content), checker.Equals, expected)
}

func (s *DockerSuite) TestGetContainersWsAttachContainerNotFound(c *check.C) {
	status, body, err := request.SockRequest("GET", "/containers/doesnotexist/attach/ws", nil, daemonHost())
	c.Assert(status, checker.Equals, http.StatusNotFound)
	c.Assert(err, checker.IsNil)
	expected := "No such container: doesnotexist"
	c.Assert(getErrorMessage(c, body), checker.Contains, expected)
}

func (s *DockerSuite) TestPostContainersAttach(c *check.C) {
	testRequires(c, DaemonIsLinux)

	expectSuccess := func(conn net.Conn, br *bufio.Reader, stream string, tty bool) {
		defer conn.Close()
		expected := []byte("success")
		_, err := conn.Write(expected)
		c.Assert(err, checker.IsNil)

		conn.SetReadDeadline(time.Now().Add(time.Second))
		lenHeader := 0
		if !tty {
			lenHeader = 8
		}
		actual := make([]byte, len(expected)+lenHeader)
		_, err = io.ReadFull(br, actual)
		c.Assert(err, checker.IsNil)
		if !tty {
			fdMap := map[string]byte{
				"stdin":  0,
				"stdout": 1,
				"stderr": 2,
			}
			c.Assert(actual[0], checker.Equals, fdMap[stream])
		}
		c.Assert(actual[lenHeader:], checker.DeepEquals, expected, check.Commentf("Attach didn't return the expected data from %s", stream))
	}

	expectTimeout := func(conn net.Conn, br *bufio.Reader, stream string) {
		defer conn.Close()
		_, err := conn.Write([]byte{'t'})
		c.Assert(err, checker.IsNil)

		conn.SetReadDeadline(time.Now().Add(time.Second))
		actual := make([]byte, 1)
		_, err = io.ReadFull(br, actual)
		opErr, ok := err.(*net.OpError)
		c.Assert(ok, checker.Equals, true, check.Commentf("Error is expected to be *net.OpError, got %v", err))
		c.Assert(opErr.Timeout(), checker.Equals, true, check.Commentf("Read from %s is expected to timeout", stream))
	}

	// Create a container that only emits stdout.
	cid, _ := dockerCmd(c, "run", "-di", "busybox", "cat")
	cid = strings.TrimSpace(cid)
	// Attach to the container's stdout stream.
	conn, br, err := request.SockRequestHijack("POST", "/containers/"+cid+"/attach?stream=1&stdin=1&stdout=1", nil, "text/plain", daemonHost())
	c.Assert(err, checker.IsNil)
	// Check if the data from stdout can be received.
	expectSuccess(conn, br, "stdout", false)
	// Attach to the container's stderr stream.
	conn, br, err = request.SockRequestHijack("POST", "/containers/"+cid+"/attach?stream=1&stdin=1&stderr=1", nil, "text/plain", daemonHost())
	c.Assert(err, checker.IsNil)
	// Since the container only emits stdout, attaching to stderr should return nothing.
	expectTimeout(conn, br, "stdout")

	// Test the similar functions of the stderr stream.
	cid, _ = dockerCmd(c, "run", "-di", "busybox", "/bin/sh", "-c", "cat >&2")
	cid = strings.TrimSpace(cid)
	conn, br, err = request.SockRequestHijack("POST", "/containers/"+cid+"/attach?stream=1&stdin=1&stderr=1", nil, "text/plain", daemonHost())
	c.Assert(err, checker.IsNil)
	expectSuccess(conn, br, "stderr", false)
	conn, br, err = request.SockRequestHijack("POST", "/containers/"+cid+"/attach?stream=1&stdin=1&stdout=1", nil, "text/plain", daemonHost())
	c.Assert(err, checker.IsNil)
	expectTimeout(conn, br, "stderr")

	// Test with tty.
	cid, _ = dockerCmd(c, "run", "-dit", "busybox", "/bin/sh", "-c", "cat >&2")
	cid = strings.TrimSpace(cid)
	// Attach to stdout only.
	conn, br, err = request.SockRequestHijack("POST", "/containers/"+cid+"/attach?stream=1&stdin=1&stdout=1", nil, "text/plain", daemonHost())
	c.Assert(err, checker.IsNil)
	expectSuccess(conn, br, "stdout", true)

	// Attach without stdout stream.
	conn, br, err = request.SockRequestHijack("POST", "/containers/"+cid+"/attach?stream=1&stdin=1&stderr=1", nil, "text/plain", daemonHost())
	c.Assert(err, checker.IsNil)
	// Nothing should be received because both the stdout and stderr of the container will be
	// sent to the client as stdout when tty is enabled.
	expectTimeout(conn, br, "stdout")

	// Test the client API
	// Make sure we don't see "hello" if Logs is false
	client, err := client.NewEnvClient()
	c.Assert(err, checker.IsNil)

	cid, _ = dockerCmd(c, "run", "-di", "busybox", "/bin/sh", "-c", "echo hello; cat")
	cid = strings.TrimSpace(cid)

	attachOpts := types.ContainerAttachOptions{
		Stream: true,
		Stdin:  true,
		Stdout: true,
	}

	resp, err := client.ContainerAttach(context.Background(), cid, attachOpts)
	c.Assert(err, checker.IsNil)
	expectSuccess(resp.Conn, resp.Reader, "stdout", false)

	// Make sure we do see "hello" if Logs is true
	attachOpts.Logs = true
	resp, err = client.ContainerAttach(context.Background(), cid, attachOpts)
	c.Assert(err, checker.IsNil)

	defer resp.Conn.Close()
	resp.Conn.SetReadDeadline(time.Now().Add(time.Second))

	_, err = resp.Conn.Write([]byte("success"))
	c.Assert(err, checker.IsNil)

	actualStdout := new(bytes.Buffer)
	actualStderr := new(bytes.Buffer)
	stdcopy.StdCopy(actualStdout, actualStderr, resp.Reader)
	c.Assert(actualStdout.Bytes(), checker.DeepEquals, []byte("hello\nsuccess"), check.Commentf("Attach didn't return the expected data from stdout"))
}
