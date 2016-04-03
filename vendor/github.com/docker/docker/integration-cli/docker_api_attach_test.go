package main

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httputil"
	"strings"
	"time"

	"github.com/go-check/check"
	"golang.org/x/net/websocket"
)

func (s *DockerSuite) TestGetContainersAttachWebsocket(c *check.C) {
	out, _ := dockerCmd(c, "run", "-dit", "busybox", "cat")

	rwc, err := sockConn(time.Duration(10 * time.Second))
	if err != nil {
		c.Fatal(err)
	}

	cleanedContainerID := strings.TrimSpace(out)
	config, err := websocket.NewConfig(
		"/containers/"+cleanedContainerID+"/attach/ws?stream=1&stdin=1&stdout=1&stderr=1",
		"http://localhost",
	)
	if err != nil {
		c.Fatal(err)
	}

	ws, err := websocket.NewClient(config, rwc)
	if err != nil {
		c.Fatal(err)
	}
	defer ws.Close()

	expected := []byte("hello")
	actual := make([]byte, len(expected))

	outChan := make(chan error)
	go func() {
		_, err := ws.Read(actual)
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
		if err != nil {
			c.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		c.Fatal("Timeout writing to ws")
	}

	select {
	case err := <-outChan:
		if err != nil {
			c.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		c.Fatal("Timeout reading from ws")
	}

	if !bytes.Equal(expected, actual) {
		c.Fatal("Expected output on websocket to match input")
	}
}

// regression gh14320
func (s *DockerSuite) TestPostContainersAttachContainerNotFound(c *check.C) {
	status, body, err := sockRequest("POST", "/containers/doesnotexist/attach", nil)
	c.Assert(status, check.Equals, http.StatusNotFound)
	c.Assert(err, check.IsNil)
	expected := "no such id: doesnotexist\n"
	if !strings.Contains(string(body), expected) {
		c.Fatalf("Expected response body to contain %q", expected)
	}
}

func (s *DockerSuite) TestGetContainersWsAttachContainerNotFound(c *check.C) {
	status, body, err := sockRequest("GET", "/containers/doesnotexist/attach/ws", nil)
	c.Assert(status, check.Equals, http.StatusNotFound)
	c.Assert(err, check.IsNil)
	expected := "no such id: doesnotexist\n"
	if !strings.Contains(string(body), expected) {
		c.Fatalf("Expected response body to contain %q", expected)
	}
}

func (s *DockerSuite) TestPostContainersAttach(c *check.C) {
	out, _ := dockerCmd(c, "run", "-dit", "busybox", "cat")

	r, w := io.Pipe()
	defer r.Close()
	defer w.Close()

	conn, err := sockConn(time.Duration(10 * time.Second))
	c.Assert(err, check.IsNil)

	containerID := strings.TrimSpace(out)

	req, err := http.NewRequest("POST", "/containers/"+containerID+"/attach?stream=1&stdin=1&stdout=1&stderr=1", bytes.NewReader([]byte{}))
	c.Assert(err, check.IsNil)

	client := httputil.NewClientConn(conn, nil)
	defer client.Close()

	// Do POST attach request
	resp, err := client.Do(req)
	c.Assert(resp.StatusCode, check.Equals, http.StatusOK)
	// If we check the err, we get a ErrPersistEOF = &http.ProtocolError{ErrorString: "persistent connection closed"}
	// This means that the remote requested this be the last request serviced, is this okay?

	// Test read and write to the attached container
	expected := []byte("hello")
	actual := make([]byte, len(expected))

	outChan := make(chan error)
	go func() {
		_, err := r.Read(actual)
		outChan <- err
		close(outChan)
	}()

	inChan := make(chan error)
	go func() {
		_, err := w.Write(expected)
		inChan <- err
		close(inChan)
	}()

	select {
	case err := <-inChan:
		c.Assert(err, check.IsNil)
	case <-time.After(5 * time.Second):
		c.Fatal("Timeout writing to stdout")
	}

	select {
	case err := <-outChan:
		c.Assert(err, check.IsNil)
	case <-time.After(5 * time.Second):
		c.Fatal("Timeout reading from stdin")
	}

	if !bytes.Equal(expected, actual) {
		c.Fatal("Expected output to match input")
	}

	resp.Body.Close()
}

func (s *DockerSuite) TestPostContainersAttachStderr(c *check.C) {
	out, _ := dockerCmd(c, "run", "-dit", "busybox", "/bin/sh", "-c", "cat >&2")

	r, w := io.Pipe()
	defer r.Close()
	defer w.Close()

	conn, err := sockConn(time.Duration(10 * time.Second))
	c.Assert(err, check.IsNil)

	containerID := strings.TrimSpace(out)

	req, err := http.NewRequest("POST", "/containers/"+containerID+"/attach?stream=1&stdin=1&stdout=1&stderr=1", bytes.NewReader([]byte{}))
	c.Assert(err, check.IsNil)

	client := httputil.NewClientConn(conn, nil)
	defer client.Close()

	// Do POST attach request
	resp, err := client.Do(req)
	c.Assert(resp.StatusCode, check.Equals, http.StatusOK)
	// If we check the err, we get a ErrPersistEOF = &http.ProtocolError{ErrorString: "persistent connection closed"}
	// This means that the remote requested this be the last request serviced, is this okay?

	// Test read and write to the attached container
	expected := []byte("hello")
	actual := make([]byte, len(expected))

	outChan := make(chan error)
	go func() {
		_, err := r.Read(actual)
		outChan <- err
		close(outChan)
	}()

	inChan := make(chan error)
	go func() {
		_, err := w.Write(expected)
		inChan <- err
		close(inChan)
	}()

	select {
	case err := <-inChan:
		c.Assert(err, check.IsNil)
	case <-time.After(5 * time.Second):
		c.Fatal("Timeout writing to stdout")
	}

	select {
	case err := <-outChan:
		c.Assert(err, check.IsNil)
	case <-time.After(5 * time.Second):
		c.Fatal("Timeout reading from stdin")
	}

	if !bytes.Equal(expected, actual) {
		c.Fatal("Expected output to match input")
	}

	resp.Body.Close()
}
