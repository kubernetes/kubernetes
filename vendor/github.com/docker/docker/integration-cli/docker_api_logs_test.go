package main

import (
	"bufio"
	"bytes"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/go-check/check"
)

func (s *DockerSuite) TestLogsApiWithStdout(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "-t", "busybox", "/bin/sh", "-c", "while true; do echo hello; sleep 1; done")
	id := strings.TrimSpace(out)
	if err := waitRun(id); err != nil {
		c.Fatal(err)
	}

	type logOut struct {
		out string
		res *http.Response
		err error
	}
	chLog := make(chan logOut)

	go func() {
		res, body, err := sockRequestRaw("GET", fmt.Sprintf("/containers/%s/logs?follow=1&stdout=1&timestamps=1", id), nil, "")
		out, _ := bufio.NewReader(body).ReadString('\n')
		chLog <- logOut{strings.TrimSpace(out), res, err}
	}()

	select {
	case l := <-chLog:
		c.Assert(l.err, check.IsNil)
		c.Assert(l.res.StatusCode, check.Equals, http.StatusOK)
		if !strings.HasSuffix(l.out, "hello") {
			c.Fatalf("expected log output to container 'hello', but it does not")
		}
	case <-time.After(2 * time.Second):
		c.Fatal("timeout waiting for logs to exit")
	}
}

func (s *DockerSuite) TestLogsApiNoStdoutNorStderr(c *check.C) {
	name := "logs_test"
	dockerCmd(c, "run", "-d", "-t", "--name", name, "busybox", "/bin/sh")

	status, body, err := sockRequest("GET", fmt.Sprintf("/containers/%s/logs", name), nil)
	c.Assert(status, check.Equals, http.StatusBadRequest)
	c.Assert(err, check.IsNil)

	expected := "Bad parameters: you must choose at least one stream"
	if !bytes.Contains(body, []byte(expected)) {
		c.Fatalf("Expected %s, got %s", expected, string(body[:]))
	}
}

// Regression test for #12704
func (s *DockerSuite) TestLogsApiFollowEmptyOutput(c *check.C) {
	name := "logs_test"
	t0 := time.Now()
	dockerCmd(c, "run", "-d", "-t", "--name", name, "busybox", "sleep", "10")

	_, body, err := sockRequestRaw("GET", fmt.Sprintf("/containers/%s/logs?follow=1&stdout=1&stderr=1&tail=all", name), bytes.NewBuffer(nil), "")
	t1 := time.Now()
	body.Close()
	if err != nil {
		c.Fatal(err)
	}
	elapsed := t1.Sub(t0).Seconds()
	if elapsed > 5.0 {
		c.Fatalf("HTTP response was not immediate (elapsed %.1fs)", elapsed)
	}
}
