package main

import (
	"bufio"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/request"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestLogsAPIWithStdout(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "-t", "busybox", "/bin/sh", "-c", "while true; do echo hello; sleep 1; done")
	id := strings.TrimSpace(out)
	c.Assert(waitRun(id), checker.IsNil)

	type logOut struct {
		out string
		err error
	}

	chLog := make(chan logOut)
	res, body, err := request.Get(fmt.Sprintf("/containers/%s/logs?follow=1&stdout=1&timestamps=1", id))
	c.Assert(err, checker.IsNil)
	c.Assert(res.StatusCode, checker.Equals, http.StatusOK)

	go func() {
		defer body.Close()
		out, err := bufio.NewReader(body).ReadString('\n')
		if err != nil {
			chLog <- logOut{"", err}
			return
		}
		chLog <- logOut{strings.TrimSpace(out), err}
	}()

	select {
	case l := <-chLog:
		c.Assert(l.err, checker.IsNil)
		if !strings.HasSuffix(l.out, "hello") {
			c.Fatalf("expected log output to container 'hello', but it does not")
		}
	case <-time.After(30 * time.Second):
		c.Fatal("timeout waiting for logs to exit")
	}
}

func (s *DockerSuite) TestLogsAPINoStdoutNorStderr(c *check.C) {
	name := "logs_test"
	dockerCmd(c, "run", "-d", "-t", "--name", name, "busybox", "/bin/sh")

	status, body, err := request.SockRequest("GET", fmt.Sprintf("/containers/%s/logs", name), nil, daemonHost())
	c.Assert(status, checker.Equals, http.StatusBadRequest)
	c.Assert(err, checker.IsNil)

	expected := "Bad parameters: you must choose at least one stream"
	c.Assert(getErrorMessage(c, body), checker.Contains, expected)
}

// Regression test for #12704
func (s *DockerSuite) TestLogsAPIFollowEmptyOutput(c *check.C) {
	name := "logs_test"
	t0 := time.Now()
	dockerCmd(c, "run", "-d", "-t", "--name", name, "busybox", "sleep", "10")

	_, body, err := request.Get(fmt.Sprintf("/containers/%s/logs?follow=1&stdout=1&stderr=1&tail=all", name))
	t1 := time.Now()
	c.Assert(err, checker.IsNil)
	body.Close()
	elapsed := t1.Sub(t0).Seconds()
	if elapsed > 20.0 {
		c.Fatalf("HTTP response was not immediate (elapsed %.1fs)", elapsed)
	}
}

func (s *DockerSuite) TestLogsAPIContainerNotFound(c *check.C) {
	name := "nonExistentContainer"
	resp, _, err := request.Get(fmt.Sprintf("/containers/%s/logs?follow=1&stdout=1&stderr=1&tail=all", name))
	c.Assert(err, checker.IsNil)
	c.Assert(resp.StatusCode, checker.Equals, http.StatusNotFound)
}
