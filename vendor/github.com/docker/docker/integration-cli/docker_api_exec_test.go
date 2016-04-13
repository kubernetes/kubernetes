// +build !test_no_exec

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/go-check/check"
)

// Regression test for #9414
func (s *DockerSuite) TestExecApiCreateNoCmd(c *check.C) {
	name := "exec_test"
	dockerCmd(c, "run", "-d", "-t", "--name", name, "busybox", "/bin/sh")

	status, body, err := sockRequest("POST", fmt.Sprintf("/containers/%s/exec", name), map[string]interface{}{"Cmd": nil})
	c.Assert(status, check.Equals, http.StatusInternalServerError)
	c.Assert(err, check.IsNil)

	if !bytes.Contains(body, []byte("No exec command specified")) {
		c.Fatalf("Expected message when creating exec command with no Cmd specified")
	}
}

func (s *DockerSuite) TestExecApiCreateNoValidContentType(c *check.C) {
	name := "exec_test"
	dockerCmd(c, "run", "-d", "-t", "--name", name, "busybox", "/bin/sh")

	jsonData := bytes.NewBuffer(nil)
	if err := json.NewEncoder(jsonData).Encode(map[string]interface{}{"Cmd": nil}); err != nil {
		c.Fatalf("Can not encode data to json %s", err)
	}

	res, body, err := sockRequestRaw("POST", fmt.Sprintf("/containers/%s/exec", name), jsonData, "text/plain")
	c.Assert(err, check.IsNil)
	c.Assert(res.StatusCode, check.Equals, http.StatusInternalServerError)

	b, err := readBody(body)
	c.Assert(err, check.IsNil)

	if !bytes.Contains(b, []byte("Content-Type specified")) {
		c.Fatalf("Expected message when creating exec command with invalid Content-Type specified")
	}
}
