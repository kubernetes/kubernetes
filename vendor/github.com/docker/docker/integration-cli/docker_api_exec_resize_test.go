package main

import (
	"net/http"
	"strings"

	"github.com/go-check/check"
)

func (s *DockerSuite) TestExecResizeApiHeightWidthNoInt(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "top")
	cleanedContainerID := strings.TrimSpace(out)

	endpoint := "/exec/" + cleanedContainerID + "/resize?h=foo&w=bar"
	status, _, err := sockRequest("POST", endpoint, nil)
	c.Assert(status, check.Equals, http.StatusInternalServerError)
	c.Assert(err, check.IsNil)
}
