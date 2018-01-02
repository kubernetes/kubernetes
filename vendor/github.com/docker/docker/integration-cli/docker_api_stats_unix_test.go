// +build !windows

package main

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/request"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestAPIStatsContainerGetMemoryLimit(c *check.C) {
	testRequires(c, DaemonIsLinux, memoryLimitSupport)

	resp, body, err := request.Get("/info", request.JSON)
	c.Assert(err, checker.IsNil)
	c.Assert(resp.StatusCode, checker.Equals, http.StatusOK)
	var info types.Info
	err = json.NewDecoder(body).Decode(&info)
	c.Assert(err, checker.IsNil)
	body.Close()

	// don't set a memory limit, the memory limit should be system memory
	conName := "foo"
	dockerCmd(c, "run", "-d", "--name", conName, "busybox", "top")
	c.Assert(waitRun(conName), checker.IsNil)

	resp, body, err = request.Get(fmt.Sprintf("/containers/%s/stats?stream=false", conName))
	c.Assert(err, checker.IsNil)
	c.Assert(resp.StatusCode, checker.Equals, http.StatusOK)
	c.Assert(resp.Header.Get("Content-Type"), checker.Equals, "application/json")

	var v *types.Stats
	err = json.NewDecoder(body).Decode(&v)
	c.Assert(err, checker.IsNil)
	body.Close()
	c.Assert(fmt.Sprintf("%d", v.MemoryStats.Limit), checker.Equals, fmt.Sprintf("%d", info.MemTotal))
}
