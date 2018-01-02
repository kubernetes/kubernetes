package main

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/request"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestPluginLogDriver(c *check.C) {
	testRequires(c, IsAmd64, DaemonIsLinux)

	pluginName := "cpuguy83/docker-logdriver-test:latest"

	dockerCmd(c, "plugin", "install", pluginName)
	dockerCmd(c, "run", "--log-driver", pluginName, "--name=test", "busybox", "echo", "hello")
	out, _ := dockerCmd(c, "logs", "test")
	c.Assert(strings.TrimSpace(out), checker.Equals, "hello")

	dockerCmd(c, "start", "-a", "test")
	out, _ = dockerCmd(c, "logs", "test")
	c.Assert(strings.TrimSpace(out), checker.Equals, "hello\nhello")

	dockerCmd(c, "rm", "test")
	dockerCmd(c, "plugin", "disable", pluginName)
	dockerCmd(c, "plugin", "rm", pluginName)
}

// Make sure log drivers are listed in info, and v2 plugins are not.
func (s *DockerSuite) TestPluginLogDriverInfoList(c *check.C) {
	testRequires(c, IsAmd64, DaemonIsLinux)
	pluginName := "cpuguy83/docker-logdriver-test"

	dockerCmd(c, "plugin", "install", pluginName)
	status, body, err := request.SockRequest("GET", "/info", nil, daemonHost())
	c.Assert(status, checker.Equals, http.StatusOK)
	c.Assert(err, checker.IsNil)

	var info types.Info
	err = json.Unmarshal(body, &info)
	c.Assert(err, checker.IsNil)
	drivers := strings.Join(info.Plugins.Log, " ")
	c.Assert(drivers, checker.Contains, "json-file")
	c.Assert(drivers, checker.Not(checker.Contains), pluginName)
}
