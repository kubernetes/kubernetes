// +build !windows

package main

import (
	"fmt"
	"strings"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/go-check/check"
)

func (s *DockerSwarmSuite) TestServiceScale(c *check.C) {
	d := s.AddDaemon(c, true, true)

	service1Name := "TestService1"
	service1Args := append([]string{"service", "create", "--no-resolve-image", "--name", service1Name, defaultSleepImage}, sleepCommandForDaemonPlatform()...)

	// global mode
	service2Name := "TestService2"
	service2Args := append([]string{"service", "create", "--no-resolve-image", "--name", service2Name, "--mode=global", defaultSleepImage}, sleepCommandForDaemonPlatform()...)

	// Create services
	out, err := d.Cmd(service1Args...)
	c.Assert(err, checker.IsNil)

	out, err = d.Cmd(service2Args...)
	c.Assert(err, checker.IsNil)

	out, err = d.Cmd("service", "scale", "TestService1=2")
	c.Assert(err, checker.IsNil)

	out, err = d.Cmd("service", "scale", "TestService1=foobar")
	c.Assert(err, checker.NotNil)

	str := fmt.Sprintf("%s: invalid replicas value %s", service1Name, "foobar")
	if !strings.Contains(out, str) {
		c.Errorf("got: %s, expected has sub string: %s", out, str)
	}

	out, err = d.Cmd("service", "scale", "TestService1=-1")
	c.Assert(err, checker.NotNil)

	str = fmt.Sprintf("%s: invalid replicas value %s", service1Name, "-1")
	if !strings.Contains(out, str) {
		c.Errorf("got: %s, expected has sub string: %s", out, str)
	}

	// TestService2 is a global mode
	out, err = d.Cmd("service", "scale", "TestService2=2")
	c.Assert(err, checker.NotNil)

	str = fmt.Sprintf("%s: scale can only be used with replicated mode\n", service2Name)
	if out != str {
		c.Errorf("got: %s, expected: %s", out, str)
	}
}
