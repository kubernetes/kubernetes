// +build !windows

package main

import (
	"encoding/json"
	"strings"
	"time"

	"github.com/docker/docker/api/types/swarm"
	"github.com/docker/docker/integration-cli/checker"
	"github.com/go-check/check"
)

func (s *DockerSwarmSuite) TestSwarmVolumePlugin(c *check.C) {
	d := s.AddDaemon(c, true, true)

	out, err := d.Cmd("service", "create", "--no-resolve-image", "--mount", "type=volume,source=my-volume,destination=/foo,volume-driver=customvolumedriver", "--name", "top", "busybox", "top")
	c.Assert(err, checker.IsNil, check.Commentf(out))

	// Make sure task stays pending before plugin is available
	waitAndAssert(c, defaultReconciliationTimeout, d.CheckServiceTasksInState("top", swarm.TaskStatePending, "missing plugin on 1 node"), checker.Equals, 1)

	plugin := newVolumePlugin(c, "customvolumedriver")
	defer plugin.Close()

	// create a dummy volume to trigger lazy loading of the plugin
	out, err = d.Cmd("volume", "create", "-d", "customvolumedriver", "hello")

	// TODO(aaronl): It will take about 15 seconds for swarm to realize the
	// plugin was loaded. Switching the test over to plugin v2 would avoid
	// this long delay.

	// make sure task has been deployed.
	waitAndAssert(c, defaultReconciliationTimeout, d.CheckActiveContainerCount, checker.Equals, 1)

	out, err = d.Cmd("ps", "-q")
	c.Assert(err, checker.IsNil)
	containerID := strings.TrimSpace(out)

	out, err = d.Cmd("inspect", "-f", "{{json .Mounts}}", containerID)
	c.Assert(err, checker.IsNil)

	var mounts []struct {
		Name   string
		Driver string
	}

	c.Assert(json.NewDecoder(strings.NewReader(out)).Decode(&mounts), checker.IsNil)
	c.Assert(len(mounts), checker.Equals, 1, check.Commentf(out))
	c.Assert(mounts[0].Name, checker.Equals, "my-volume")
	c.Assert(mounts[0].Driver, checker.Equals, "customvolumedriver")
}

// Test network plugin filter in swarm
func (s *DockerSwarmSuite) TestSwarmNetworkPluginV2(c *check.C) {
	testRequires(c, IsAmd64)
	d1 := s.AddDaemon(c, true, true)
	d2 := s.AddDaemon(c, true, false)

	// install plugin on d1 and d2
	pluginName := "aragunathan/global-net-plugin:latest"

	_, err := d1.Cmd("plugin", "install", pluginName, "--grant-all-permissions")
	c.Assert(err, checker.IsNil)

	_, err = d2.Cmd("plugin", "install", pluginName, "--grant-all-permissions")
	c.Assert(err, checker.IsNil)

	// create network
	networkName := "globalnet"
	_, err = d1.Cmd("network", "create", "--driver", pluginName, networkName)
	c.Assert(err, checker.IsNil)

	// create a global service to ensure that both nodes will have an instance
	serviceName := "my-service"
	_, err = d1.Cmd("service", "create", "--no-resolve-image", "--name", serviceName, "--mode=global", "--network", networkName, "busybox", "top")
	c.Assert(err, checker.IsNil)

	// wait for tasks ready
	waitAndAssert(c, defaultReconciliationTimeout, reducedCheck(sumAsIntegers, d1.CheckActiveContainerCount, d2.CheckActiveContainerCount), checker.Equals, 2)

	// remove service
	_, err = d1.Cmd("service", "rm", serviceName)
	c.Assert(err, checker.IsNil)

	// wait to ensure all containers have exited before removing the plugin. Else there's a
	// possibility of container exits erroring out due to plugins being unavailable.
	waitAndAssert(c, defaultReconciliationTimeout, reducedCheck(sumAsIntegers, d1.CheckActiveContainerCount, d2.CheckActiveContainerCount), checker.Equals, 0)

	// disable plugin on worker
	_, err = d2.Cmd("plugin", "disable", "-f", pluginName)
	c.Assert(err, checker.IsNil)

	time.Sleep(20 * time.Second)

	image := "busybox"
	// create a new global service again.
	_, err = d1.Cmd("service", "create", "--no-resolve-image", "--name", serviceName, "--mode=global", "--network", networkName, image, "top")
	c.Assert(err, checker.IsNil)

	waitAndAssert(c, defaultReconciliationTimeout, d1.CheckRunningTaskImages, checker.DeepEquals,
		map[string]int{image: 1})
}
