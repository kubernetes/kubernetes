// +build !windows

package main

import (
	"encoding/json"

	"github.com/docker/docker/api/types/swarm"
	"github.com/docker/docker/integration-cli/checker"
	"github.com/go-check/check"
)

func (s *DockerSwarmSuite) TestConfigInspect(c *check.C) {
	d := s.AddDaemon(c, true, true)

	testName := "test_config"
	id := d.CreateConfig(c, swarm.ConfigSpec{
		Annotations: swarm.Annotations{
			Name: testName,
		},
		Data: []byte("TESTINGDATA"),
	})
	c.Assert(id, checker.Not(checker.Equals), "", check.Commentf("configs: %s", id))

	config := d.GetConfig(c, id)
	c.Assert(config.Spec.Name, checker.Equals, testName)

	out, err := d.Cmd("config", "inspect", testName)
	c.Assert(err, checker.IsNil, check.Commentf(out))

	var configs []swarm.Config
	c.Assert(json.Unmarshal([]byte(out), &configs), checker.IsNil)
	c.Assert(configs, checker.HasLen, 1)
}

func (s *DockerSwarmSuite) TestConfigInspectMultiple(c *check.C) {
	d := s.AddDaemon(c, true, true)

	testNames := []string{
		"test0",
		"test1",
	}
	for _, n := range testNames {
		id := d.CreateConfig(c, swarm.ConfigSpec{
			Annotations: swarm.Annotations{
				Name: n,
			},
			Data: []byte("TESTINGDATA"),
		})
		c.Assert(id, checker.Not(checker.Equals), "", check.Commentf("configs: %s", id))

		config := d.GetConfig(c, id)
		c.Assert(config.Spec.Name, checker.Equals, n)

	}

	args := []string{
		"config",
		"inspect",
	}
	args = append(args, testNames...)
	out, err := d.Cmd(args...)
	c.Assert(err, checker.IsNil, check.Commentf(out))

	var configs []swarm.Config
	c.Assert(json.Unmarshal([]byte(out), &configs), checker.IsNil)
	c.Assert(configs, checker.HasLen, 2)
}
