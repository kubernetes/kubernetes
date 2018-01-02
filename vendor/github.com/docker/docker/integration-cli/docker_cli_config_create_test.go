// +build !windows

package main

import (
	"io/ioutil"
	"os"
	"strings"

	"github.com/docker/docker/api/types/swarm"
	"github.com/docker/docker/integration-cli/checker"
	"github.com/go-check/check"
)

func (s *DockerSwarmSuite) TestConfigCreate(c *check.C) {
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
}

func (s *DockerSwarmSuite) TestConfigCreateWithLabels(c *check.C) {
	d := s.AddDaemon(c, true, true)

	testName := "test_config"
	id := d.CreateConfig(c, swarm.ConfigSpec{
		Annotations: swarm.Annotations{
			Name: testName,
			Labels: map[string]string{
				"key1": "value1",
				"key2": "value2",
			},
		},
		Data: []byte("TESTINGDATA"),
	})
	c.Assert(id, checker.Not(checker.Equals), "", check.Commentf("configs: %s", id))

	config := d.GetConfig(c, id)
	c.Assert(config.Spec.Name, checker.Equals, testName)
	c.Assert(len(config.Spec.Labels), checker.Equals, 2)
	c.Assert(config.Spec.Labels["key1"], checker.Equals, "value1")
	c.Assert(config.Spec.Labels["key2"], checker.Equals, "value2")
}

// Test case for 28884
func (s *DockerSwarmSuite) TestConfigCreateResolve(c *check.C) {
	d := s.AddDaemon(c, true, true)

	name := "test_config"
	id := d.CreateConfig(c, swarm.ConfigSpec{
		Annotations: swarm.Annotations{
			Name: name,
		},
		Data: []byte("foo"),
	})
	c.Assert(id, checker.Not(checker.Equals), "", check.Commentf("configs: %s", id))

	fake := d.CreateConfig(c, swarm.ConfigSpec{
		Annotations: swarm.Annotations{
			Name: id,
		},
		Data: []byte("fake foo"),
	})
	c.Assert(fake, checker.Not(checker.Equals), "", check.Commentf("configs: %s", fake))

	out, err := d.Cmd("config", "ls")
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Contains, name)
	c.Assert(out, checker.Contains, fake)

	out, err = d.Cmd("config", "rm", id)
	c.Assert(out, checker.Contains, id)

	// Fake one will remain
	out, err = d.Cmd("config", "ls")
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Not(checker.Contains), name)
	c.Assert(out, checker.Contains, fake)

	// Remove based on name prefix of the fake one
	// (which is the same as the ID of foo one) should not work
	// as search is only done based on:
	// - Full ID
	// - Full Name
	// - Partial ID (prefix)
	out, err = d.Cmd("config", "rm", id[:5])
	c.Assert(out, checker.Not(checker.Contains), id)
	out, err = d.Cmd("config", "ls")
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Not(checker.Contains), name)
	c.Assert(out, checker.Contains, fake)

	// Remove based on ID prefix of the fake one should succeed
	out, err = d.Cmd("config", "rm", fake[:5])
	c.Assert(out, checker.Contains, fake[:5])
	out, err = d.Cmd("config", "ls")
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Not(checker.Contains), name)
	c.Assert(out, checker.Not(checker.Contains), id)
	c.Assert(out, checker.Not(checker.Contains), fake)
}

func (s *DockerSwarmSuite) TestConfigCreateWithFile(c *check.C) {
	d := s.AddDaemon(c, true, true)

	testFile, err := ioutil.TempFile("", "configCreateTest")
	c.Assert(err, checker.IsNil, check.Commentf("failed to create temporary file"))
	defer os.Remove(testFile.Name())

	testData := "TESTINGDATA"
	_, err = testFile.Write([]byte(testData))
	c.Assert(err, checker.IsNil, check.Commentf("failed to write to temporary file"))

	testName := "test_config"
	out, err := d.Cmd("config", "create", testName, testFile.Name())
	c.Assert(err, checker.IsNil)
	c.Assert(strings.TrimSpace(out), checker.Not(checker.Equals), "", check.Commentf(out))

	id := strings.TrimSpace(out)
	config := d.GetConfig(c, id)
	c.Assert(config.Spec.Name, checker.Equals, testName)
}
