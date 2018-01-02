// +build !windows

package main

import (
	"strings"

	"github.com/docker/docker/api/types/swarm"
	"github.com/docker/docker/integration-cli/checker"
	"github.com/go-check/check"
)

func (s *DockerSwarmSuite) TestSecretList(c *check.C) {
	d := s.AddDaemon(c, true, true)

	testName0 := "test0"
	testName1 := "test1"

	// create secret test0
	id0 := d.CreateSecret(c, swarm.SecretSpec{
		Annotations: swarm.Annotations{
			Name:   testName0,
			Labels: map[string]string{"type": "test"},
		},
		Data: []byte("TESTINGDATA0"),
	})
	c.Assert(id0, checker.Not(checker.Equals), "", check.Commentf("secrets: %s", id0))

	secret := d.GetSecret(c, id0)
	c.Assert(secret.Spec.Name, checker.Equals, testName0)

	// create secret test1
	id1 := d.CreateSecret(c, swarm.SecretSpec{
		Annotations: swarm.Annotations{
			Name:   testName1,
			Labels: map[string]string{"type": "production"},
		},
		Data: []byte("TESTINGDATA1"),
	})
	c.Assert(id1, checker.Not(checker.Equals), "", check.Commentf("secrets: %s", id1))

	secret = d.GetSecret(c, id1)
	c.Assert(secret.Spec.Name, checker.Equals, testName1)

	// test by command `docker secret ls`
	out, err := d.Cmd("secret", "ls")
	c.Assert(err, checker.IsNil, check.Commentf(out))
	c.Assert(strings.TrimSpace(out), checker.Contains, testName0)
	c.Assert(strings.TrimSpace(out), checker.Contains, testName1)

	// test filter by name `docker secret ls --filter name=xxx`
	args := []string{
		"secret",
		"ls",
		"--filter",
		"name=test0",
	}
	out, err = d.Cmd(args...)
	c.Assert(err, checker.IsNil, check.Commentf(out))

	c.Assert(strings.TrimSpace(out), checker.Contains, testName0)
	c.Assert(strings.TrimSpace(out), checker.Not(checker.Contains), testName1)

	// test filter by id `docker secret ls --filter id=xxx`
	args = []string{
		"secret",
		"ls",
		"--filter",
		"id=" + id1,
	}
	out, err = d.Cmd(args...)
	c.Assert(err, checker.IsNil, check.Commentf(out))

	c.Assert(strings.TrimSpace(out), checker.Not(checker.Contains), testName0)
	c.Assert(strings.TrimSpace(out), checker.Contains, testName1)

	// test filter by label `docker secret ls --filter label=xxx`
	args = []string{
		"secret",
		"ls",
		"--filter",
		"label=type",
	}
	out, err = d.Cmd(args...)
	c.Assert(err, checker.IsNil, check.Commentf(out))

	c.Assert(strings.TrimSpace(out), checker.Contains, testName0)
	c.Assert(strings.TrimSpace(out), checker.Contains, testName1)

	args = []string{
		"secret",
		"ls",
		"--filter",
		"label=type=test",
	}
	out, err = d.Cmd(args...)
	c.Assert(err, checker.IsNil, check.Commentf(out))

	c.Assert(strings.TrimSpace(out), checker.Contains, testName0)
	c.Assert(strings.TrimSpace(out), checker.Not(checker.Contains), testName1)

	args = []string{
		"secret",
		"ls",
		"--filter",
		"label=type=production",
	}
	out, err = d.Cmd(args...)
	c.Assert(err, checker.IsNil, check.Commentf(out))

	c.Assert(strings.TrimSpace(out), checker.Not(checker.Contains), testName0)
	c.Assert(strings.TrimSpace(out), checker.Contains, testName1)

	// test invalid filter `docker secret ls --filter noexisttype=xxx`
	args = []string{
		"secret",
		"ls",
		"--filter",
		"noexisttype=test0",
	}
	out, err = d.Cmd(args...)
	c.Assert(err, checker.NotNil, check.Commentf(out))

	c.Assert(strings.TrimSpace(out), checker.Contains, "Error response from daemon: Invalid filter 'noexisttype'")
}
