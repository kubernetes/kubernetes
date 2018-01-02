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

func (s *DockerSwarmSuite) TestSecretCreate(c *check.C) {
	d := s.AddDaemon(c, true, true)

	testName := "test_secret"
	id := d.CreateSecret(c, swarm.SecretSpec{
		Annotations: swarm.Annotations{
			Name: testName,
		},
		Data: []byte("TESTINGDATA"),
	})
	c.Assert(id, checker.Not(checker.Equals), "", check.Commentf("secrets: %s", id))

	secret := d.GetSecret(c, id)
	c.Assert(secret.Spec.Name, checker.Equals, testName)
}

func (s *DockerSwarmSuite) TestSecretCreateWithLabels(c *check.C) {
	d := s.AddDaemon(c, true, true)

	testName := "test_secret"
	id := d.CreateSecret(c, swarm.SecretSpec{
		Annotations: swarm.Annotations{
			Name: testName,
			Labels: map[string]string{
				"key1": "value1",
				"key2": "value2",
			},
		},
		Data: []byte("TESTINGDATA"),
	})
	c.Assert(id, checker.Not(checker.Equals), "", check.Commentf("secrets: %s", id))

	secret := d.GetSecret(c, id)
	c.Assert(secret.Spec.Name, checker.Equals, testName)
	c.Assert(len(secret.Spec.Labels), checker.Equals, 2)
	c.Assert(secret.Spec.Labels["key1"], checker.Equals, "value1")
	c.Assert(secret.Spec.Labels["key2"], checker.Equals, "value2")
}

// Test case for 28884
func (s *DockerSwarmSuite) TestSecretCreateResolve(c *check.C) {
	d := s.AddDaemon(c, true, true)

	name := "test_secret"
	id := d.CreateSecret(c, swarm.SecretSpec{
		Annotations: swarm.Annotations{
			Name: name,
		},
		Data: []byte("foo"),
	})
	c.Assert(id, checker.Not(checker.Equals), "", check.Commentf("secrets: %s", id))

	fake := d.CreateSecret(c, swarm.SecretSpec{
		Annotations: swarm.Annotations{
			Name: id,
		},
		Data: []byte("fake foo"),
	})
	c.Assert(fake, checker.Not(checker.Equals), "", check.Commentf("secrets: %s", fake))

	out, err := d.Cmd("secret", "ls")
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Contains, name)
	c.Assert(out, checker.Contains, fake)

	out, err = d.Cmd("secret", "rm", id)
	c.Assert(out, checker.Contains, id)

	// Fake one will remain
	out, err = d.Cmd("secret", "ls")
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Not(checker.Contains), name)
	c.Assert(out, checker.Contains, fake)

	// Remove based on name prefix of the fake one
	// (which is the same as the ID of foo one) should not work
	// as search is only done based on:
	// - Full ID
	// - Full Name
	// - Partial ID (prefix)
	out, err = d.Cmd("secret", "rm", id[:5])
	c.Assert(out, checker.Not(checker.Contains), id)
	out, err = d.Cmd("secret", "ls")
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Not(checker.Contains), name)
	c.Assert(out, checker.Contains, fake)

	// Remove based on ID prefix of the fake one should succeed
	out, err = d.Cmd("secret", "rm", fake[:5])
	c.Assert(out, checker.Contains, fake[:5])
	out, err = d.Cmd("secret", "ls")
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Not(checker.Contains), name)
	c.Assert(out, checker.Not(checker.Contains), id)
	c.Assert(out, checker.Not(checker.Contains), fake)
}

func (s *DockerSwarmSuite) TestSecretCreateWithFile(c *check.C) {
	d := s.AddDaemon(c, true, true)

	testFile, err := ioutil.TempFile("", "secretCreateTest")
	c.Assert(err, checker.IsNil, check.Commentf("failed to create temporary file"))
	defer os.Remove(testFile.Name())

	testData := "TESTINGDATA"
	_, err = testFile.Write([]byte(testData))
	c.Assert(err, checker.IsNil, check.Commentf("failed to write to temporary file"))

	testName := "test_secret"
	out, err := d.Cmd("secret", "create", testName, testFile.Name())
	c.Assert(err, checker.IsNil)
	c.Assert(strings.TrimSpace(out), checker.Not(checker.Equals), "", check.Commentf(out))

	id := strings.TrimSpace(out)
	secret := d.GetSecret(c, id)
	c.Assert(secret.Spec.Name, checker.Equals, testName)
}
