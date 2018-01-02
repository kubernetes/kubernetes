// +build !windows

package main

import (
	"fmt"
	"net/http"

	"github.com/docker/docker/api/types/swarm"
	"github.com/docker/docker/integration-cli/checker"
	"github.com/go-check/check"
)

func (s *DockerSwarmSuite) TestAPISwarmSecretsEmptyList(c *check.C) {
	d := s.AddDaemon(c, true, true)

	secrets := d.ListSecrets(c)
	c.Assert(secrets, checker.NotNil)
	c.Assert(len(secrets), checker.Equals, 0, check.Commentf("secrets: %#v", secrets))
}

func (s *DockerSwarmSuite) TestAPISwarmSecretsCreate(c *check.C) {
	d := s.AddDaemon(c, true, true)

	testName := "test_secret"
	secretSpec := swarm.SecretSpec{
		Annotations: swarm.Annotations{
			Name: testName,
		},
		Data: []byte("TESTINGDATA"),
	}

	id := d.CreateSecret(c, secretSpec)
	c.Assert(id, checker.Not(checker.Equals), "", check.Commentf("secrets: %s", id))

	secrets := d.ListSecrets(c)
	c.Assert(len(secrets), checker.Equals, 1, check.Commentf("secrets: %#v", secrets))
	name := secrets[0].Spec.Annotations.Name
	c.Assert(name, checker.Equals, testName, check.Commentf("secret: %s", name))

	// create an already existing secret, daemon should return a status code of 409
	status, out, err := d.SockRequest("POST", "/secrets/create", secretSpec)
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusConflict, check.Commentf("secret create: %s", string(out)))
}

func (s *DockerSwarmSuite) TestAPISwarmSecretsDelete(c *check.C) {
	d := s.AddDaemon(c, true, true)

	testName := "test_secret"
	id := d.CreateSecret(c, swarm.SecretSpec{Annotations: swarm.Annotations{
		Name: testName,
	},
		Data: []byte("TESTINGDATA"),
	})
	c.Assert(id, checker.Not(checker.Equals), "", check.Commentf("secrets: %s", id))

	secret := d.GetSecret(c, id)
	c.Assert(secret.ID, checker.Equals, id, check.Commentf("secret: %v", secret))

	d.DeleteSecret(c, secret.ID)
	status, out, err := d.SockRequest("GET", "/secrets/"+id, nil)
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusNotFound, check.Commentf("secret delete: %s", string(out)))

	// delete non-existing secret, daemon should return a status code of 404
	id = "non-existing"
	status, out, err = d.SockRequest("DELETE", "/secrets/"+id, nil)
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusNotFound, check.Commentf("secret delete: %s", string(out)))

}

func (s *DockerSwarmSuite) TestAPISwarmSecretsUpdate(c *check.C) {
	d := s.AddDaemon(c, true, true)

	testName := "test_secret"
	id := d.CreateSecret(c, swarm.SecretSpec{
		Annotations: swarm.Annotations{
			Name: testName,
			Labels: map[string]string{
				"test": "test1",
			},
		},
		Data: []byte("TESTINGDATA"),
	})
	c.Assert(id, checker.Not(checker.Equals), "", check.Commentf("secrets: %s", id))

	secret := d.GetSecret(c, id)
	c.Assert(secret.ID, checker.Equals, id, check.Commentf("secret: %v", secret))

	// test UpdateSecret with full ID
	d.UpdateSecret(c, id, func(s *swarm.Secret) {
		s.Spec.Labels = map[string]string{
			"test": "test1",
		}
	})

	secret = d.GetSecret(c, id)
	c.Assert(secret.Spec.Labels["test"], checker.Equals, "test1", check.Commentf("secret: %v", secret))

	// test UpdateSecret with full name
	d.UpdateSecret(c, secret.Spec.Name, func(s *swarm.Secret) {
		s.Spec.Labels = map[string]string{
			"test": "test2",
		}
	})

	secret = d.GetSecret(c, id)
	c.Assert(secret.Spec.Labels["test"], checker.Equals, "test2", check.Commentf("secret: %v", secret))

	// test UpdateSecret with prefix ID
	d.UpdateSecret(c, id[:1], func(s *swarm.Secret) {
		s.Spec.Labels = map[string]string{
			"test": "test3",
		}
	})

	secret = d.GetSecret(c, id)
	c.Assert(secret.Spec.Labels["test"], checker.Equals, "test3", check.Commentf("secret: %v", secret))

	// test UpdateSecret in updating Data which is not supported in daemon
	// this test will produce an error in func UpdateSecret
	secret = d.GetSecret(c, id)
	secret.Spec.Data = []byte("TESTINGDATA2")

	url := fmt.Sprintf("/secrets/%s/update?version=%d", secret.ID, secret.Version.Index)
	status, out, err := d.SockRequest("POST", url, secret.Spec)

	c.Assert(err, checker.IsNil, check.Commentf(string(out)))
	c.Assert(status, checker.Equals, http.StatusBadRequest, check.Commentf("output: %q", string(out)))
}
