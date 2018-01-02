package main

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"os"

	"github.com/docker/docker/integration-cli/registry"
	"github.com/go-check/check"
)

func makefile(path string, contents string) (string, error) {
	f, err := ioutil.TempFile(path, "tmp")
	if err != nil {
		return "", err
	}
	err = ioutil.WriteFile(f.Name(), []byte(contents), os.ModePerm)
	if err != nil {
		return "", err
	}
	return f.Name(), nil
}

// TestV2Only ensures that a daemon by default does not
// attempt to contact any v1 registry endpoints.
func (s *DockerRegistrySuite) TestV2Only(c *check.C) {
	reg, err := registry.NewMock(c)
	defer reg.Close()
	c.Assert(err, check.IsNil)

	reg.RegisterHandler("/v2/", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(404)
	})

	reg.RegisterHandler("/v1/.*", func(w http.ResponseWriter, r *http.Request) {
		c.Fatal("V1 registry contacted")
	})

	repoName := fmt.Sprintf("%s/busybox", reg.URL())

	s.d.Start(c, "--insecure-registry", reg.URL())

	tmp, err := ioutil.TempDir("", "integration-cli-")
	c.Assert(err, check.IsNil)
	defer os.RemoveAll(tmp)

	dockerfileName, err := makefile(tmp, fmt.Sprintf("FROM %s/busybox", reg.URL()))
	c.Assert(err, check.IsNil, check.Commentf("Unable to create test dockerfile"))

	s.d.Cmd("build", "--file", dockerfileName, tmp)

	s.d.Cmd("run", repoName)
	s.d.Cmd("login", "-u", "richard", "-p", "testtest", reg.URL())
	s.d.Cmd("tag", "busybox", repoName)
	s.d.Cmd("push", repoName)
	s.d.Cmd("pull", repoName)
}

// TestV1 starts a daemon with legacy registries enabled
// and ensure v1 endpoints are hit for the following operations:
// login, push, pull, build & run
func (s *DockerRegistrySuite) TestV1(c *check.C) {
	reg, err := registry.NewMock(c)
	defer reg.Close()
	c.Assert(err, check.IsNil)

	v2Pings := 0
	reg.RegisterHandler("/v2/", func(w http.ResponseWriter, r *http.Request) {
		v2Pings++
		// V2 ping 404 causes fallback to v1
		w.WriteHeader(404)
	})

	v1Pings := 0
	reg.RegisterHandler("/v1/_ping", func(w http.ResponseWriter, r *http.Request) {
		v1Pings++
	})

	v1Logins := 0
	reg.RegisterHandler("/v1/users/", func(w http.ResponseWriter, r *http.Request) {
		v1Logins++
	})

	v1Repo := 0
	reg.RegisterHandler("/v1/repositories/busybox/", func(w http.ResponseWriter, r *http.Request) {
		v1Repo++
	})

	reg.RegisterHandler("/v1/repositories/busybox/images", func(w http.ResponseWriter, r *http.Request) {
		v1Repo++
	})

	s.d.Start(c, "--insecure-registry", reg.URL(), "--disable-legacy-registry=false")

	tmp, err := ioutil.TempDir("", "integration-cli-")
	c.Assert(err, check.IsNil)
	defer os.RemoveAll(tmp)

	dockerfileName, err := makefile(tmp, fmt.Sprintf("FROM %s/busybox", reg.URL()))
	c.Assert(err, check.IsNil, check.Commentf("Unable to create test dockerfile"))

	s.d.Cmd("build", "--file", dockerfileName, tmp)
	c.Assert(v1Repo, check.Equals, 1, check.Commentf("Expected v1 repository access after build"))

	repoName := fmt.Sprintf("%s/busybox", reg.URL())
	s.d.Cmd("run", repoName)
	c.Assert(v1Repo, check.Equals, 2, check.Commentf("Expected v1 repository access after run"))

	s.d.Cmd("login", "-u", "richard", "-p", "testtest", reg.URL())
	c.Assert(v1Logins, check.Equals, 1, check.Commentf("Expected v1 login attempt"))

	s.d.Cmd("tag", "busybox", repoName)
	s.d.Cmd("push", repoName)

	c.Assert(v1Repo, check.Equals, 2)

	s.d.Cmd("pull", repoName)
	c.Assert(v1Repo, check.Equals, 3, check.Commentf("Expected v1 repository access after pull"))
}
