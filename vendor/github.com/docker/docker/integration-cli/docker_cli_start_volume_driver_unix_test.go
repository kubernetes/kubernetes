// +build !windows

package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"

	"github.com/go-check/check"
)

func init() {
	check.Suite(&DockerExternalVolumeSuite{
		ds: &DockerSuite{},
	})
}

type eventCounter struct {
	activations int
	creations   int
	removals    int
	mounts      int
	unmounts    int
	paths       int
}

type DockerExternalVolumeSuite struct {
	server *httptest.Server
	ds     *DockerSuite
	d      *Daemon
	ec     *eventCounter
}

func (s *DockerExternalVolumeSuite) SetUpTest(c *check.C) {
	s.d = NewDaemon(c)
	s.ec = &eventCounter{}
}

func (s *DockerExternalVolumeSuite) TearDownTest(c *check.C) {
	s.d.Stop()
	s.ds.TearDownTest(c)
}

func (s *DockerExternalVolumeSuite) SetUpSuite(c *check.C) {
	mux := http.NewServeMux()
	s.server = httptest.NewServer(mux)

	type pluginRequest struct {
		name string
	}

	mux.HandleFunc("/Plugin.Activate", func(w http.ResponseWriter, r *http.Request) {
		s.ec.activations++

		w.Header().Set("Content-Type", "application/vnd.docker.plugins.v1+json")
		fmt.Fprintln(w, `{"Implements": ["VolumeDriver"]}`)
	})

	mux.HandleFunc("/VolumeDriver.Create", func(w http.ResponseWriter, r *http.Request) {
		s.ec.creations++

		w.Header().Set("Content-Type", "application/vnd.docker.plugins.v1+json")
		fmt.Fprintln(w, `{}`)
	})

	mux.HandleFunc("/VolumeDriver.Remove", func(w http.ResponseWriter, r *http.Request) {
		s.ec.removals++

		w.Header().Set("Content-Type", "application/vnd.docker.plugins.v1+json")
		fmt.Fprintln(w, `{}`)
	})

	mux.HandleFunc("/VolumeDriver.Path", func(w http.ResponseWriter, r *http.Request) {
		s.ec.paths++

		var pr pluginRequest
		if err := json.NewDecoder(r.Body).Decode(&pr); err != nil {
			http.Error(w, err.Error(), 500)
		}

		p := hostVolumePath(pr.name)

		w.Header().Set("Content-Type", "application/vnd.docker.plugins.v1+json")
		fmt.Fprintln(w, fmt.Sprintf("{\"Mountpoint\": \"%s\"}", p))
	})

	mux.HandleFunc("/VolumeDriver.Mount", func(w http.ResponseWriter, r *http.Request) {
		s.ec.mounts++

		var pr pluginRequest
		if err := json.NewDecoder(r.Body).Decode(&pr); err != nil {
			http.Error(w, err.Error(), 500)
		}

		p := hostVolumePath(pr.name)
		if err := os.MkdirAll(p, 0755); err != nil {
			http.Error(w, err.Error(), 500)
		}

		if err := ioutil.WriteFile(filepath.Join(p, "test"), []byte(s.server.URL), 0644); err != nil {
			http.Error(w, err.Error(), 500)
		}

		w.Header().Set("Content-Type", "application/vnd.docker.plugins.v1+json")
		fmt.Fprintln(w, fmt.Sprintf("{\"Mountpoint\": \"%s\"}", p))
	})

	mux.HandleFunc("/VolumeDriver.Unmount", func(w http.ResponseWriter, r *http.Request) {
		s.ec.unmounts++

		var pr pluginRequest
		if err := json.NewDecoder(r.Body).Decode(&pr); err != nil {
			http.Error(w, err.Error(), 500)
		}

		p := hostVolumePath(pr.name)
		if err := os.RemoveAll(p); err != nil {
			http.Error(w, err.Error(), 500)
		}

		w.Header().Set("Content-Type", "application/vnd.docker.plugins.v1+json")
		fmt.Fprintln(w, `{}`)
	})

	if err := os.MkdirAll("/etc/docker/plugins", 0755); err != nil {
		c.Fatal(err)
	}

	if err := ioutil.WriteFile("/etc/docker/plugins/test-external-volume-driver.spec", []byte(s.server.URL), 0644); err != nil {
		c.Fatal(err)
	}
}

func (s *DockerExternalVolumeSuite) TearDownSuite(c *check.C) {
	s.server.Close()

	if err := os.RemoveAll("/etc/docker/plugins"); err != nil {
		c.Fatal(err)
	}
}

func (s *DockerExternalVolumeSuite) TestStartExternalNamedVolumeDriver(c *check.C) {
	if err := s.d.StartWithBusybox(); err != nil {
		c.Fatal(err)
	}

	out, err := s.d.Cmd("run", "--rm", "--name", "test-data", "-v", "external-volume-test:/tmp/external-volume-test", "--volume-driver", "test-external-volume-driver", "busybox:latest", "cat", "/tmp/external-volume-test/test")
	if err != nil {
		c.Fatal(err)
	}

	if !strings.Contains(out, s.server.URL) {
		c.Fatalf("External volume mount failed. Output: %s\n", out)
	}

	p := hostVolumePath("external-volume-test")
	_, err = os.Lstat(p)
	if err == nil {
		c.Fatalf("Expected error checking volume path in host: %s\n", p)
	}

	if !os.IsNotExist(err) {
		c.Fatalf("Expected volume path in host to not exist: %s, %v\n", p, err)
	}

	c.Assert(s.ec.activations, check.Equals, 1)
	c.Assert(s.ec.creations, check.Equals, 1)
	c.Assert(s.ec.removals, check.Equals, 1)
	c.Assert(s.ec.mounts, check.Equals, 1)
	c.Assert(s.ec.unmounts, check.Equals, 1)
}

func (s *DockerExternalVolumeSuite) TestStartExternalVolumeUnnamedDriver(c *check.C) {
	if err := s.d.StartWithBusybox(); err != nil {
		c.Fatal(err)
	}

	out, err := s.d.Cmd("run", "--rm", "--name", "test-data", "-v", "/tmp/external-volume-test", "--volume-driver", "test-external-volume-driver", "busybox:latest", "cat", "/tmp/external-volume-test/test")
	if err != nil {
		c.Fatal(err)
	}

	if !strings.Contains(out, s.server.URL) {
		c.Fatalf("External volume mount failed. Output: %s\n", out)
	}

	c.Assert(s.ec.activations, check.Equals, 1)
	c.Assert(s.ec.creations, check.Equals, 1)
	c.Assert(s.ec.removals, check.Equals, 1)
	c.Assert(s.ec.mounts, check.Equals, 1)
	c.Assert(s.ec.unmounts, check.Equals, 1)
}

func (s DockerExternalVolumeSuite) TestStartExternalVolumeDriverVolumesFrom(c *check.C) {
	if err := s.d.StartWithBusybox(); err != nil {
		c.Fatal(err)
	}

	if _, err := s.d.Cmd("run", "-d", "--name", "vol-test1", "-v", "/foo", "--volume-driver", "test-external-volume-driver", "busybox:latest"); err != nil {
		c.Fatal(err)
	}

	if _, err := s.d.Cmd("run", "--rm", "--volumes-from", "vol-test1", "--name", "vol-test2", "busybox", "ls", "/tmp"); err != nil {
		c.Fatal(err)
	}

	if _, err := s.d.Cmd("rm", "-f", "vol-test1"); err != nil {
		c.Fatal(err)
	}

	c.Assert(s.ec.activations, check.Equals, 1)
	c.Assert(s.ec.creations, check.Equals, 2)
	c.Assert(s.ec.removals, check.Equals, 1)
	c.Assert(s.ec.mounts, check.Equals, 2)
	c.Assert(s.ec.unmounts, check.Equals, 2)
}

func (s DockerExternalVolumeSuite) TestStartExternalVolumeDriverDeleteContainer(c *check.C) {
	if err := s.d.StartWithBusybox(); err != nil {
		c.Fatal(err)
	}

	if _, err := s.d.Cmd("run", "-d", "--name", "vol-test1", "-v", "/foo", "--volume-driver", "test-external-volume-driver", "busybox:latest"); err != nil {
		c.Fatal(err)
	}

	if _, err := s.d.Cmd("rm", "-fv", "vol-test1"); err != nil {
		c.Fatal(err)
	}

	c.Assert(s.ec.activations, check.Equals, 1)
	c.Assert(s.ec.creations, check.Equals, 1)
	c.Assert(s.ec.removals, check.Equals, 1)
	c.Assert(s.ec.mounts, check.Equals, 1)
	c.Assert(s.ec.unmounts, check.Equals, 1)
}

func hostVolumePath(name string) string {
	return fmt.Sprintf("/var/lib/docker/volumes/%s", name)
}
