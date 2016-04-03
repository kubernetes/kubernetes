// +build experimental

package main

import (
	"github.com/docker/docker/api/types"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestInspectNamedMountPoint(c *check.C) {
	dockerCmd(c, "run", "-d", "--name", "test", "-v", "data:/data", "busybox", "cat")

	vol, err := inspectFieldJSON("test", "Mounts")
	c.Assert(err, check.IsNil)

	var mp []types.MountPoint
	err = unmarshalJSON([]byte(vol), &mp)
	c.Assert(err, check.IsNil)

	if len(mp) != 1 {
		c.Fatalf("Expected 1 mount point, was %v\n", len(mp))
	}

	m := mp[0]
	if m.Name != "data" {
		c.Fatalf("Expected name data, was %s\n", m.Name)
	}

	if m.Driver != "local" {
		c.Fatalf("Expected driver local, was %s\n", m.Driver)
	}

	if m.Source == "" {
		c.Fatalf("Expected source to not be empty")
	}

	if m.RW != true {
		c.Fatalf("Expected rw to be true")
	}

	if m.Destination != "/data" {
		c.Fatalf("Expected destination /data, was %s\n", m.Destination)
	}
}
