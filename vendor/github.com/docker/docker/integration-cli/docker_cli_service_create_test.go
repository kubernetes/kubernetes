// +build !windows

package main

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/mount"
	"github.com/docker/docker/api/types/swarm"
	"github.com/docker/docker/integration-cli/checker"
	"github.com/go-check/check"
)

func (s *DockerSwarmSuite) TestServiceCreateMountVolume(c *check.C) {
	d := s.AddDaemon(c, true, true)
	out, err := d.Cmd("service", "create", "--no-resolve-image", "--detach=true", "--mount", "type=volume,source=foo,target=/foo,volume-nocopy", "busybox", "top")
	c.Assert(err, checker.IsNil, check.Commentf(out))
	id := strings.TrimSpace(out)

	var tasks []swarm.Task
	waitAndAssert(c, defaultReconciliationTimeout, func(c *check.C) (interface{}, check.CommentInterface) {
		tasks = d.GetServiceTasks(c, id)
		return len(tasks) > 0, nil
	}, checker.Equals, true)

	task := tasks[0]
	waitAndAssert(c, defaultReconciliationTimeout, func(c *check.C) (interface{}, check.CommentInterface) {
		if task.NodeID == "" || task.Status.ContainerStatus.ContainerID == "" {
			task = d.GetTask(c, task.ID)
		}
		return task.NodeID != "" && task.Status.ContainerStatus.ContainerID != "", nil
	}, checker.Equals, true)

	// check container mount config
	out, err = s.nodeCmd(c, task.NodeID, "inspect", "--format", "{{json .HostConfig.Mounts}}", task.Status.ContainerStatus.ContainerID)
	c.Assert(err, checker.IsNil, check.Commentf(out))

	var mountConfig []mount.Mount
	c.Assert(json.Unmarshal([]byte(out), &mountConfig), checker.IsNil)
	c.Assert(mountConfig, checker.HasLen, 1)

	c.Assert(mountConfig[0].Source, checker.Equals, "foo")
	c.Assert(mountConfig[0].Target, checker.Equals, "/foo")
	c.Assert(mountConfig[0].Type, checker.Equals, mount.TypeVolume)
	c.Assert(mountConfig[0].VolumeOptions, checker.NotNil)
	c.Assert(mountConfig[0].VolumeOptions.NoCopy, checker.True)

	// check container mounts actual
	out, err = s.nodeCmd(c, task.NodeID, "inspect", "--format", "{{json .Mounts}}", task.Status.ContainerStatus.ContainerID)
	c.Assert(err, checker.IsNil, check.Commentf(out))

	var mounts []types.MountPoint
	c.Assert(json.Unmarshal([]byte(out), &mounts), checker.IsNil)
	c.Assert(mounts, checker.HasLen, 1)

	c.Assert(mounts[0].Type, checker.Equals, mount.TypeVolume)
	c.Assert(mounts[0].Name, checker.Equals, "foo")
	c.Assert(mounts[0].Destination, checker.Equals, "/foo")
	c.Assert(mounts[0].RW, checker.Equals, true)
}

func (s *DockerSwarmSuite) TestServiceCreateWithSecretSimple(c *check.C) {
	d := s.AddDaemon(c, true, true)

	serviceName := "test-service-secret"
	testName := "test_secret"
	id := d.CreateSecret(c, swarm.SecretSpec{
		Annotations: swarm.Annotations{
			Name: testName,
		},
		Data: []byte("TESTINGDATA"),
	})
	c.Assert(id, checker.Not(checker.Equals), "", check.Commentf("secrets: %s", id))

	out, err := d.Cmd("service", "create", "--no-resolve-image", "--name", serviceName, "--secret", testName, "busybox", "top")
	c.Assert(err, checker.IsNil, check.Commentf(out))

	out, err = d.Cmd("service", "inspect", "--format", "{{ json .Spec.TaskTemplate.ContainerSpec.Secrets }}", serviceName)
	c.Assert(err, checker.IsNil)

	var refs []swarm.SecretReference
	c.Assert(json.Unmarshal([]byte(out), &refs), checker.IsNil)
	c.Assert(refs, checker.HasLen, 1)

	c.Assert(refs[0].SecretName, checker.Equals, testName)
	c.Assert(refs[0].File, checker.Not(checker.IsNil))
	c.Assert(refs[0].File.Name, checker.Equals, testName)
	c.Assert(refs[0].File.UID, checker.Equals, "0")
	c.Assert(refs[0].File.GID, checker.Equals, "0")

	out, err = d.Cmd("service", "rm", serviceName)
	c.Assert(err, checker.IsNil, check.Commentf(out))
	d.DeleteSecret(c, testName)
}

func (s *DockerSwarmSuite) TestServiceCreateWithSecretSourceTargetPaths(c *check.C) {
	d := s.AddDaemon(c, true, true)

	testPaths := map[string]string{
		"app":                  "/etc/secret",
		"test_secret":          "test_secret",
		"relative_secret":      "relative/secret",
		"escapes_in_container": "../secret",
	}

	var secretFlags []string

	for testName, testTarget := range testPaths {
		id := d.CreateSecret(c, swarm.SecretSpec{
			Annotations: swarm.Annotations{
				Name: testName,
			},
			Data: []byte("TESTINGDATA " + testName + " " + testTarget),
		})
		c.Assert(id, checker.Not(checker.Equals), "", check.Commentf("secrets: %s", id))

		secretFlags = append(secretFlags, "--secret", fmt.Sprintf("source=%s,target=%s", testName, testTarget))
	}

	serviceName := "svc"
	serviceCmd := []string{"service", "create", "--no-resolve-image", "--name", serviceName}
	serviceCmd = append(serviceCmd, secretFlags...)
	serviceCmd = append(serviceCmd, "busybox", "top")
	out, err := d.Cmd(serviceCmd...)
	c.Assert(err, checker.IsNil, check.Commentf(out))

	out, err = d.Cmd("service", "inspect", "--format", "{{ json .Spec.TaskTemplate.ContainerSpec.Secrets }}", serviceName)
	c.Assert(err, checker.IsNil)

	var refs []swarm.SecretReference
	c.Assert(json.Unmarshal([]byte(out), &refs), checker.IsNil)
	c.Assert(refs, checker.HasLen, len(testPaths))

	var tasks []swarm.Task
	waitAndAssert(c, defaultReconciliationTimeout, func(c *check.C) (interface{}, check.CommentInterface) {
		tasks = d.GetServiceTasks(c, serviceName)
		return len(tasks) > 0, nil
	}, checker.Equals, true)

	task := tasks[0]
	waitAndAssert(c, defaultReconciliationTimeout, func(c *check.C) (interface{}, check.CommentInterface) {
		if task.NodeID == "" || task.Status.ContainerStatus.ContainerID == "" {
			task = d.GetTask(c, task.ID)
		}
		return task.NodeID != "" && task.Status.ContainerStatus.ContainerID != "", nil
	}, checker.Equals, true)

	for testName, testTarget := range testPaths {
		path := testTarget
		if !filepath.IsAbs(path) {
			path = filepath.Join("/run/secrets", path)
		}
		out, err := d.Cmd("exec", task.Status.ContainerStatus.ContainerID, "cat", path)
		c.Assert(err, checker.IsNil)
		c.Assert(out, checker.Equals, "TESTINGDATA "+testName+" "+testTarget)
	}

	out, err = d.Cmd("service", "rm", serviceName)
	c.Assert(err, checker.IsNil, check.Commentf(out))
}

func (s *DockerSwarmSuite) TestServiceCreateWithSecretReferencedTwice(c *check.C) {
	d := s.AddDaemon(c, true, true)

	id := d.CreateSecret(c, swarm.SecretSpec{
		Annotations: swarm.Annotations{
			Name: "mysecret",
		},
		Data: []byte("TESTINGDATA"),
	})
	c.Assert(id, checker.Not(checker.Equals), "", check.Commentf("secrets: %s", id))

	serviceName := "svc"
	out, err := d.Cmd("service", "create", "--no-resolve-image", "--name", serviceName, "--secret", "source=mysecret,target=target1", "--secret", "source=mysecret,target=target2", "busybox", "top")
	c.Assert(err, checker.IsNil, check.Commentf(out))

	out, err = d.Cmd("service", "inspect", "--format", "{{ json .Spec.TaskTemplate.ContainerSpec.Secrets }}", serviceName)
	c.Assert(err, checker.IsNil)

	var refs []swarm.SecretReference
	c.Assert(json.Unmarshal([]byte(out), &refs), checker.IsNil)
	c.Assert(refs, checker.HasLen, 2)

	var tasks []swarm.Task
	waitAndAssert(c, defaultReconciliationTimeout, func(c *check.C) (interface{}, check.CommentInterface) {
		tasks = d.GetServiceTasks(c, serviceName)
		return len(tasks) > 0, nil
	}, checker.Equals, true)

	task := tasks[0]
	waitAndAssert(c, defaultReconciliationTimeout, func(c *check.C) (interface{}, check.CommentInterface) {
		if task.NodeID == "" || task.Status.ContainerStatus.ContainerID == "" {
			task = d.GetTask(c, task.ID)
		}
		return task.NodeID != "" && task.Status.ContainerStatus.ContainerID != "", nil
	}, checker.Equals, true)

	for _, target := range []string{"target1", "target2"} {
		c.Assert(err, checker.IsNil, check.Commentf(out))
		path := filepath.Join("/run/secrets", target)
		out, err := d.Cmd("exec", task.Status.ContainerStatus.ContainerID, "cat", path)
		c.Assert(err, checker.IsNil)
		c.Assert(out, checker.Equals, "TESTINGDATA")
	}

	out, err = d.Cmd("service", "rm", serviceName)
	c.Assert(err, checker.IsNil, check.Commentf(out))
}

func (s *DockerSwarmSuite) TestServiceCreateWithConfigSimple(c *check.C) {
	d := s.AddDaemon(c, true, true)

	serviceName := "test-service-config"
	testName := "test_config"
	id := d.CreateConfig(c, swarm.ConfigSpec{
		Annotations: swarm.Annotations{
			Name: testName,
		},
		Data: []byte("TESTINGDATA"),
	})
	c.Assert(id, checker.Not(checker.Equals), "", check.Commentf("configs: %s", id))

	out, err := d.Cmd("service", "create", "--no-resolve-image", "--name", serviceName, "--config", testName, "busybox", "top")
	c.Assert(err, checker.IsNil, check.Commentf(out))

	out, err = d.Cmd("service", "inspect", "--format", "{{ json .Spec.TaskTemplate.ContainerSpec.Configs }}", serviceName)
	c.Assert(err, checker.IsNil)

	var refs []swarm.ConfigReference
	c.Assert(json.Unmarshal([]byte(out), &refs), checker.IsNil)
	c.Assert(refs, checker.HasLen, 1)

	c.Assert(refs[0].ConfigName, checker.Equals, testName)
	c.Assert(refs[0].File, checker.Not(checker.IsNil))
	c.Assert(refs[0].File.Name, checker.Equals, testName)
	c.Assert(refs[0].File.UID, checker.Equals, "0")
	c.Assert(refs[0].File.GID, checker.Equals, "0")

	out, err = d.Cmd("service", "rm", serviceName)
	c.Assert(err, checker.IsNil, check.Commentf(out))
	d.DeleteConfig(c, testName)
}

func (s *DockerSwarmSuite) TestServiceCreateWithConfigSourceTargetPaths(c *check.C) {
	d := s.AddDaemon(c, true, true)

	testPaths := map[string]string{
		"app":             "/etc/config",
		"test_config":     "test_config",
		"relative_config": "relative/config",
	}

	var configFlags []string

	for testName, testTarget := range testPaths {
		id := d.CreateConfig(c, swarm.ConfigSpec{
			Annotations: swarm.Annotations{
				Name: testName,
			},
			Data: []byte("TESTINGDATA " + testName + " " + testTarget),
		})
		c.Assert(id, checker.Not(checker.Equals), "", check.Commentf("configs: %s", id))

		configFlags = append(configFlags, "--config", fmt.Sprintf("source=%s,target=%s", testName, testTarget))
	}

	serviceName := "svc"
	serviceCmd := []string{"service", "create", "--no-resolve-image", "--name", serviceName}
	serviceCmd = append(serviceCmd, configFlags...)
	serviceCmd = append(serviceCmd, "busybox", "top")
	out, err := d.Cmd(serviceCmd...)
	c.Assert(err, checker.IsNil, check.Commentf(out))

	out, err = d.Cmd("service", "inspect", "--format", "{{ json .Spec.TaskTemplate.ContainerSpec.Configs }}", serviceName)
	c.Assert(err, checker.IsNil)

	var refs []swarm.ConfigReference
	c.Assert(json.Unmarshal([]byte(out), &refs), checker.IsNil)
	c.Assert(refs, checker.HasLen, len(testPaths))

	var tasks []swarm.Task
	waitAndAssert(c, defaultReconciliationTimeout, func(c *check.C) (interface{}, check.CommentInterface) {
		tasks = d.GetServiceTasks(c, serviceName)
		return len(tasks) > 0, nil
	}, checker.Equals, true)

	task := tasks[0]
	waitAndAssert(c, defaultReconciliationTimeout, func(c *check.C) (interface{}, check.CommentInterface) {
		if task.NodeID == "" || task.Status.ContainerStatus.ContainerID == "" {
			task = d.GetTask(c, task.ID)
		}
		return task.NodeID != "" && task.Status.ContainerStatus.ContainerID != "", nil
	}, checker.Equals, true)

	for testName, testTarget := range testPaths {
		path := testTarget
		if !filepath.IsAbs(path) {
			path = filepath.Join("/", path)
		}
		out, err := d.Cmd("exec", task.Status.ContainerStatus.ContainerID, "cat", path)
		c.Assert(err, checker.IsNil)
		c.Assert(out, checker.Equals, "TESTINGDATA "+testName+" "+testTarget)
	}

	out, err = d.Cmd("service", "rm", serviceName)
	c.Assert(err, checker.IsNil, check.Commentf(out))
}

func (s *DockerSwarmSuite) TestServiceCreateWithConfigReferencedTwice(c *check.C) {
	d := s.AddDaemon(c, true, true)

	id := d.CreateConfig(c, swarm.ConfigSpec{
		Annotations: swarm.Annotations{
			Name: "myconfig",
		},
		Data: []byte("TESTINGDATA"),
	})
	c.Assert(id, checker.Not(checker.Equals), "", check.Commentf("configs: %s", id))

	serviceName := "svc"
	out, err := d.Cmd("service", "create", "--no-resolve-image", "--name", serviceName, "--config", "source=myconfig,target=target1", "--config", "source=myconfig,target=target2", "busybox", "top")
	c.Assert(err, checker.IsNil, check.Commentf(out))

	out, err = d.Cmd("service", "inspect", "--format", "{{ json .Spec.TaskTemplate.ContainerSpec.Configs }}", serviceName)
	c.Assert(err, checker.IsNil)

	var refs []swarm.ConfigReference
	c.Assert(json.Unmarshal([]byte(out), &refs), checker.IsNil)
	c.Assert(refs, checker.HasLen, 2)

	var tasks []swarm.Task
	waitAndAssert(c, defaultReconciliationTimeout, func(c *check.C) (interface{}, check.CommentInterface) {
		tasks = d.GetServiceTasks(c, serviceName)
		return len(tasks) > 0, nil
	}, checker.Equals, true)

	task := tasks[0]
	waitAndAssert(c, defaultReconciliationTimeout, func(c *check.C) (interface{}, check.CommentInterface) {
		if task.NodeID == "" || task.Status.ContainerStatus.ContainerID == "" {
			task = d.GetTask(c, task.ID)
		}
		return task.NodeID != "" && task.Status.ContainerStatus.ContainerID != "", nil
	}, checker.Equals, true)

	for _, target := range []string{"target1", "target2"} {
		c.Assert(err, checker.IsNil, check.Commentf(out))
		path := filepath.Join("/", target)
		out, err := d.Cmd("exec", task.Status.ContainerStatus.ContainerID, "cat", path)
		c.Assert(err, checker.IsNil)
		c.Assert(out, checker.Equals, "TESTINGDATA")
	}

	out, err = d.Cmd("service", "rm", serviceName)
	c.Assert(err, checker.IsNil, check.Commentf(out))
}

func (s *DockerSwarmSuite) TestServiceCreateMountTmpfs(c *check.C) {
	d := s.AddDaemon(c, true, true)
	out, err := d.Cmd("service", "create", "--no-resolve-image", "--detach=true", "--mount", "type=tmpfs,target=/foo,tmpfs-size=1MB", "busybox", "sh", "-c", "mount | grep foo; tail -f /dev/null")
	c.Assert(err, checker.IsNil, check.Commentf(out))
	id := strings.TrimSpace(out)

	var tasks []swarm.Task
	waitAndAssert(c, defaultReconciliationTimeout, func(c *check.C) (interface{}, check.CommentInterface) {
		tasks = d.GetServiceTasks(c, id)
		return len(tasks) > 0, nil
	}, checker.Equals, true)

	task := tasks[0]
	waitAndAssert(c, defaultReconciliationTimeout, func(c *check.C) (interface{}, check.CommentInterface) {
		if task.NodeID == "" || task.Status.ContainerStatus.ContainerID == "" {
			task = d.GetTask(c, task.ID)
		}
		return task.NodeID != "" && task.Status.ContainerStatus.ContainerID != "", nil
	}, checker.Equals, true)

	// check container mount config
	out, err = s.nodeCmd(c, task.NodeID, "inspect", "--format", "{{json .HostConfig.Mounts}}", task.Status.ContainerStatus.ContainerID)
	c.Assert(err, checker.IsNil, check.Commentf(out))

	var mountConfig []mount.Mount
	c.Assert(json.Unmarshal([]byte(out), &mountConfig), checker.IsNil)
	c.Assert(mountConfig, checker.HasLen, 1)

	c.Assert(mountConfig[0].Source, checker.Equals, "")
	c.Assert(mountConfig[0].Target, checker.Equals, "/foo")
	c.Assert(mountConfig[0].Type, checker.Equals, mount.TypeTmpfs)
	c.Assert(mountConfig[0].TmpfsOptions, checker.NotNil)
	c.Assert(mountConfig[0].TmpfsOptions.SizeBytes, checker.Equals, int64(1048576))

	// check container mounts actual
	out, err = s.nodeCmd(c, task.NodeID, "inspect", "--format", "{{json .Mounts}}", task.Status.ContainerStatus.ContainerID)
	c.Assert(err, checker.IsNil, check.Commentf(out))

	var mounts []types.MountPoint
	c.Assert(json.Unmarshal([]byte(out), &mounts), checker.IsNil)
	c.Assert(mounts, checker.HasLen, 1)

	c.Assert(mounts[0].Type, checker.Equals, mount.TypeTmpfs)
	c.Assert(mounts[0].Name, checker.Equals, "")
	c.Assert(mounts[0].Destination, checker.Equals, "/foo")
	c.Assert(mounts[0].RW, checker.Equals, true)

	out, err = s.nodeCmd(c, task.NodeID, "logs", task.Status.ContainerStatus.ContainerID)
	c.Assert(err, checker.IsNil, check.Commentf(out))
	c.Assert(strings.TrimSpace(out), checker.HasPrefix, "tmpfs on /foo type tmpfs")
	c.Assert(strings.TrimSpace(out), checker.Contains, "size=1024k")
}

func (s *DockerSwarmSuite) TestServiceCreateWithNetworkAlias(c *check.C) {
	d := s.AddDaemon(c, true, true)
	out, err := d.Cmd("network", "create", "--scope=swarm", "test_swarm_br")
	c.Assert(err, checker.IsNil, check.Commentf(out))

	out, err = d.Cmd("service", "create", "--no-resolve-image", "--detach=true", "--network=name=test_swarm_br,alias=srv_alias", "--name=alias_tst_container", "busybox", "top")
	c.Assert(err, checker.IsNil, check.Commentf(out))
	id := strings.TrimSpace(out)

	var tasks []swarm.Task
	waitAndAssert(c, defaultReconciliationTimeout, func(c *check.C) (interface{}, check.CommentInterface) {
		tasks = d.GetServiceTasks(c, id)
		return len(tasks) > 0, nil
	}, checker.Equals, true)

	task := tasks[0]
	waitAndAssert(c, defaultReconciliationTimeout, func(c *check.C) (interface{}, check.CommentInterface) {
		if task.NodeID == "" || task.Status.ContainerStatus.ContainerID == "" {
			task = d.GetTask(c, task.ID)
		}
		return task.NodeID != "" && task.Status.ContainerStatus.ContainerID != "", nil
	}, checker.Equals, true)

	// check container alias config
	out, err = s.nodeCmd(c, task.NodeID, "inspect", "--format", "{{json .NetworkSettings.Networks.test_swarm_br.Aliases}}", task.Status.ContainerStatus.ContainerID)
	c.Assert(err, checker.IsNil, check.Commentf(out))

	// Make sure the only alias seen is the container-id
	var aliases []string
	c.Assert(json.Unmarshal([]byte(out), &aliases), checker.IsNil)
	c.Assert(aliases, checker.HasLen, 1)

	c.Assert(task.Status.ContainerStatus.ContainerID, checker.Contains, aliases[0])
}
