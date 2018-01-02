// +build !windows

package main

import (
	"strconv"
	"strings"

	"github.com/docker/docker/api/types/swarm"
	"github.com/docker/docker/daemon/cluster/executor/container"
	"github.com/docker/docker/integration-cli/checker"
	"github.com/go-check/check"
)

// start a service, and then make its task unhealthy during running
// finally, unhealthy task should be detected and killed
func (s *DockerSwarmSuite) TestServiceHealthRun(c *check.C) {
	testRequires(c, DaemonIsLinux) // busybox doesn't work on Windows

	d := s.AddDaemon(c, true, true)

	// build image with health-check
	// note: use `daemon.buildImageWithOut` to build, do not use `buildImage` to build
	imageName := "testhealth"
	_, _, err := d.BuildImageWithOut(imageName,
		`FROM busybox
		RUN touch /status
		HEALTHCHECK --interval=1s --timeout=1s --retries=1\
		  CMD cat /status`,
		true)
	c.Check(err, check.IsNil)

	serviceName := "healthServiceRun"
	out, err := d.Cmd("service", "create", "--no-resolve-image", "--detach=true", "--name", serviceName, imageName, "top")
	c.Assert(err, checker.IsNil, check.Commentf(out))
	id := strings.TrimSpace(out)

	var tasks []swarm.Task
	waitAndAssert(c, defaultReconciliationTimeout, func(c *check.C) (interface{}, check.CommentInterface) {
		tasks = d.GetServiceTasks(c, id)
		return tasks, nil
	}, checker.HasLen, 1)

	task := tasks[0]

	// wait for task to start
	waitAndAssert(c, defaultReconciliationTimeout, func(c *check.C) (interface{}, check.CommentInterface) {
		task = d.GetTask(c, task.ID)
		return task.Status.State, nil
	}, checker.Equals, swarm.TaskStateRunning)
	containerID := task.Status.ContainerStatus.ContainerID

	// wait for container to be healthy
	waitAndAssert(c, defaultReconciliationTimeout, func(c *check.C) (interface{}, check.CommentInterface) {
		out, _ := d.Cmd("inspect", "--format={{.State.Health.Status}}", containerID)
		return strings.TrimSpace(out), nil
	}, checker.Equals, "healthy")

	// make it fail
	d.Cmd("exec", containerID, "rm", "/status")
	// wait for container to be unhealthy
	waitAndAssert(c, defaultReconciliationTimeout, func(c *check.C) (interface{}, check.CommentInterface) {
		out, _ := d.Cmd("inspect", "--format={{.State.Health.Status}}", containerID)
		return strings.TrimSpace(out), nil
	}, checker.Equals, "unhealthy")

	// Task should be terminated
	waitAndAssert(c, defaultReconciliationTimeout, func(c *check.C) (interface{}, check.CommentInterface) {
		task = d.GetTask(c, task.ID)
		return task.Status.State, nil
	}, checker.Equals, swarm.TaskStateFailed)

	if !strings.Contains(task.Status.Err, container.ErrContainerUnhealthy.Error()) {
		c.Fatal("unhealthy task exits because of other error")
	}
}

// start a service whose task is unhealthy at beginning
// its tasks should be blocked in starting stage, until health check is passed
func (s *DockerSwarmSuite) TestServiceHealthStart(c *check.C) {
	testRequires(c, DaemonIsLinux) // busybox doesn't work on Windows

	d := s.AddDaemon(c, true, true)

	// service started from this image won't pass health check
	imageName := "testhealth"
	_, _, err := d.BuildImageWithOut(imageName,
		`FROM busybox
		HEALTHCHECK --interval=1s --timeout=1s --retries=1024\
		  CMD cat /status`,
		true)
	c.Check(err, check.IsNil)

	serviceName := "healthServiceStart"
	out, err := d.Cmd("service", "create", "--no-resolve-image", "--detach=true", "--name", serviceName, imageName, "top")
	c.Assert(err, checker.IsNil, check.Commentf(out))
	id := strings.TrimSpace(out)

	var tasks []swarm.Task
	waitAndAssert(c, defaultReconciliationTimeout, func(c *check.C) (interface{}, check.CommentInterface) {
		tasks = d.GetServiceTasks(c, id)
		return tasks, nil
	}, checker.HasLen, 1)

	task := tasks[0]

	// wait for task to start
	waitAndAssert(c, defaultReconciliationTimeout, func(c *check.C) (interface{}, check.CommentInterface) {
		task = d.GetTask(c, task.ID)
		return task.Status.State, nil
	}, checker.Equals, swarm.TaskStateStarting)

	containerID := task.Status.ContainerStatus.ContainerID

	// wait for health check to work
	waitAndAssert(c, defaultReconciliationTimeout, func(c *check.C) (interface{}, check.CommentInterface) {
		out, _ := d.Cmd("inspect", "--format={{.State.Health.FailingStreak}}", containerID)
		failingStreak, _ := strconv.Atoi(strings.TrimSpace(out))
		return failingStreak, nil
	}, checker.GreaterThan, 0)

	// task should be blocked at starting status
	task = d.GetTask(c, task.ID)
	c.Assert(task.Status.State, check.Equals, swarm.TaskStateStarting)

	// make it healthy
	d.Cmd("exec", containerID, "touch", "/status")

	// Task should be at running status
	waitAndAssert(c, defaultReconciliationTimeout, func(c *check.C) (interface{}, check.CommentInterface) {
		task = d.GetTask(c, task.ID)
		return task.Status.State, nil
	}, checker.Equals, swarm.TaskStateRunning)
}
