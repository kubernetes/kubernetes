package main

import (
	"fmt"
	"strings"

	"github.com/go-check/check"
)

func (s *DockerSuite) TestPause(c *check.C) {
	defer unpauseAllContainers()

	name := "testeventpause"
	out, _ := dockerCmd(c, "images", "-q")
	image := strings.Split(out, "\n")[0]
	dockerCmd(c, "run", "-d", "--name", name, image, "top")

	dockerCmd(c, "pause", name)
	pausedContainers, err := getSliceOfPausedContainers()
	if err != nil {
		c.Fatalf("error thrown while checking if containers were paused: %v", err)
	}
	if len(pausedContainers) != 1 {
		c.Fatalf("there should be one paused container and not %d", len(pausedContainers))
	}

	dockerCmd(c, "unpause", name)

	out, _ = dockerCmd(c, "events", "--since=0", fmt.Sprintf("--until=%d", daemonTime(c).Unix()))
	events := strings.Split(out, "\n")
	if len(events) <= 1 {
		c.Fatalf("Missing expected event")
	}

	pauseEvent := strings.Fields(events[len(events)-3])
	unpauseEvent := strings.Fields(events[len(events)-2])

	if pauseEvent[len(pauseEvent)-1] != "pause" {
		c.Fatalf("event should be pause, not %#v", pauseEvent)
	}
	if unpauseEvent[len(unpauseEvent)-1] != "unpause" {
		c.Fatalf("event should be unpause, not %#v", unpauseEvent)
	}

}

func (s *DockerSuite) TestPauseMultipleContainers(c *check.C) {
	defer unpauseAllContainers()

	containers := []string{
		"testpausewithmorecontainers1",
		"testpausewithmorecontainers2",
	}
	out, _ := dockerCmd(c, "images", "-q")
	image := strings.Split(out, "\n")[0]
	for _, name := range containers {
		dockerCmd(c, "run", "-d", "--name", name, image, "top")
	}
	dockerCmd(c, append([]string{"pause"}, containers...)...)
	pausedContainers, err := getSliceOfPausedContainers()
	if err != nil {
		c.Fatalf("error thrown while checking if containers were paused: %v", err)
	}
	if len(pausedContainers) != len(containers) {
		c.Fatalf("there should be %d paused container and not %d", len(containers), len(pausedContainers))
	}

	dockerCmd(c, append([]string{"unpause"}, containers...)...)

	out, _ = dockerCmd(c, "events", "--since=0", fmt.Sprintf("--until=%d", daemonTime(c).Unix()))
	events := strings.Split(out, "\n")
	if len(events) <= len(containers)*3-2 {
		c.Fatalf("Missing expected event")
	}

	pauseEvents := make([][]string, len(containers))
	unpauseEvents := make([][]string, len(containers))
	for i := range containers {
		pauseEvents[i] = strings.Fields(events[len(events)-len(containers)*2-1+i])
		unpauseEvents[i] = strings.Fields(events[len(events)-len(containers)-1+i])
	}

	for _, pauseEvent := range pauseEvents {
		if pauseEvent[len(pauseEvent)-1] != "pause" {
			c.Fatalf("event should be pause, not %#v", pauseEvent)
		}
	}
	for _, unpauseEvent := range unpauseEvents {
		if unpauseEvent[len(unpauseEvent)-1] != "unpause" {
			c.Fatalf("event should be unpause, not %#v", unpauseEvent)
		}
	}

}
