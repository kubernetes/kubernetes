package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"time"

	eventtypes "github.com/docker/docker/api/types/events"
	eventstestutils "github.com/docker/docker/daemon/events/testutils"
	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/cli"
	"github.com/docker/docker/integration-cli/cli/build"
	"github.com/docker/docker/integration-cli/request"
	"github.com/docker/docker/pkg/testutil"
	icmd "github.com/docker/docker/pkg/testutil/cmd"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestEventsTimestampFormats(c *check.C) {
	name := "events-time-format-test"

	// Start stopwatch, generate an event
	start := daemonTime(c)
	time.Sleep(1100 * time.Millisecond) // so that first event occur in different second from since (just for the case)
	dockerCmd(c, "run", "--rm", "--name", name, "busybox", "true")
	time.Sleep(1100 * time.Millisecond) // so that until > since
	end := daemonTime(c)

	// List of available time formats to --since
	unixTs := func(t time.Time) string { return fmt.Sprintf("%v", t.Unix()) }
	rfc3339 := func(t time.Time) string { return t.Format(time.RFC3339) }
	duration := func(t time.Time) string { return time.Now().Sub(t).String() }

	// --since=$start must contain only the 'untag' event
	for _, f := range []func(time.Time) string{unixTs, rfc3339, duration} {
		since, until := f(start), f(end)
		out, _ := dockerCmd(c, "events", "--since="+since, "--until="+until)
		events := strings.Split(out, "\n")
		events = events[:len(events)-1]

		nEvents := len(events)
		c.Assert(nEvents, checker.GreaterOrEqualThan, 5) //Missing expected event
		containerEvents := eventActionsByIDAndType(c, events, name, "container")
		c.Assert(containerEvents, checker.HasLen, 5, check.Commentf("events: %v", events))

		c.Assert(containerEvents[0], checker.Equals, "create", check.Commentf(out))
		c.Assert(containerEvents[1], checker.Equals, "attach", check.Commentf(out))
		c.Assert(containerEvents[2], checker.Equals, "start", check.Commentf(out))
		c.Assert(containerEvents[3], checker.Equals, "die", check.Commentf(out))
		c.Assert(containerEvents[4], checker.Equals, "destroy", check.Commentf(out))
	}
}

func (s *DockerSuite) TestEventsUntag(c *check.C) {
	image := "busybox"
	dockerCmd(c, "tag", image, "utest:tag1")
	dockerCmd(c, "tag", image, "utest:tag2")
	dockerCmd(c, "rmi", "utest:tag1")
	dockerCmd(c, "rmi", "utest:tag2")

	result := icmd.RunCmd(icmd.Cmd{
		Command: []string{dockerBinary, "events", "--since=1"},
		Timeout: time.Millisecond * 2500,
	})
	c.Assert(result, icmd.Matches, icmd.Expected{Timeout: true})

	events := strings.Split(result.Stdout(), "\n")
	nEvents := len(events)
	// The last element after the split above will be an empty string, so we
	// get the two elements before the last, which are the untags we're
	// looking for.
	for _, v := range events[nEvents-3 : nEvents-1] {
		c.Assert(v, checker.Contains, "untag", check.Commentf("event should be untag"))
	}
}

func (s *DockerSuite) TestEventsLimit(c *check.C) {
	// Windows: Limit to 4 goroutines creating containers in order to prevent
	// timeouts creating so many containers simultaneously. This is a due to
	// a bug in the Windows platform. It will be fixed in a Windows Update.
	numContainers := 17
	numConcurrentContainers := numContainers
	if testEnv.DaemonPlatform() == "windows" {
		numConcurrentContainers = 4
	}
	sem := make(chan bool, numConcurrentContainers)
	errChan := make(chan error, numContainers)

	args := []string{"run", "--rm", "busybox", "true"}
	for i := 0; i < numContainers; i++ {
		sem <- true
		go func() {
			defer func() { <-sem }()
			out, err := exec.Command(dockerBinary, args...).CombinedOutput()
			if err != nil {
				err = fmt.Errorf("%v: %s", err, string(out))
			}
			errChan <- err
		}()
	}

	// Wait for all goroutines to finish
	for i := 0; i < cap(sem); i++ {
		sem <- true
	}
	close(errChan)

	for err := range errChan {
		c.Assert(err, checker.IsNil, check.Commentf("%q failed with error", strings.Join(args, " ")))
	}

	out, _ := dockerCmd(c, "events", "--since=0", "--until", daemonUnixTime(c))
	events := strings.Split(out, "\n")
	nEvents := len(events) - 1
	c.Assert(nEvents, checker.Equals, 256, check.Commentf("events should be limited to 256, but received %d", nEvents))
}

func (s *DockerSuite) TestEventsContainerEvents(c *check.C) {
	dockerCmd(c, "run", "--rm", "--name", "container-events-test", "busybox", "true")

	out, _ := dockerCmd(c, "events", "--until", daemonUnixTime(c))
	events := strings.Split(out, "\n")
	events = events[:len(events)-1]

	nEvents := len(events)
	c.Assert(nEvents, checker.GreaterOrEqualThan, 5) //Missing expected event
	containerEvents := eventActionsByIDAndType(c, events, "container-events-test", "container")
	c.Assert(containerEvents, checker.HasLen, 5, check.Commentf("events: %v", events))

	c.Assert(containerEvents[0], checker.Equals, "create", check.Commentf(out))
	c.Assert(containerEvents[1], checker.Equals, "attach", check.Commentf(out))
	c.Assert(containerEvents[2], checker.Equals, "start", check.Commentf(out))
	c.Assert(containerEvents[3], checker.Equals, "die", check.Commentf(out))
	c.Assert(containerEvents[4], checker.Equals, "destroy", check.Commentf(out))
}

func (s *DockerSuite) TestEventsContainerEventsAttrSort(c *check.C) {
	since := daemonUnixTime(c)
	dockerCmd(c, "run", "--rm", "--name", "container-events-test", "busybox", "true")

	out, _ := dockerCmd(c, "events", "--filter", "container=container-events-test", "--since", since, "--until", daemonUnixTime(c))
	events := strings.Split(out, "\n")

	nEvents := len(events)
	c.Assert(nEvents, checker.GreaterOrEqualThan, 3) //Missing expected event
	matchedEvents := 0
	for _, event := range events {
		matches := eventstestutils.ScanMap(event)
		if matches["eventType"] == "container" && matches["action"] == "create" {
			matchedEvents++
			c.Assert(out, checker.Contains, "(image=busybox, name=container-events-test)", check.Commentf("Event attributes not sorted"))
		} else if matches["eventType"] == "container" && matches["action"] == "start" {
			matchedEvents++
			c.Assert(out, checker.Contains, "(image=busybox, name=container-events-test)", check.Commentf("Event attributes not sorted"))
		}
	}
	c.Assert(matchedEvents, checker.Equals, 2, check.Commentf("missing events for container container-events-test:\n%s", out))
}

func (s *DockerSuite) TestEventsContainerEventsSinceUnixEpoch(c *check.C) {
	dockerCmd(c, "run", "--rm", "--name", "since-epoch-test", "busybox", "true")
	timeBeginning := time.Unix(0, 0).Format(time.RFC3339Nano)
	timeBeginning = strings.Replace(timeBeginning, "Z", ".000000000Z", -1)
	out, _ := dockerCmd(c, "events", "--since", timeBeginning, "--until", daemonUnixTime(c))
	events := strings.Split(out, "\n")
	events = events[:len(events)-1]

	nEvents := len(events)
	c.Assert(nEvents, checker.GreaterOrEqualThan, 5) //Missing expected event
	containerEvents := eventActionsByIDAndType(c, events, "since-epoch-test", "container")
	c.Assert(containerEvents, checker.HasLen, 5, check.Commentf("events: %v", events))

	c.Assert(containerEvents[0], checker.Equals, "create", check.Commentf(out))
	c.Assert(containerEvents[1], checker.Equals, "attach", check.Commentf(out))
	c.Assert(containerEvents[2], checker.Equals, "start", check.Commentf(out))
	c.Assert(containerEvents[3], checker.Equals, "die", check.Commentf(out))
	c.Assert(containerEvents[4], checker.Equals, "destroy", check.Commentf(out))
}

func (s *DockerSuite) TestEventsImageTag(c *check.C) {
	time.Sleep(1 * time.Second) // because API has seconds granularity
	since := daemonUnixTime(c)
	image := "testimageevents:tag"
	dockerCmd(c, "tag", "busybox", image)

	out, _ := dockerCmd(c, "events",
		"--since", since, "--until", daemonUnixTime(c))

	events := strings.Split(strings.TrimSpace(out), "\n")
	c.Assert(events, checker.HasLen, 1, check.Commentf("was expecting 1 event. out=%s", out))
	event := strings.TrimSpace(events[0])

	matches := eventstestutils.ScanMap(event)
	c.Assert(matchEventID(matches, image), checker.True, check.Commentf("matches: %v\nout:\n%s", matches, out))
	c.Assert(matches["action"], checker.Equals, "tag")
}

func (s *DockerSuite) TestEventsImagePull(c *check.C) {
	// TODO Windows: Enable this test once pull and reliable image names are available
	testRequires(c, DaemonIsLinux)
	since := daemonUnixTime(c)
	testRequires(c, Network)

	dockerCmd(c, "pull", "hello-world")

	out, _ := dockerCmd(c, "events",
		"--since", since, "--until", daemonUnixTime(c))

	events := strings.Split(strings.TrimSpace(out), "\n")
	event := strings.TrimSpace(events[len(events)-1])
	matches := eventstestutils.ScanMap(event)
	c.Assert(matches["id"], checker.Equals, "hello-world:latest")
	c.Assert(matches["action"], checker.Equals, "pull")

}

func (s *DockerSuite) TestEventsImageImport(c *check.C) {
	// TODO Windows CI. This should be portable once export/import are
	// more reliable (@swernli)
	testRequires(c, DaemonIsLinux)

	out, _ := dockerCmd(c, "run", "-d", "busybox", "true")
	cleanedContainerID := strings.TrimSpace(out)

	since := daemonUnixTime(c)
	out, _, err := testutil.RunCommandPipelineWithOutput(
		exec.Command(dockerBinary, "export", cleanedContainerID),
		exec.Command(dockerBinary, "import", "-"),
	)
	c.Assert(err, checker.IsNil, check.Commentf("import failed with output: %q", out))
	imageRef := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "events", "--since", since, "--until", daemonUnixTime(c), "--filter", "event=import")
	events := strings.Split(strings.TrimSpace(out), "\n")
	c.Assert(events, checker.HasLen, 1)
	matches := eventstestutils.ScanMap(events[0])
	c.Assert(matches["id"], checker.Equals, imageRef, check.Commentf("matches: %v\nout:\n%s\n", matches, out))
	c.Assert(matches["action"], checker.Equals, "import", check.Commentf("matches: %v\nout:\n%s\n", matches, out))
}

func (s *DockerSuite) TestEventsImageLoad(c *check.C) {
	testRequires(c, DaemonIsLinux)
	myImageName := "footest:v1"
	dockerCmd(c, "tag", "busybox", myImageName)
	since := daemonUnixTime(c)

	out, _ := dockerCmd(c, "images", "-q", "--no-trunc", myImageName)
	longImageID := strings.TrimSpace(out)
	c.Assert(longImageID, checker.Not(check.Equals), "", check.Commentf("Id should not be empty"))

	dockerCmd(c, "save", "-o", "saveimg.tar", myImageName)
	dockerCmd(c, "rmi", myImageName)
	out, _ = dockerCmd(c, "images", "-q", myImageName)
	noImageID := strings.TrimSpace(out)
	c.Assert(noImageID, checker.Equals, "", check.Commentf("Should not have any image"))
	dockerCmd(c, "load", "-i", "saveimg.tar")

	result := icmd.RunCommand("rm", "-rf", "saveimg.tar")
	c.Assert(result, icmd.Matches, icmd.Success)

	out, _ = dockerCmd(c, "images", "-q", "--no-trunc", myImageName)
	imageID := strings.TrimSpace(out)
	c.Assert(imageID, checker.Equals, longImageID, check.Commentf("Should have same image id as before"))

	out, _ = dockerCmd(c, "events", "--since", since, "--until", daemonUnixTime(c), "--filter", "event=load")
	events := strings.Split(strings.TrimSpace(out), "\n")
	c.Assert(events, checker.HasLen, 1)
	matches := eventstestutils.ScanMap(events[0])
	c.Assert(matches["id"], checker.Equals, imageID, check.Commentf("matches: %v\nout:\n%s\n", matches, out))
	c.Assert(matches["action"], checker.Equals, "load", check.Commentf("matches: %v\nout:\n%s\n", matches, out))

	out, _ = dockerCmd(c, "events", "--since", since, "--until", daemonUnixTime(c), "--filter", "event=save")
	events = strings.Split(strings.TrimSpace(out), "\n")
	c.Assert(events, checker.HasLen, 1)
	matches = eventstestutils.ScanMap(events[0])
	c.Assert(matches["id"], checker.Equals, imageID, check.Commentf("matches: %v\nout:\n%s\n", matches, out))
	c.Assert(matches["action"], checker.Equals, "save", check.Commentf("matches: %v\nout:\n%s\n", matches, out))
}

func (s *DockerSuite) TestEventsPluginOps(c *check.C) {
	testRequires(c, DaemonIsLinux, IsAmd64, Network)

	since := daemonUnixTime(c)

	dockerCmd(c, "plugin", "install", pNameWithTag, "--grant-all-permissions")
	dockerCmd(c, "plugin", "disable", pNameWithTag)
	dockerCmd(c, "plugin", "remove", pNameWithTag)

	out, _ := dockerCmd(c, "events", "--since", since, "--until", daemonUnixTime(c))
	events := strings.Split(out, "\n")
	events = events[:len(events)-1]

	nEvents := len(events)
	c.Assert(nEvents, checker.GreaterOrEqualThan, 4)

	pluginEvents := eventActionsByIDAndType(c, events, pNameWithTag, "plugin")
	c.Assert(pluginEvents, checker.HasLen, 4, check.Commentf("events: %v", events))

	c.Assert(pluginEvents[0], checker.Equals, "pull", check.Commentf(out))
	c.Assert(pluginEvents[1], checker.Equals, "enable", check.Commentf(out))
	c.Assert(pluginEvents[2], checker.Equals, "disable", check.Commentf(out))
	c.Assert(pluginEvents[3], checker.Equals, "remove", check.Commentf(out))
}

func (s *DockerSuite) TestEventsFilters(c *check.C) {
	since := daemonUnixTime(c)
	dockerCmd(c, "run", "--rm", "busybox", "true")
	dockerCmd(c, "run", "--rm", "busybox", "true")
	out, _ := dockerCmd(c, "events", "--since", since, "--until", daemonUnixTime(c), "--filter", "event=die")
	parseEvents(c, out, "die")

	out, _ = dockerCmd(c, "events", "--since", since, "--until", daemonUnixTime(c), "--filter", "event=die", "--filter", "event=start")
	parseEvents(c, out, "die|start")

	// make sure we at least got 2 start events
	count := strings.Count(out, "start")
	c.Assert(strings.Count(out, "start"), checker.GreaterOrEqualThan, 2, check.Commentf("should have had 2 start events but had %d, out: %s", count, out))

}

func (s *DockerSuite) TestEventsFilterImageName(c *check.C) {
	since := daemonUnixTime(c)

	out, _ := dockerCmd(c, "run", "--name", "container_1", "-d", "busybox:latest", "true")
	container1 := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "run", "--name", "container_2", "-d", "busybox", "true")
	container2 := strings.TrimSpace(out)

	name := "busybox"
	out, _ = dockerCmd(c, "events", "--since", since, "--until", daemonUnixTime(c), "--filter", fmt.Sprintf("image=%s", name))
	events := strings.Split(out, "\n")
	events = events[:len(events)-1]
	c.Assert(events, checker.Not(checker.HasLen), 0) //Expected events but found none for the image busybox:latest
	count1 := 0
	count2 := 0

	for _, e := range events {
		if strings.Contains(e, container1) {
			count1++
		} else if strings.Contains(e, container2) {
			count2++
		}
	}
	c.Assert(count1, checker.Not(checker.Equals), 0, check.Commentf("Expected event from container but got %d from %s", count1, container1))
	c.Assert(count2, checker.Not(checker.Equals), 0, check.Commentf("Expected event from container but got %d from %s", count2, container2))

}

func (s *DockerSuite) TestEventsFilterLabels(c *check.C) {
	since := daemonUnixTime(c)
	label := "io.docker.testing=foo"

	out, _ := dockerCmd(c, "run", "-d", "-l", label, "busybox:latest", "true")
	container1 := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "run", "-d", "busybox", "true")
	container2 := strings.TrimSpace(out)

	out, _ = dockerCmd(
		c,
		"events",
		"--since", since,
		"--until", daemonUnixTime(c),
		"--filter", fmt.Sprintf("label=%s", label))

	events := strings.Split(strings.TrimSpace(out), "\n")
	c.Assert(len(events), checker.Equals, 3)

	for _, e := range events {
		c.Assert(e, checker.Contains, container1)
		c.Assert(e, checker.Not(checker.Contains), container2)
	}
}

func (s *DockerSuite) TestEventsFilterImageLabels(c *check.C) {
	since := daemonUnixTime(c)
	name := "labelfiltertest"
	label := "io.docker.testing=image"

	// Build a test image.
	buildImageSuccessfully(c, name, build.WithDockerfile(fmt.Sprintf(`
		FROM busybox:latest
		LABEL %s`, label)))
	dockerCmd(c, "tag", name, "labelfiltertest:tag1")
	dockerCmd(c, "tag", name, "labelfiltertest:tag2")
	dockerCmd(c, "tag", "busybox:latest", "labelfiltertest:tag3")

	out, _ := dockerCmd(
		c,
		"events",
		"--since", since,
		"--until", daemonUnixTime(c),
		"--filter", fmt.Sprintf("label=%s", label),
		"--filter", "type=image")

	events := strings.Split(strings.TrimSpace(out), "\n")

	// 2 events from the "docker tag" command, another one is from "docker build"
	c.Assert(events, checker.HasLen, 3, check.Commentf("Events == %s", events))
	for _, e := range events {
		c.Assert(e, checker.Contains, "labelfiltertest")
	}
}

func (s *DockerSuite) TestEventsFilterContainer(c *check.C) {
	since := daemonUnixTime(c)
	nameID := make(map[string]string)

	for _, name := range []string{"container_1", "container_2"} {
		dockerCmd(c, "run", "--name", name, "busybox", "true")
		id := inspectField(c, name, "Id")
		nameID[name] = id
	}

	until := daemonUnixTime(c)

	checkEvents := func(id string, events []string) error {
		if len(events) != 4 { // create, attach, start, die
			return fmt.Errorf("expected 4 events, got %v", events)
		}
		for _, event := range events {
			matches := eventstestutils.ScanMap(event)
			if !matchEventID(matches, id) {
				return fmt.Errorf("expected event for container id %s: %s - parsed container id: %s", id, event, matches["id"])
			}
		}
		return nil
	}

	for name, ID := range nameID {
		// filter by names
		out, _ := dockerCmd(c, "events", "--since", since, "--until", until, "--filter", "container="+name)
		events := strings.Split(strings.TrimSuffix(out, "\n"), "\n")
		c.Assert(checkEvents(ID, events), checker.IsNil)

		// filter by ID's
		out, _ = dockerCmd(c, "events", "--since", since, "--until", until, "--filter", "container="+ID)
		events = strings.Split(strings.TrimSuffix(out, "\n"), "\n")
		c.Assert(checkEvents(ID, events), checker.IsNil)
	}
}

func (s *DockerSuite) TestEventsCommit(c *check.C) {
	// Problematic on Windows as cannot commit a running container
	testRequires(c, DaemonIsLinux)

	out := runSleepingContainer(c)
	cID := strings.TrimSpace(out)
	cli.WaitRun(c, cID)

	cli.DockerCmd(c, "commit", "-m", "test", cID)
	cli.DockerCmd(c, "stop", cID)
	cli.WaitExited(c, cID, 5*time.Second)

	until := daemonUnixTime(c)
	out = cli.DockerCmd(c, "events", "-f", "container="+cID, "--until="+until).Combined()
	c.Assert(out, checker.Contains, "commit", check.Commentf("Missing 'commit' log event"))
}

func (s *DockerSuite) TestEventsCopy(c *check.C) {
	// Build a test image.
	buildImageSuccessfully(c, "cpimg", build.WithDockerfile(`
		  FROM busybox
		  RUN echo HI > /file`))
	id := getIDByName(c, "cpimg")

	// Create an empty test file.
	tempFile, err := ioutil.TempFile("", "test-events-copy-")
	c.Assert(err, checker.IsNil)
	defer os.Remove(tempFile.Name())

	c.Assert(tempFile.Close(), checker.IsNil)

	dockerCmd(c, "create", "--name=cptest", id)

	dockerCmd(c, "cp", "cptest:/file", tempFile.Name())

	until := daemonUnixTime(c)
	out, _ := dockerCmd(c, "events", "--since=0", "-f", "container=cptest", "--until="+until)
	c.Assert(out, checker.Contains, "archive-path", check.Commentf("Missing 'archive-path' log event\n"))

	dockerCmd(c, "cp", tempFile.Name(), "cptest:/filecopy")

	until = daemonUnixTime(c)
	out, _ = dockerCmd(c, "events", "-f", "container=cptest", "--until="+until)
	c.Assert(out, checker.Contains, "extract-to-dir", check.Commentf("Missing 'extract-to-dir' log event"))
}

func (s *DockerSuite) TestEventsResize(c *check.C) {
	out := runSleepingContainer(c, "-d")
	cID := strings.TrimSpace(out)
	c.Assert(waitRun(cID), checker.IsNil)

	endpoint := "/containers/" + cID + "/resize?h=80&w=24"
	status, _, err := request.SockRequest("POST", endpoint, nil, daemonHost())
	c.Assert(status, checker.Equals, http.StatusOK)
	c.Assert(err, checker.IsNil)

	dockerCmd(c, "stop", cID)

	until := daemonUnixTime(c)
	out, _ = dockerCmd(c, "events", "-f", "container="+cID, "--until="+until)
	c.Assert(out, checker.Contains, "resize", check.Commentf("Missing 'resize' log event"))
}

func (s *DockerSuite) TestEventsAttach(c *check.C) {
	// TODO Windows CI: Figure out why this test fails intermittently (TP5).
	testRequires(c, DaemonIsLinux)

	out := cli.DockerCmd(c, "run", "-di", "busybox", "cat").Combined()
	cID := strings.TrimSpace(out)
	cli.WaitRun(c, cID)

	cmd := exec.Command(dockerBinary, "attach", cID)
	stdin, err := cmd.StdinPipe()
	c.Assert(err, checker.IsNil)
	defer stdin.Close()
	stdout, err := cmd.StdoutPipe()
	c.Assert(err, checker.IsNil)
	defer stdout.Close()
	c.Assert(cmd.Start(), checker.IsNil)
	defer cmd.Process.Kill()

	// Make sure we're done attaching by writing/reading some stuff
	_, err = stdin.Write([]byte("hello\n"))
	c.Assert(err, checker.IsNil)
	out, err = bufio.NewReader(stdout).ReadString('\n')
	c.Assert(err, checker.IsNil)
	c.Assert(strings.TrimSpace(out), checker.Equals, "hello", check.Commentf("expected 'hello'"))

	c.Assert(stdin.Close(), checker.IsNil)

	cli.DockerCmd(c, "kill", cID)
	cli.WaitExited(c, cID, 5*time.Second)

	until := daemonUnixTime(c)
	out = cli.DockerCmd(c, "events", "-f", "container="+cID, "--until="+until).Combined()
	c.Assert(out, checker.Contains, "attach", check.Commentf("Missing 'attach' log event"))
}

func (s *DockerSuite) TestEventsRename(c *check.C) {
	out, _ := dockerCmd(c, "run", "--name", "oldName", "busybox", "true")
	cID := strings.TrimSpace(out)
	dockerCmd(c, "rename", "oldName", "newName")

	until := daemonUnixTime(c)
	// filter by the container id because the name in the event will be the new name.
	out, _ = dockerCmd(c, "events", "-f", "container="+cID, "--until", until)
	c.Assert(out, checker.Contains, "rename", check.Commentf("Missing 'rename' log event\n"))
}

func (s *DockerSuite) TestEventsTop(c *check.C) {
	// Problematic on Windows as Windows does not support top
	testRequires(c, DaemonIsLinux)

	out := runSleepingContainer(c, "-d")
	cID := strings.TrimSpace(out)
	c.Assert(waitRun(cID), checker.IsNil)

	dockerCmd(c, "top", cID)
	dockerCmd(c, "stop", cID)

	until := daemonUnixTime(c)
	out, _ = dockerCmd(c, "events", "-f", "container="+cID, "--until="+until)
	c.Assert(out, checker.Contains, " top", check.Commentf("Missing 'top' log event"))
}

// #14316
func (s *DockerRegistrySuite) TestEventsImageFilterPush(c *check.C) {
	// Problematic to port for Windows CI during TP5 timeframe until
	// supporting push
	testRequires(c, DaemonIsLinux)
	testRequires(c, Network)
	repoName := fmt.Sprintf("%v/dockercli/testf", privateRegistryURL)

	out, _ := dockerCmd(c, "run", "-d", "busybox", "top")
	cID := strings.TrimSpace(out)
	c.Assert(waitRun(cID), checker.IsNil)

	dockerCmd(c, "commit", cID, repoName)
	dockerCmd(c, "stop", cID)
	dockerCmd(c, "push", repoName)

	until := daemonUnixTime(c)
	out, _ = dockerCmd(c, "events", "-f", "image="+repoName, "-f", "event=push", "--until", until)
	c.Assert(out, checker.Contains, repoName, check.Commentf("Missing 'push' log event for %s", repoName))
}

func (s *DockerSuite) TestEventsFilterType(c *check.C) {
	since := daemonUnixTime(c)
	name := "labelfiltertest"
	label := "io.docker.testing=image"

	// Build a test image.
	buildImageSuccessfully(c, name, build.WithDockerfile(fmt.Sprintf(`
		FROM busybox:latest
		LABEL %s`, label)))
	dockerCmd(c, "tag", name, "labelfiltertest:tag1")
	dockerCmd(c, "tag", name, "labelfiltertest:tag2")
	dockerCmd(c, "tag", "busybox:latest", "labelfiltertest:tag3")

	out, _ := dockerCmd(
		c,
		"events",
		"--since", since,
		"--until", daemonUnixTime(c),
		"--filter", fmt.Sprintf("label=%s", label),
		"--filter", "type=image")

	events := strings.Split(strings.TrimSpace(out), "\n")

	// 2 events from the "docker tag" command, another one is from "docker build"
	c.Assert(events, checker.HasLen, 3, check.Commentf("Events == %s", events))
	for _, e := range events {
		c.Assert(e, checker.Contains, "labelfiltertest")
	}

	out, _ = dockerCmd(
		c,
		"events",
		"--since", since,
		"--until", daemonUnixTime(c),
		"--filter", fmt.Sprintf("label=%s", label),
		"--filter", "type=container")
	events = strings.Split(strings.TrimSpace(out), "\n")

	// Events generated by the container that builds the image
	c.Assert(events, checker.HasLen, 3, check.Commentf("Events == %s", events))

	out, _ = dockerCmd(
		c,
		"events",
		"--since", since,
		"--until", daemonUnixTime(c),
		"--filter", "type=network")
	events = strings.Split(strings.TrimSpace(out), "\n")
	c.Assert(len(events), checker.GreaterOrEqualThan, 1, check.Commentf("Events == %s", events))
}

// #25798
func (s *DockerSuite) TestEventsSpecialFiltersWithExecCreate(c *check.C) {
	since := daemonUnixTime(c)
	runSleepingContainer(c, "--name", "test-container", "-d")
	waitRun("test-container")

	dockerCmd(c, "exec", "test-container", "echo", "hello-world")

	out, _ := dockerCmd(
		c,
		"events",
		"--since", since,
		"--until", daemonUnixTime(c),
		"--filter",
		"event='exec_create: echo hello-world'",
	)

	events := strings.Split(strings.TrimSpace(out), "\n")
	c.Assert(len(events), checker.Equals, 1, check.Commentf(out))

	out, _ = dockerCmd(
		c,
		"events",
		"--since", since,
		"--until", daemonUnixTime(c),
		"--filter",
		"event=exec_create",
	)
	c.Assert(len(events), checker.Equals, 1, check.Commentf(out))
}

func (s *DockerSuite) TestEventsFilterImageInContainerAction(c *check.C) {
	since := daemonUnixTime(c)
	dockerCmd(c, "run", "--name", "test-container", "-d", "busybox", "true")
	waitRun("test-container")

	out, _ := dockerCmd(c, "events", "--filter", "image=busybox", "--since", since, "--until", daemonUnixTime(c))
	events := strings.Split(strings.TrimSpace(out), "\n")
	c.Assert(len(events), checker.GreaterThan, 1, check.Commentf(out))
}

func (s *DockerSuite) TestEventsContainerRestart(c *check.C) {
	dockerCmd(c, "run", "-d", "--name=testEvent", "--restart=on-failure:3", "busybox", "false")

	// wait until test2 is auto removed.
	waitTime := 10 * time.Second
	if testEnv.DaemonPlatform() == "windows" {
		// Windows takes longer...
		waitTime = 90 * time.Second
	}

	err := waitInspect("testEvent", "{{ .State.Restarting }} {{ .State.Running }}", "false false", waitTime)
	c.Assert(err, checker.IsNil)

	var (
		createCount int
		startCount  int
		dieCount    int
	)
	out, _ := dockerCmd(c, "events", "--since=0", "--until", daemonUnixTime(c), "-f", "container=testEvent")
	events := strings.Split(strings.TrimSpace(out), "\n")

	nEvents := len(events)
	c.Assert(nEvents, checker.GreaterOrEqualThan, 1) //Missing expected event
	actions := eventActionsByIDAndType(c, events, "testEvent", "container")

	for _, a := range actions {
		switch a {
		case "create":
			createCount++
		case "start":
			startCount++
		case "die":
			dieCount++
		}
	}
	c.Assert(createCount, checker.Equals, 1, check.Commentf("testEvent should be created 1 times: %v", actions))
	c.Assert(startCount, checker.Equals, 4, check.Commentf("testEvent should start 4 times: %v", actions))
	c.Assert(dieCount, checker.Equals, 4, check.Commentf("testEvent should die 4 times: %v", actions))
}

func (s *DockerSuite) TestEventsSinceInTheFuture(c *check.C) {
	dockerCmd(c, "run", "--name", "test-container", "-d", "busybox", "true")
	waitRun("test-container")

	since := daemonTime(c)
	until := since.Add(time.Duration(-24) * time.Hour)
	out, _, err := dockerCmdWithError("events", "--filter", "image=busybox", "--since", parseEventTime(since), "--until", parseEventTime(until))

	c.Assert(err, checker.NotNil)
	c.Assert(out, checker.Contains, "cannot be after `until`")
}

func (s *DockerSuite) TestEventsUntilInThePast(c *check.C) {
	since := daemonUnixTime(c)

	dockerCmd(c, "run", "--name", "test-container", "-d", "busybox", "true")
	waitRun("test-container")

	until := daemonUnixTime(c)

	dockerCmd(c, "run", "--name", "test-container2", "-d", "busybox", "true")
	waitRun("test-container2")

	out, _ := dockerCmd(c, "events", "--filter", "image=busybox", "--since", since, "--until", until)

	c.Assert(out, checker.Not(checker.Contains), "test-container2")
	c.Assert(out, checker.Contains, "test-container")
}

func (s *DockerSuite) TestEventsFormat(c *check.C) {
	since := daemonUnixTime(c)
	dockerCmd(c, "run", "--rm", "busybox", "true")
	dockerCmd(c, "run", "--rm", "busybox", "true")
	out, _ := dockerCmd(c, "events", "--since", since, "--until", daemonUnixTime(c), "--format", "{{json .}}")
	dec := json.NewDecoder(strings.NewReader(out))
	// make sure we got 2 start events
	startCount := 0
	for {
		var err error
		var ev eventtypes.Message
		if err = dec.Decode(&ev); err == io.EOF {
			break
		}
		c.Assert(err, checker.IsNil)
		if ev.Status == "start" {
			startCount++
		}
	}

	c.Assert(startCount, checker.Equals, 2, check.Commentf("should have had 2 start events but had %d, out: %s", startCount, out))
}

func (s *DockerSuite) TestEventsFormatBadFunc(c *check.C) {
	// make sure it fails immediately, without receiving any event
	result := dockerCmdWithResult("events", "--format", "{{badFuncString .}}")
	c.Assert(result, icmd.Matches, icmd.Expected{
		Error:    "exit status 64",
		ExitCode: 64,
		Err:      "Error parsing format: template: :1: function \"badFuncString\" not defined",
	})
}

func (s *DockerSuite) TestEventsFormatBadField(c *check.C) {
	// make sure it fails immediately, without receiving any event
	result := dockerCmdWithResult("events", "--format", "{{.badFieldString}}")
	c.Assert(result, icmd.Matches, icmd.Expected{
		Error:    "exit status 64",
		ExitCode: 64,
		Err:      "Error parsing format: template: :1:2: executing \"\" at <.badFieldString>: can't evaluate field badFieldString in type *events.Message",
	})
}
