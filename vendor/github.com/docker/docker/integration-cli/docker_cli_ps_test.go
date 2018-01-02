package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/cli"
	"github.com/docker/docker/integration-cli/cli/build"
	"github.com/docker/docker/pkg/stringid"
	icmd "github.com/docker/docker/pkg/testutil/cmd"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestPsListContainersBase(c *check.C) {
	out := runSleepingContainer(c, "-d")
	firstID := strings.TrimSpace(out)

	out = runSleepingContainer(c, "-d")
	secondID := strings.TrimSpace(out)

	// not long running
	out, _ = dockerCmd(c, "run", "-d", "busybox", "true")
	thirdID := strings.TrimSpace(out)

	out = runSleepingContainer(c, "-d")
	fourthID := strings.TrimSpace(out)

	// make sure the second is running
	c.Assert(waitRun(secondID), checker.IsNil)

	// make sure third one is not running
	dockerCmd(c, "wait", thirdID)

	// make sure the forth is running
	c.Assert(waitRun(fourthID), checker.IsNil)

	// all
	out, _ = dockerCmd(c, "ps", "-a")
	c.Assert(assertContainerList(out, []string{fourthID, thirdID, secondID, firstID}), checker.Equals, true, check.Commentf("ALL: Container list is not in the correct order: \n%s", out))

	// running
	out, _ = dockerCmd(c, "ps")
	c.Assert(assertContainerList(out, []string{fourthID, secondID, firstID}), checker.Equals, true, check.Commentf("RUNNING: Container list is not in the correct order: \n%s", out))

	// limit
	out, _ = dockerCmd(c, "ps", "-n=2", "-a")
	expected := []string{fourthID, thirdID}
	c.Assert(assertContainerList(out, expected), checker.Equals, true, check.Commentf("LIMIT & ALL: Container list is not in the correct order: \n%s", out))

	out, _ = dockerCmd(c, "ps", "-n=2")
	c.Assert(assertContainerList(out, expected), checker.Equals, true, check.Commentf("LIMIT: Container list is not in the correct order: \n%s", out))

	// filter since
	out, _ = dockerCmd(c, "ps", "-f", "since="+firstID, "-a")
	expected = []string{fourthID, thirdID, secondID}
	c.Assert(assertContainerList(out, expected), checker.Equals, true, check.Commentf("SINCE filter & ALL: Container list is not in the correct order: \n%s", out))

	out, _ = dockerCmd(c, "ps", "-f", "since="+firstID)
	expected = []string{fourthID, secondID}
	c.Assert(assertContainerList(out, expected), checker.Equals, true, check.Commentf("SINCE filter: Container list is not in the correct order: \n%s", out))

	out, _ = dockerCmd(c, "ps", "-f", "since="+thirdID)
	expected = []string{fourthID}
	c.Assert(assertContainerList(out, expected), checker.Equals, true, check.Commentf("SINCE filter: Container list is not in the correct order: \n%s", out))

	// filter before
	out, _ = dockerCmd(c, "ps", "-f", "before="+fourthID, "-a")
	expected = []string{thirdID, secondID, firstID}
	c.Assert(assertContainerList(out, expected), checker.Equals, true, check.Commentf("BEFORE filter & ALL: Container list is not in the correct order: \n%s", out))

	out, _ = dockerCmd(c, "ps", "-f", "before="+fourthID)
	expected = []string{secondID, firstID}
	c.Assert(assertContainerList(out, expected), checker.Equals, true, check.Commentf("BEFORE filter: Container list is not in the correct order: \n%s", out))

	out, _ = dockerCmd(c, "ps", "-f", "before="+thirdID)
	expected = []string{secondID, firstID}
	c.Assert(assertContainerList(out, expected), checker.Equals, true, check.Commentf("SINCE filter: Container list is not in the correct order: \n%s", out))

	// filter since & before
	out, _ = dockerCmd(c, "ps", "-f", "since="+firstID, "-f", "before="+fourthID, "-a")
	expected = []string{thirdID, secondID}
	c.Assert(assertContainerList(out, expected), checker.Equals, true, check.Commentf("SINCE filter, BEFORE filter & ALL: Container list is not in the correct order: \n%s", out))

	out, _ = dockerCmd(c, "ps", "-f", "since="+firstID, "-f", "before="+fourthID)
	expected = []string{secondID}
	c.Assert(assertContainerList(out, expected), checker.Equals, true, check.Commentf("SINCE filter, BEFORE filter: Container list is not in the correct order: \n%s", out))

	// filter since & limit
	out, _ = dockerCmd(c, "ps", "-f", "since="+firstID, "-n=2", "-a")
	expected = []string{fourthID, thirdID}

	c.Assert(assertContainerList(out, expected), checker.Equals, true, check.Commentf("SINCE filter, LIMIT & ALL: Container list is not in the correct order: \n%s", out))

	out, _ = dockerCmd(c, "ps", "-f", "since="+firstID, "-n=2")
	c.Assert(assertContainerList(out, expected), checker.Equals, true, check.Commentf("SINCE filter, LIMIT: Container list is not in the correct order: \n%s", out))

	// filter before & limit
	out, _ = dockerCmd(c, "ps", "-f", "before="+fourthID, "-n=1", "-a")
	expected = []string{thirdID}
	c.Assert(assertContainerList(out, expected), checker.Equals, true, check.Commentf("BEFORE filter, LIMIT & ALL: Container list is not in the correct order: \n%s", out))

	out, _ = dockerCmd(c, "ps", "-f", "before="+fourthID, "-n=1")
	c.Assert(assertContainerList(out, expected), checker.Equals, true, check.Commentf("BEFORE filter, LIMIT: Container list is not in the correct order: \n%s", out))

	// filter since & filter before & limit
	out, _ = dockerCmd(c, "ps", "-f", "since="+firstID, "-f", "before="+fourthID, "-n=1", "-a")
	expected = []string{thirdID}
	c.Assert(assertContainerList(out, expected), checker.Equals, true, check.Commentf("SINCE filter, BEFORE filter, LIMIT & ALL: Container list is not in the correct order: \n%s", out))

	out, _ = dockerCmd(c, "ps", "-f", "since="+firstID, "-f", "before="+fourthID, "-n=1")
	c.Assert(assertContainerList(out, expected), checker.Equals, true, check.Commentf("SINCE filter, BEFORE filter, LIMIT: Container list is not in the correct order: \n%s", out))

}

func assertContainerList(out string, expected []string) bool {
	lines := strings.Split(strings.Trim(out, "\n "), "\n")

	if len(lines)-1 != len(expected) {
		return false
	}

	containerIDIndex := strings.Index(lines[0], "CONTAINER ID")
	for i := 0; i < len(expected); i++ {
		foundID := lines[i+1][containerIDIndex : containerIDIndex+12]
		if foundID != expected[i][:12] {
			return false
		}
	}

	return true
}

// FIXME(vdemeester) Move this into a unit test in daemon package
func (s *DockerSuite) TestPsListContainersInvalidFilterName(c *check.C) {
	out, _, err := dockerCmdWithError("ps", "-f", "invalidFilter=test")
	c.Assert(err, checker.NotNil)
	c.Assert(out, checker.Contains, "Invalid filter")
}

func (s *DockerSuite) TestPsListContainersSize(c *check.C) {
	// Problematic on Windows as it doesn't report the size correctly @swernli
	testRequires(c, DaemonIsLinux)
	dockerCmd(c, "run", "-d", "busybox")

	baseOut, _ := dockerCmd(c, "ps", "-s", "-n=1")
	baseLines := strings.Split(strings.Trim(baseOut, "\n "), "\n")
	baseSizeIndex := strings.Index(baseLines[0], "SIZE")
	baseFoundsize := baseLines[1][baseSizeIndex:]
	baseBytes, err := strconv.Atoi(strings.Split(baseFoundsize, "B")[0])
	c.Assert(err, checker.IsNil)

	name := "test_size"
	dockerCmd(c, "run", "--name", name, "busybox", "sh", "-c", "echo 1 > test")
	id := getIDByName(c, name)

	var result *icmd.Result

	wait := make(chan struct{})
	go func() {
		result = icmd.RunCommand(dockerBinary, "ps", "-s", "-n=1")
		close(wait)
	}()
	select {
	case <-wait:
	case <-time.After(3 * time.Second):
		c.Fatalf("Calling \"docker ps -s\" timed out!")
	}
	result.Assert(c, icmd.Success)
	lines := strings.Split(strings.Trim(result.Combined(), "\n "), "\n")
	c.Assert(lines, checker.HasLen, 2, check.Commentf("Expected 2 lines for 'ps -s -n=1' output, got %d", len(lines)))
	sizeIndex := strings.Index(lines[0], "SIZE")
	idIndex := strings.Index(lines[0], "CONTAINER ID")
	foundID := lines[1][idIndex : idIndex+12]
	c.Assert(foundID, checker.Equals, id[:12], check.Commentf("Expected id %s, got %s", id[:12], foundID))
	expectedSize := fmt.Sprintf("%dB", (2 + baseBytes))
	foundSize := lines[1][sizeIndex:]
	c.Assert(foundSize, checker.Contains, expectedSize, check.Commentf("Expected size %q, got %q", expectedSize, foundSize))
}

func (s *DockerSuite) TestPsListContainersFilterStatus(c *check.C) {
	// start exited container
	out := cli.DockerCmd(c, "run", "-d", "busybox").Combined()
	firstID := strings.TrimSpace(out)

	// make sure the exited container is not running
	cli.DockerCmd(c, "wait", firstID)

	// start running container
	out = cli.DockerCmd(c, "run", "-itd", "busybox").Combined()
	secondID := strings.TrimSpace(out)

	// filter containers by exited
	out = cli.DockerCmd(c, "ps", "--no-trunc", "-q", "--filter=status=exited").Combined()
	containerOut := strings.TrimSpace(out)
	c.Assert(containerOut, checker.Equals, firstID)

	out = cli.DockerCmd(c, "ps", "-a", "--no-trunc", "-q", "--filter=status=running").Combined()
	containerOut = strings.TrimSpace(out)
	c.Assert(containerOut, checker.Equals, secondID)

	result := cli.Docker(cli.Args("ps", "-a", "-q", "--filter=status=rubbish"), cli.WithTimeout(time.Second*60))
	c.Assert(result, icmd.Matches, icmd.Expected{
		ExitCode: 1,
		Err:      "Unrecognised filter value for status",
	})

	// Windows doesn't support pausing of containers
	if testEnv.DaemonPlatform() != "windows" {
		// pause running container
		out = cli.DockerCmd(c, "run", "-itd", "busybox").Combined()
		pausedID := strings.TrimSpace(out)
		cli.DockerCmd(c, "pause", pausedID)
		// make sure the container is unpaused to let the daemon stop it properly
		defer func() { cli.DockerCmd(c, "unpause", pausedID) }()

		out = cli.DockerCmd(c, "ps", "--no-trunc", "-q", "--filter=status=paused").Combined()
		containerOut = strings.TrimSpace(out)
		c.Assert(containerOut, checker.Equals, pausedID)
	}
}

func (s *DockerSuite) TestPsListContainersFilterHealth(c *check.C) {
	// Test legacy no health check
	out := runSleepingContainer(c, "--name=none_legacy")
	containerID := strings.TrimSpace(out)

	cli.WaitRun(c, containerID)

	out = cli.DockerCmd(c, "ps", "-q", "-l", "--no-trunc", "--filter=health=none").Combined()
	containerOut := strings.TrimSpace(out)
	c.Assert(containerOut, checker.Equals, containerID, check.Commentf("Expected id %s, got %s for legacy none filter, output: %q", containerID, containerOut, out))

	// Test no health check specified explicitly
	out = runSleepingContainer(c, "--name=none", "--no-healthcheck")
	containerID = strings.TrimSpace(out)

	cli.WaitRun(c, containerID)

	out = cli.DockerCmd(c, "ps", "-q", "-l", "--no-trunc", "--filter=health=none").Combined()
	containerOut = strings.TrimSpace(out)
	c.Assert(containerOut, checker.Equals, containerID, check.Commentf("Expected id %s, got %s for none filter, output: %q", containerID, containerOut, out))

	// Test failing health check
	out = runSleepingContainer(c, "--name=failing_container", "--health-cmd=exit 1", "--health-interval=1s")
	containerID = strings.TrimSpace(out)

	waitForHealthStatus(c, "failing_container", "starting", "unhealthy")

	out = cli.DockerCmd(c, "ps", "-q", "--no-trunc", "--filter=health=unhealthy").Combined()
	containerOut = strings.TrimSpace(out)
	c.Assert(containerOut, checker.Equals, containerID, check.Commentf("Expected containerID %s, got %s for unhealthy filter, output: %q", containerID, containerOut, out))

	// Check passing healthcheck
	out = runSleepingContainer(c, "--name=passing_container", "--health-cmd=exit 0", "--health-interval=1s")
	containerID = strings.TrimSpace(out)

	waitForHealthStatus(c, "passing_container", "starting", "healthy")

	out = cli.DockerCmd(c, "ps", "-q", "--no-trunc", "--filter=health=healthy").Combined()
	containerOut = strings.TrimSpace(out)
	c.Assert(containerOut, checker.Equals, containerID, check.Commentf("Expected containerID %s, got %s for healthy filter, output: %q", containerID, containerOut, out))
}

func (s *DockerSuite) TestPsListContainersFilterID(c *check.C) {
	// start container
	out, _ := dockerCmd(c, "run", "-d", "busybox")
	firstID := strings.TrimSpace(out)

	// start another container
	runSleepingContainer(c)

	// filter containers by id
	out, _ = dockerCmd(c, "ps", "-a", "-q", "--filter=id="+firstID)
	containerOut := strings.TrimSpace(out)
	c.Assert(containerOut, checker.Equals, firstID[:12], check.Commentf("Expected id %s, got %s for exited filter, output: %q", firstID[:12], containerOut, out))
}

func (s *DockerSuite) TestPsListContainersFilterName(c *check.C) {
	// start container
	dockerCmd(c, "run", "--name=a_name_to_match", "busybox")
	id := getIDByName(c, "a_name_to_match")

	// start another container
	runSleepingContainer(c, "--name=b_name_to_match")

	// filter containers by name
	out, _ := dockerCmd(c, "ps", "-a", "-q", "--filter=name=a_name_to_match")
	containerOut := strings.TrimSpace(out)
	c.Assert(containerOut, checker.Equals, id[:12], check.Commentf("Expected id %s, got %s for exited filter, output: %q", id[:12], containerOut, out))
}

// Test for the ancestor filter for ps.
// There is also the same test but with image:tag@digest in docker_cli_by_digest_test.go
//
// What the test setups :
// - Create 2 image based on busybox using the same repository but different tags
// - Create an image based on the previous image (images_ps_filter_test2)
// - Run containers for each of those image (busybox, images_ps_filter_test1, images_ps_filter_test2)
// - Filter them out :P
func (s *DockerSuite) TestPsListContainersFilterAncestorImage(c *check.C) {
	// Build images
	imageName1 := "images_ps_filter_test1"
	buildImageSuccessfully(c, imageName1, build.WithDockerfile(`FROM busybox
		 LABEL match me 1`))
	imageID1 := getIDByName(c, imageName1)

	imageName1Tagged := "images_ps_filter_test1:tag"
	buildImageSuccessfully(c, imageName1Tagged, build.WithDockerfile(`FROM busybox
		 LABEL match me 1 tagged`))
	imageID1Tagged := getIDByName(c, imageName1Tagged)

	imageName2 := "images_ps_filter_test2"
	buildImageSuccessfully(c, imageName2, build.WithDockerfile(fmt.Sprintf(`FROM %s
		 LABEL match me 2`, imageName1)))
	imageID2 := getIDByName(c, imageName2)

	// start containers
	dockerCmd(c, "run", "--name=first", "busybox", "echo", "hello")
	firstID := getIDByName(c, "first")

	// start another container
	dockerCmd(c, "run", "--name=second", "busybox", "echo", "hello")
	secondID := getIDByName(c, "second")

	// start third container
	dockerCmd(c, "run", "--name=third", imageName1, "echo", "hello")
	thirdID := getIDByName(c, "third")

	// start fourth container
	dockerCmd(c, "run", "--name=fourth", imageName1Tagged, "echo", "hello")
	fourthID := getIDByName(c, "fourth")

	// start fifth container
	dockerCmd(c, "run", "--name=fifth", imageName2, "echo", "hello")
	fifthID := getIDByName(c, "fifth")

	var filterTestSuite = []struct {
		filterName  string
		expectedIDs []string
	}{
		// non existent stuff
		{"nonexistent", []string{}},
		{"nonexistent:tag", []string{}},
		// image
		{"busybox", []string{firstID, secondID, thirdID, fourthID, fifthID}},
		{imageName1, []string{thirdID, fifthID}},
		{imageName2, []string{fifthID}},
		// image:tag
		{fmt.Sprintf("%s:latest", imageName1), []string{thirdID, fifthID}},
		{imageName1Tagged, []string{fourthID}},
		// short-id
		{stringid.TruncateID(imageID1), []string{thirdID, fifthID}},
		{stringid.TruncateID(imageID2), []string{fifthID}},
		// full-id
		{imageID1, []string{thirdID, fifthID}},
		{imageID1Tagged, []string{fourthID}},
		{imageID2, []string{fifthID}},
	}

	var out string
	for _, filter := range filterTestSuite {
		out, _ = dockerCmd(c, "ps", "-a", "-q", "--no-trunc", "--filter=ancestor="+filter.filterName)
		checkPsAncestorFilterOutput(c, out, filter.filterName, filter.expectedIDs)
	}

	// Multiple ancestor filter
	out, _ = dockerCmd(c, "ps", "-a", "-q", "--no-trunc", "--filter=ancestor="+imageName2, "--filter=ancestor="+imageName1Tagged)
	checkPsAncestorFilterOutput(c, out, imageName2+","+imageName1Tagged, []string{fourthID, fifthID})
}

func checkPsAncestorFilterOutput(c *check.C, out string, filterName string, expectedIDs []string) {
	actualIDs := []string{}
	if out != "" {
		actualIDs = strings.Split(out[:len(out)-1], "\n")
	}
	sort.Strings(actualIDs)
	sort.Strings(expectedIDs)

	c.Assert(actualIDs, checker.HasLen, len(expectedIDs), check.Commentf("Expected filtered container(s) for %s ancestor filter to be %v:%v, got %v:%v", filterName, len(expectedIDs), expectedIDs, len(actualIDs), actualIDs))
	if len(expectedIDs) > 0 {
		same := true
		for i := range expectedIDs {
			if actualIDs[i] != expectedIDs[i] {
				c.Logf("%s, %s", actualIDs[i], expectedIDs[i])
				same = false
				break
			}
		}
		c.Assert(same, checker.Equals, true, check.Commentf("Expected filtered container(s) for %s ancestor filter to be %v, got %v", filterName, expectedIDs, actualIDs))
	}
}

func (s *DockerSuite) TestPsListContainersFilterLabel(c *check.C) {
	// start container
	dockerCmd(c, "run", "--name=first", "-l", "match=me", "-l", "second=tag", "busybox")
	firstID := getIDByName(c, "first")

	// start another container
	dockerCmd(c, "run", "--name=second", "-l", "match=me too", "busybox")
	secondID := getIDByName(c, "second")

	// start third container
	dockerCmd(c, "run", "--name=third", "-l", "nomatch=me", "busybox")
	thirdID := getIDByName(c, "third")

	// filter containers by exact match
	out, _ := dockerCmd(c, "ps", "-a", "-q", "--no-trunc", "--filter=label=match=me")
	containerOut := strings.TrimSpace(out)
	c.Assert(containerOut, checker.Equals, firstID, check.Commentf("Expected id %s, got %s for exited filter, output: %q", firstID, containerOut, out))

	// filter containers by two labels
	out, _ = dockerCmd(c, "ps", "-a", "-q", "--no-trunc", "--filter=label=match=me", "--filter=label=second=tag")
	containerOut = strings.TrimSpace(out)
	c.Assert(containerOut, checker.Equals, firstID, check.Commentf("Expected id %s, got %s for exited filter, output: %q", firstID, containerOut, out))

	// filter containers by two labels, but expect not found because of AND behavior
	out, _ = dockerCmd(c, "ps", "-a", "-q", "--no-trunc", "--filter=label=match=me", "--filter=label=second=tag-no")
	containerOut = strings.TrimSpace(out)
	c.Assert(containerOut, checker.Equals, "", check.Commentf("Expected nothing, got %s for exited filter, output: %q", containerOut, out))

	// filter containers by exact key
	out, _ = dockerCmd(c, "ps", "-a", "-q", "--no-trunc", "--filter=label=match")
	containerOut = strings.TrimSpace(out)
	c.Assert(containerOut, checker.Contains, firstID)
	c.Assert(containerOut, checker.Contains, secondID)
	c.Assert(containerOut, checker.Not(checker.Contains), thirdID)
}

func (s *DockerSuite) TestPsListContainersFilterExited(c *check.C) {
	runSleepingContainer(c, "--name=sleep")

	dockerCmd(c, "run", "--name", "zero1", "busybox", "true")
	firstZero := getIDByName(c, "zero1")

	dockerCmd(c, "run", "--name", "zero2", "busybox", "true")
	secondZero := getIDByName(c, "zero2")

	out, _, err := dockerCmdWithError("run", "--name", "nonzero1", "busybox", "false")
	c.Assert(err, checker.NotNil, check.Commentf("Should fail.", out, err))

	firstNonZero := getIDByName(c, "nonzero1")

	out, _, err = dockerCmdWithError("run", "--name", "nonzero2", "busybox", "false")
	c.Assert(err, checker.NotNil, check.Commentf("Should fail.", out, err))
	secondNonZero := getIDByName(c, "nonzero2")

	// filter containers by exited=0
	out, _ = dockerCmd(c, "ps", "-a", "-q", "--no-trunc", "--filter=exited=0")
	ids := strings.Split(strings.TrimSpace(out), "\n")
	c.Assert(ids, checker.HasLen, 2, check.Commentf("Should be 2 zero exited containers got %d: %s", len(ids), out))
	c.Assert(ids[0], checker.Equals, secondZero, check.Commentf("First in list should be %q, got %q", secondZero, ids[0]))
	c.Assert(ids[1], checker.Equals, firstZero, check.Commentf("Second in list should be %q, got %q", firstZero, ids[1]))

	out, _ = dockerCmd(c, "ps", "-a", "-q", "--no-trunc", "--filter=exited=1")
	ids = strings.Split(strings.TrimSpace(out), "\n")
	c.Assert(ids, checker.HasLen, 2, check.Commentf("Should be 2 zero exited containers got %d", len(ids)))
	c.Assert(ids[0], checker.Equals, secondNonZero, check.Commentf("First in list should be %q, got %q", secondNonZero, ids[0]))
	c.Assert(ids[1], checker.Equals, firstNonZero, check.Commentf("Second in list should be %q, got %q", firstNonZero, ids[1]))

}

func (s *DockerSuite) TestPsRightTagName(c *check.C) {
	// TODO Investigate further why this fails on Windows to Windows CI
	testRequires(c, DaemonIsLinux)
	tag := "asybox:shmatest"
	dockerCmd(c, "tag", "busybox", tag)

	var id1 string
	out := runSleepingContainer(c)
	id1 = strings.TrimSpace(string(out))

	var id2 string
	out = runSleepingContainerInImage(c, tag)
	id2 = strings.TrimSpace(string(out))

	var imageID string
	out = inspectField(c, "busybox", "Id")
	imageID = strings.TrimSpace(string(out))

	var id3 string
	out = runSleepingContainerInImage(c, imageID)
	id3 = strings.TrimSpace(string(out))

	out, _ = dockerCmd(c, "ps", "--no-trunc")
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	// skip header
	lines = lines[1:]
	c.Assert(lines, checker.HasLen, 3, check.Commentf("There should be 3 running container, got %d", len(lines)))
	for _, line := range lines {
		f := strings.Fields(line)
		switch f[0] {
		case id1:
			c.Assert(f[1], checker.Equals, "busybox", check.Commentf("Expected %s tag for id %s, got %s", "busybox", id1, f[1]))
		case id2:
			c.Assert(f[1], checker.Equals, tag, check.Commentf("Expected %s tag for id %s, got %s", tag, id2, f[1]))
		case id3:
			c.Assert(f[1], checker.Equals, imageID, check.Commentf("Expected %s imageID for id %s, got %s", tag, id3, f[1]))
		default:
			c.Fatalf("Unexpected id %s, expected %s and %s and %s", f[0], id1, id2, id3)
		}
	}
}

func (s *DockerSuite) TestPsLinkedWithNoTrunc(c *check.C) {
	// Problematic on Windows as it doesn't support links as of Jan 2016
	testRequires(c, DaemonIsLinux)
	runSleepingContainer(c, "--name=first")
	runSleepingContainer(c, "--name=second", "--link=first:first")

	out, _ := dockerCmd(c, "ps", "--no-trunc")
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	// strip header
	lines = lines[1:]
	expected := []string{"second", "first,second/first"}
	var names []string
	for _, l := range lines {
		fields := strings.Fields(l)
		names = append(names, fields[len(fields)-1])
	}
	c.Assert(expected, checker.DeepEquals, names, check.Commentf("Expected array: %v, got: %v", expected, names))
}

func (s *DockerSuite) TestPsGroupPortRange(c *check.C) {
	// Problematic on Windows as it doesn't support port ranges as of Jan 2016
	testRequires(c, DaemonIsLinux)
	portRange := "3850-3900"
	dockerCmd(c, "run", "-d", "--name", "porttest", "-p", portRange+":"+portRange, "busybox", "top")

	out, _ := dockerCmd(c, "ps")

	c.Assert(string(out), checker.Contains, portRange, check.Commentf("docker ps output should have had the port range %q: %s", portRange, string(out)))

}

func (s *DockerSuite) TestPsWithSize(c *check.C) {
	// Problematic on Windows as it doesn't report the size correctly @swernli
	testRequires(c, DaemonIsLinux)
	dockerCmd(c, "run", "-d", "--name", "sizetest", "busybox", "top")

	out, _ := dockerCmd(c, "ps", "--size")
	c.Assert(out, checker.Contains, "virtual", check.Commentf("docker ps with --size should show virtual size of container"))
}

func (s *DockerSuite) TestPsListContainersFilterCreated(c *check.C) {
	// create a container
	out, _ := dockerCmd(c, "create", "busybox")
	cID := strings.TrimSpace(out)
	shortCID := cID[:12]

	// Make sure it DOESN'T show up w/o a '-a' for normal 'ps'
	out, _ = dockerCmd(c, "ps", "-q")
	c.Assert(out, checker.Not(checker.Contains), shortCID, check.Commentf("Should have not seen '%s' in ps output:\n%s", shortCID, out))

	// Make sure it DOES show up as 'Created' for 'ps -a'
	out, _ = dockerCmd(c, "ps", "-a")

	hits := 0
	for _, line := range strings.Split(out, "\n") {
		if !strings.Contains(line, shortCID) {
			continue
		}
		hits++
		c.Assert(line, checker.Contains, "Created", check.Commentf("Missing 'Created' on '%s'", line))
	}

	c.Assert(hits, checker.Equals, 1, check.Commentf("Should have seen '%s' in ps -a output once:%d\n%s", shortCID, hits, out))

	// filter containers by 'create' - note, no -a needed
	out, _ = dockerCmd(c, "ps", "-q", "-f", "status=created")
	containerOut := strings.TrimSpace(out)
	c.Assert(cID, checker.HasPrefix, containerOut)
}

func (s *DockerSuite) TestPsFormatMultiNames(c *check.C) {
	// Problematic on Windows as it doesn't support link as of Jan 2016
	testRequires(c, DaemonIsLinux)
	//create 2 containers and link them
	dockerCmd(c, "run", "--name=child", "-d", "busybox", "top")
	dockerCmd(c, "run", "--name=parent", "--link=child:linkedone", "-d", "busybox", "top")

	//use the new format capabilities to only list the names and --no-trunc to get all names
	out, _ := dockerCmd(c, "ps", "--format", "{{.Names}}", "--no-trunc")
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	expected := []string{"parent", "child,parent/linkedone"}
	var names []string
	names = append(names, lines...)
	c.Assert(expected, checker.DeepEquals, names, check.Commentf("Expected array with non-truncated names: %v, got: %v", expected, names))

	//now list without turning off truncation and make sure we only get the non-link names
	out, _ = dockerCmd(c, "ps", "--format", "{{.Names}}")
	lines = strings.Split(strings.TrimSpace(string(out)), "\n")
	expected = []string{"parent", "child"}
	var truncNames []string
	truncNames = append(truncNames, lines...)
	c.Assert(expected, checker.DeepEquals, truncNames, check.Commentf("Expected array with truncated names: %v, got: %v", expected, truncNames))
}

// Test for GitHub issue #21772
func (s *DockerSuite) TestPsNamesMultipleTime(c *check.C) {
	runSleepingContainer(c, "--name=test1")
	runSleepingContainer(c, "--name=test2")

	//use the new format capabilities to list the names twice
	out, _ := dockerCmd(c, "ps", "--format", "{{.Names}} {{.Names}}")
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	expected := []string{"test2 test2", "test1 test1"}
	var names []string
	names = append(names, lines...)
	c.Assert(expected, checker.DeepEquals, names, check.Commentf("Expected array with names displayed twice: %v, got: %v", expected, names))
}

func (s *DockerSuite) TestPsFormatHeaders(c *check.C) {
	// make sure no-container "docker ps" still prints the header row
	out, _ := dockerCmd(c, "ps", "--format", "table {{.ID}}")
	c.Assert(out, checker.Equals, "CONTAINER ID\n", check.Commentf(`Expected 'CONTAINER ID\n', got %v`, out))

	// verify that "docker ps" with a container still prints the header row also
	runSleepingContainer(c, "--name=test")
	out, _ = dockerCmd(c, "ps", "--format", "table {{.Names}}")
	c.Assert(out, checker.Equals, "NAMES\ntest\n", check.Commentf(`Expected 'NAMES\ntest\n', got %v`, out))
}

func (s *DockerSuite) TestPsDefaultFormatAndQuiet(c *check.C) {
	config := `{
		"psFormat": "default {{ .ID }}"
}`
	d, err := ioutil.TempDir("", "integration-cli-")
	c.Assert(err, checker.IsNil)
	defer os.RemoveAll(d)

	err = ioutil.WriteFile(filepath.Join(d, "config.json"), []byte(config), 0644)
	c.Assert(err, checker.IsNil)

	out := runSleepingContainer(c, "--name=test")
	id := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "--config", d, "ps", "-q")
	c.Assert(id, checker.HasPrefix, strings.TrimSpace(out), check.Commentf("Expected to print only the container id, got %v\n", out))
}

// Test for GitHub issue #12595
func (s *DockerSuite) TestPsImageIDAfterUpdate(c *check.C) {
	// TODO: Investigate why this fails on Windows to Windows CI further.
	testRequires(c, DaemonIsLinux)
	originalImageName := "busybox:TestPsImageIDAfterUpdate-original"
	updatedImageName := "busybox:TestPsImageIDAfterUpdate-updated"

	icmd.RunCommand(dockerBinary, "tag", "busybox:latest", originalImageName).Assert(c, icmd.Success)

	originalImageID := getIDByName(c, originalImageName)

	result := icmd.RunCommand(dockerBinary, append([]string{"run", "-d", originalImageName}, sleepCommandForDaemonPlatform()...)...)
	result.Assert(c, icmd.Success)
	containerID := strings.TrimSpace(result.Combined())

	result = icmd.RunCommand(dockerBinary, "ps", "--no-trunc")
	result.Assert(c, icmd.Success)

	lines := strings.Split(strings.TrimSpace(string(result.Combined())), "\n")
	// skip header
	lines = lines[1:]
	c.Assert(len(lines), checker.Equals, 1)

	for _, line := range lines {
		f := strings.Fields(line)
		c.Assert(f[1], checker.Equals, originalImageName)
	}

	icmd.RunCommand(dockerBinary, "commit", containerID, updatedImageName).Assert(c, icmd.Success)
	icmd.RunCommand(dockerBinary, "tag", updatedImageName, originalImageName).Assert(c, icmd.Success)

	result = icmd.RunCommand(dockerBinary, "ps", "--no-trunc")
	result.Assert(c, icmd.Success)

	lines = strings.Split(strings.TrimSpace(string(result.Combined())), "\n")
	// skip header
	lines = lines[1:]
	c.Assert(len(lines), checker.Equals, 1)

	for _, line := range lines {
		f := strings.Fields(line)
		c.Assert(f[1], checker.Equals, originalImageID)
	}

}

func (s *DockerSuite) TestPsNotShowPortsOfStoppedContainer(c *check.C) {
	testRequires(c, DaemonIsLinux)
	dockerCmd(c, "run", "--name=foo", "-d", "-p", "5000:5000", "busybox", "top")
	c.Assert(waitRun("foo"), checker.IsNil)
	out, _ := dockerCmd(c, "ps")
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	expected := "0.0.0.0:5000->5000/tcp"
	fields := strings.Fields(lines[1])
	c.Assert(fields[len(fields)-2], checker.Equals, expected, check.Commentf("Expected: %v, got: %v", expected, fields[len(fields)-2]))

	dockerCmd(c, "kill", "foo")
	dockerCmd(c, "wait", "foo")
	out, _ = dockerCmd(c, "ps", "-l")
	lines = strings.Split(strings.TrimSpace(string(out)), "\n")
	fields = strings.Fields(lines[1])
	c.Assert(fields[len(fields)-2], checker.Not(checker.Equals), expected, check.Commentf("Should not got %v", expected))
}

func (s *DockerSuite) TestPsShowMounts(c *check.C) {
	prefix, slash := getPrefixAndSlashFromDaemonPlatform()

	mp := prefix + slash + "test"

	dockerCmd(c, "volume", "create", "ps-volume-test")
	// volume mount containers
	runSleepingContainer(c, "--name=volume-test-1", "--volume", "ps-volume-test:"+mp)
	c.Assert(waitRun("volume-test-1"), checker.IsNil)
	runSleepingContainer(c, "--name=volume-test-2", "--volume", mp)
	c.Assert(waitRun("volume-test-2"), checker.IsNil)
	// bind mount container
	var bindMountSource string
	var bindMountDestination string
	if DaemonIsWindows() {
		bindMountSource = "c:\\"
		bindMountDestination = "c:\\t"
	} else {
		bindMountSource = "/tmp"
		bindMountDestination = "/t"
	}
	runSleepingContainer(c, "--name=bind-mount-test", "-v", bindMountSource+":"+bindMountDestination)
	c.Assert(waitRun("bind-mount-test"), checker.IsNil)

	out, _ := dockerCmd(c, "ps", "--format", "{{.Names}} {{.Mounts}}")

	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	c.Assert(lines, checker.HasLen, 3)

	fields := strings.Fields(lines[0])
	c.Assert(fields, checker.HasLen, 2)
	c.Assert(fields[0], checker.Equals, "bind-mount-test")
	c.Assert(fields[1], checker.Equals, bindMountSource)

	fields = strings.Fields(lines[1])
	c.Assert(fields, checker.HasLen, 2)

	anonymousVolumeID := fields[1]

	fields = strings.Fields(lines[2])
	c.Assert(fields[1], checker.Equals, "ps-volume-test")

	// filter by volume name
	out, _ = dockerCmd(c, "ps", "--format", "{{.Names}} {{.Mounts}}", "--filter", "volume=ps-volume-test")

	lines = strings.Split(strings.TrimSpace(string(out)), "\n")
	c.Assert(lines, checker.HasLen, 1)

	fields = strings.Fields(lines[0])
	c.Assert(fields[1], checker.Equals, "ps-volume-test")

	// empty results filtering by unknown volume
	out, _ = dockerCmd(c, "ps", "--format", "{{.Names}} {{.Mounts}}", "--filter", "volume=this-volume-should-not-exist")
	c.Assert(strings.TrimSpace(string(out)), checker.HasLen, 0)

	// filter by mount destination
	out, _ = dockerCmd(c, "ps", "--format", "{{.Names}} {{.Mounts}}", "--filter", "volume="+mp)

	lines = strings.Split(strings.TrimSpace(string(out)), "\n")
	c.Assert(lines, checker.HasLen, 2)

	fields = strings.Fields(lines[0])
	c.Assert(fields[1], checker.Equals, anonymousVolumeID)
	fields = strings.Fields(lines[1])
	c.Assert(fields[1], checker.Equals, "ps-volume-test")

	// filter by bind mount source
	out, _ = dockerCmd(c, "ps", "--format", "{{.Names}} {{.Mounts}}", "--filter", "volume="+bindMountSource)

	lines = strings.Split(strings.TrimSpace(string(out)), "\n")
	c.Assert(lines, checker.HasLen, 1)

	fields = strings.Fields(lines[0])
	c.Assert(fields, checker.HasLen, 2)
	c.Assert(fields[0], checker.Equals, "bind-mount-test")
	c.Assert(fields[1], checker.Equals, bindMountSource)

	// filter by bind mount destination
	out, _ = dockerCmd(c, "ps", "--format", "{{.Names}} {{.Mounts}}", "--filter", "volume="+bindMountDestination)

	lines = strings.Split(strings.TrimSpace(string(out)), "\n")
	c.Assert(lines, checker.HasLen, 1)

	fields = strings.Fields(lines[0])
	c.Assert(fields, checker.HasLen, 2)
	c.Assert(fields[0], checker.Equals, "bind-mount-test")
	c.Assert(fields[1], checker.Equals, bindMountSource)

	// empty results filtering by unknown mount point
	out, _ = dockerCmd(c, "ps", "--format", "{{.Names}} {{.Mounts}}", "--filter", "volume="+prefix+slash+"this-path-was-never-mounted")
	c.Assert(strings.TrimSpace(string(out)), checker.HasLen, 0)
}

func (s *DockerSuite) TestPsFormatSize(c *check.C) {
	testRequires(c, DaemonIsLinux)
	runSleepingContainer(c)

	out, _ := dockerCmd(c, "ps", "--format", "table {{.Size}}")
	lines := strings.Split(out, "\n")
	c.Assert(lines[1], checker.Not(checker.Equals), "0 B", check.Commentf("Should not display a size of 0 B"))

	out, _ = dockerCmd(c, "ps", "--size", "--format", "table {{.Size}}")
	lines = strings.Split(out, "\n")
	c.Assert(lines[0], checker.Equals, "SIZE", check.Commentf("Should only have one size column"))

	out, _ = dockerCmd(c, "ps", "--size", "--format", "raw")
	lines = strings.Split(out, "\n")
	c.Assert(lines[8], checker.HasPrefix, "size:", check.Commentf("Size should be appended on a newline"))
}

func (s *DockerSuite) TestPsListContainersFilterNetwork(c *check.C) {
	// TODO default network on Windows is not called "bridge", and creating a
	// custom network fails on Windows fails with "Error response from daemon: plugin not found")
	testRequires(c, DaemonIsLinux)

	// create some containers
	runSleepingContainer(c, "--net=bridge", "--name=onbridgenetwork")
	runSleepingContainer(c, "--net=none", "--name=onnonenetwork")

	// Filter docker ps on non existing network
	out, _ := dockerCmd(c, "ps", "--filter", "network=doesnotexist")
	containerOut := strings.TrimSpace(string(out))
	lines := strings.Split(containerOut, "\n")

	// skip header
	lines = lines[1:]

	// ps output should have no containers
	c.Assert(lines, checker.HasLen, 0)

	// Filter docker ps on network bridge
	out, _ = dockerCmd(c, "ps", "--filter", "network=bridge")
	containerOut = strings.TrimSpace(string(out))

	lines = strings.Split(containerOut, "\n")

	// skip header
	lines = lines[1:]

	// ps output should have only one container
	c.Assert(lines, checker.HasLen, 1)

	// Making sure onbridgenetwork is on the output
	c.Assert(containerOut, checker.Contains, "onbridgenetwork", check.Commentf("Missing the container on network\n"))

	// Filter docker ps on networks bridge and none
	out, _ = dockerCmd(c, "ps", "--filter", "network=bridge", "--filter", "network=none")
	containerOut = strings.TrimSpace(string(out))

	lines = strings.Split(containerOut, "\n")

	// skip header
	lines = lines[1:]

	//ps output should have both the containers
	c.Assert(lines, checker.HasLen, 2)

	// Making sure onbridgenetwork and onnonenetwork is on the output
	c.Assert(containerOut, checker.Contains, "onnonenetwork", check.Commentf("Missing the container on none network\n"))
	c.Assert(containerOut, checker.Contains, "onbridgenetwork", check.Commentf("Missing the container on bridge network\n"))

	nwID, _ := dockerCmd(c, "network", "inspect", "--format", "{{.ID}}", "bridge")

	// Filter by network ID
	out, _ = dockerCmd(c, "ps", "--filter", "network="+nwID)
	containerOut = strings.TrimSpace(string(out))

	c.Assert(containerOut, checker.Contains, "onbridgenetwork")

	// Filter by partial network ID
	partialnwID := string(nwID[0:4])

	out, _ = dockerCmd(c, "ps", "--filter", "network="+partialnwID)
	containerOut = strings.TrimSpace(string(out))

	lines = strings.Split(containerOut, "\n")
	// skip header
	lines = lines[1:]

	// ps output should have only one container
	c.Assert(lines, checker.HasLen, 1)

	// Making sure onbridgenetwork is on the output
	c.Assert(containerOut, checker.Contains, "onbridgenetwork", check.Commentf("Missing the container on network\n"))

}

func (s *DockerSuite) TestPsByOrder(c *check.C) {
	name1 := "xyz-abc"
	out := runSleepingContainer(c, "--name", name1)
	container1 := strings.TrimSpace(out)

	name2 := "xyz-123"
	out = runSleepingContainer(c, "--name", name2)
	container2 := strings.TrimSpace(out)

	name3 := "789-abc"
	out = runSleepingContainer(c, "--name", name3)

	name4 := "789-123"
	out = runSleepingContainer(c, "--name", name4)

	// Run multiple time should have the same result
	out = cli.DockerCmd(c, "ps", "--no-trunc", "-q", "-f", "name=xyz").Combined()
	c.Assert(strings.TrimSpace(out), checker.Equals, fmt.Sprintf("%s\n%s", container2, container1))

	// Run multiple time should have the same result
	out = cli.DockerCmd(c, "ps", "--no-trunc", "-q", "-f", "name=xyz").Combined()
	c.Assert(strings.TrimSpace(out), checker.Equals, fmt.Sprintf("%s\n%s", container2, container1))
}

func (s *DockerSuite) TestPsFilterMissingArgErrorCode(c *check.C) {
	_, errCode, _ := dockerCmdWithError("ps", "--filter")
	c.Assert(errCode, checker.Equals, 125)
}

// Test case for 30291
func (s *DockerSuite) TestPsFormatTemplateWithArg(c *check.C) {
	runSleepingContainer(c, "-d", "--name", "top", "--label", "some.label=label.foo-bar")
	out, _ := dockerCmd(c, "ps", "--format", `{{.Names}} {{.Label "some.label"}}`)
	c.Assert(strings.TrimSpace(out), checker.Equals, "top label.foo-bar")
}

func (s *DockerSuite) TestPsListContainersFilterPorts(c *check.C) {
	testRequires(c, DaemonIsLinux)

	out, _ := dockerCmd(c, "run", "-d", "--publish=80", "busybox", "top")
	id1 := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "run", "-d", "--expose=8080", "busybox", "top")
	id2 := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "ps", "--no-trunc", "-q")
	c.Assert(strings.TrimSpace(out), checker.Contains, id1)
	c.Assert(strings.TrimSpace(out), checker.Contains, id2)

	out, _ = dockerCmd(c, "ps", "--no-trunc", "-q", "--filter", "publish=80-8080/udp")
	c.Assert(strings.TrimSpace(out), checker.Not(checker.Equals), id1)
	c.Assert(strings.TrimSpace(out), checker.Not(checker.Equals), id2)

	out, _ = dockerCmd(c, "ps", "--no-trunc", "-q", "--filter", "expose=8081")
	c.Assert(strings.TrimSpace(out), checker.Not(checker.Equals), id1)
	c.Assert(strings.TrimSpace(out), checker.Not(checker.Equals), id2)

	out, _ = dockerCmd(c, "ps", "--no-trunc", "-q", "--filter", "publish=80-81")
	c.Assert(strings.TrimSpace(out), checker.Equals, id1)
	c.Assert(strings.TrimSpace(out), checker.Not(checker.Equals), id2)

	out, _ = dockerCmd(c, "ps", "--no-trunc", "-q", "--filter", "expose=80/tcp")
	c.Assert(strings.TrimSpace(out), checker.Equals, id1)
	c.Assert(strings.TrimSpace(out), checker.Not(checker.Equals), id2)

	out, _ = dockerCmd(c, "ps", "--no-trunc", "-q", "--filter", "expose=8080/tcp")
	c.Assert(strings.TrimSpace(out), checker.Not(checker.Equals), id1)
	c.Assert(strings.TrimSpace(out), checker.Equals, id2)
}
