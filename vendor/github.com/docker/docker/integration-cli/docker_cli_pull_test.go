package main

import (
	"fmt"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/go-check/check"
	"github.com/opencontainers/go-digest"
)

// TestPullFromCentralRegistry pulls an image from the central registry and verifies that the client
// prints all expected output.
func (s *DockerHubPullSuite) TestPullFromCentralRegistry(c *check.C) {
	testRequires(c, DaemonIsLinux)
	out := s.Cmd(c, "pull", "hello-world")
	defer deleteImages("hello-world")

	c.Assert(out, checker.Contains, "Using default tag: latest", check.Commentf("expected the 'latest' tag to be automatically assumed"))
	c.Assert(out, checker.Contains, "Pulling from library/hello-world", check.Commentf("expected the 'library/' prefix to be automatically assumed"))
	c.Assert(out, checker.Contains, "Downloaded newer image for hello-world:latest")

	matches := regexp.MustCompile(`Digest: (.+)\n`).FindAllStringSubmatch(out, -1)
	c.Assert(len(matches), checker.Equals, 1, check.Commentf("expected exactly one image digest in the output"))
	c.Assert(len(matches[0]), checker.Equals, 2, check.Commentf("unexpected number of submatches for the digest"))
	_, err := digest.Parse(matches[0][1])
	c.Check(err, checker.IsNil, check.Commentf("invalid digest %q in output", matches[0][1]))

	// We should have a single entry in images.
	img := strings.TrimSpace(s.Cmd(c, "images"))
	splitImg := strings.Split(img, "\n")
	c.Assert(splitImg, checker.HasLen, 2)
	c.Assert(splitImg[1], checker.Matches, `hello-world\s+latest.*?`, check.Commentf("invalid output for `docker images` (expected image and tag name"))
}

// TestPullNonExistingImage pulls non-existing images from the central registry, with different
// combinations of implicit tag and library prefix.
func (s *DockerHubPullSuite) TestPullNonExistingImage(c *check.C) {
	testRequires(c, DaemonIsLinux)

	type entry struct {
		repo  string
		alias string
		tag   string
	}

	entries := []entry{
		{"asdfasdf", "asdfasdf", "foobar"},
		{"asdfasdf", "library/asdfasdf", "foobar"},
		{"asdfasdf", "asdfasdf", ""},
		{"asdfasdf", "asdfasdf", "latest"},
		{"asdfasdf", "library/asdfasdf", ""},
		{"asdfasdf", "library/asdfasdf", "latest"},
	}

	// The option field indicates "-a" or not.
	type record struct {
		e      entry
		option string
		out    string
		err    error
	}

	// Execute 'docker pull' in parallel, pass results (out, err) and
	// necessary information ("-a" or not, and the image name) to channel.
	var group sync.WaitGroup
	recordChan := make(chan record, len(entries)*2)
	for _, e := range entries {
		group.Add(1)
		go func(e entry) {
			defer group.Done()
			repoName := e.alias
			if e.tag != "" {
				repoName += ":" + e.tag
			}
			out, err := s.CmdWithError("pull", repoName)
			recordChan <- record{e, "", out, err}
		}(e)
		if e.tag == "" {
			// pull -a on a nonexistent registry should fall back as well
			group.Add(1)
			go func(e entry) {
				defer group.Done()
				out, err := s.CmdWithError("pull", "-a", e.alias)
				recordChan <- record{e, "-a", out, err}
			}(e)
		}
	}

	// Wait for completion
	group.Wait()
	close(recordChan)

	// Process the results (out, err).
	for record := range recordChan {
		if len(record.option) == 0 {
			c.Assert(record.err, checker.NotNil, check.Commentf("expected non-zero exit status when pulling non-existing image: %s", record.out))
			c.Assert(record.out, checker.Contains, fmt.Sprintf("pull access denied for %s, repository does not exist or may require 'docker login'", record.e.repo), check.Commentf("expected image not found error messages"))
		} else {
			// pull -a on a nonexistent registry should fall back as well
			c.Assert(record.err, checker.NotNil, check.Commentf("expected non-zero exit status when pulling non-existing image: %s", record.out))
			c.Assert(record.out, checker.Contains, fmt.Sprintf("pull access denied for %s, repository does not exist or may require 'docker login'", record.e.repo), check.Commentf("expected image not found error messages"))
			c.Assert(record.out, checker.Not(checker.Contains), "unauthorized", check.Commentf(`message should not contain "unauthorized"`))
		}
	}

}

// TestPullFromCentralRegistryImplicitRefParts pulls an image from the central registry and verifies
// that pulling the same image with different combinations of implicit elements of the image
// reference (tag, repository, central registry url, ...) doesn't trigger a new pull nor leads to
// multiple images.
func (s *DockerHubPullSuite) TestPullFromCentralRegistryImplicitRefParts(c *check.C) {
	testRequires(c, DaemonIsLinux)

	// Pull hello-world from v2
	pullFromV2 := func(ref string) (int, string) {
		out := s.Cmd(c, "pull", "hello-world")
		v1Retries := 0
		for strings.Contains(out, "this image was pulled from a legacy registry") {
			// Some network errors may cause fallbacks to the v1
			// protocol, which would violate the test's assumption
			// that it will get the same images. To make the test
			// more robust against these network glitches, allow a
			// few retries if we end up with a v1 pull.

			if v1Retries > 2 {
				c.Fatalf("too many v1 fallback incidents when pulling %s", ref)
			}

			s.Cmd(c, "rmi", ref)
			out = s.Cmd(c, "pull", ref)

			v1Retries++
		}

		return v1Retries, out
	}

	pullFromV2("hello-world")
	defer deleteImages("hello-world")

	s.Cmd(c, "tag", "hello-world", "hello-world-backup")

	for _, ref := range []string{
		"hello-world",
		"hello-world:latest",
		"library/hello-world",
		"library/hello-world:latest",
		"docker.io/library/hello-world",
		"index.docker.io/library/hello-world",
	} {
		var out string
		for {
			var v1Retries int
			v1Retries, out = pullFromV2(ref)

			// Keep repeating the test case until we don't hit a v1
			// fallback case. We won't get the right "Image is up
			// to date" message if the local image was replaced
			// with one pulled from v1.
			if v1Retries == 0 {
				break
			}
			s.Cmd(c, "rmi", ref)
			s.Cmd(c, "tag", "hello-world-backup", "hello-world")
		}
		c.Assert(out, checker.Contains, "Image is up to date for hello-world:latest")
	}

	s.Cmd(c, "rmi", "hello-world-backup")

	// We should have a single entry in images.
	img := strings.TrimSpace(s.Cmd(c, "images"))
	splitImg := strings.Split(img, "\n")
	c.Assert(splitImg, checker.HasLen, 2)
	c.Assert(splitImg[1], checker.Matches, `hello-world\s+latest.*?`, check.Commentf("invalid output for `docker images` (expected image and tag name"))
}

// TestPullScratchNotAllowed verifies that pulling 'scratch' is rejected.
func (s *DockerHubPullSuite) TestPullScratchNotAllowed(c *check.C) {
	testRequires(c, DaemonIsLinux)
	out, err := s.CmdWithError("pull", "scratch")
	c.Assert(err, checker.NotNil, check.Commentf("expected pull of scratch to fail"))
	c.Assert(out, checker.Contains, "'scratch' is a reserved name")
	c.Assert(out, checker.Not(checker.Contains), "Pulling repository scratch")
}

// TestPullAllTagsFromCentralRegistry pulls using `all-tags` for a given image and verifies that it
// results in more images than a naked pull.
func (s *DockerHubPullSuite) TestPullAllTagsFromCentralRegistry(c *check.C) {
	testRequires(c, DaemonIsLinux)
	s.Cmd(c, "pull", "busybox")
	outImageCmd := s.Cmd(c, "images", "busybox")
	splitOutImageCmd := strings.Split(strings.TrimSpace(outImageCmd), "\n")
	c.Assert(splitOutImageCmd, checker.HasLen, 2)

	s.Cmd(c, "pull", "--all-tags=true", "busybox")
	outImageAllTagCmd := s.Cmd(c, "images", "busybox")
	linesCount := strings.Count(outImageAllTagCmd, "\n")
	c.Assert(linesCount, checker.GreaterThan, 2, check.Commentf("pulling all tags should provide more than two images, got %s", outImageAllTagCmd))

	// Verify that the line for 'busybox:latest' is left unchanged.
	var latestLine string
	for _, line := range strings.Split(outImageAllTagCmd, "\n") {
		if strings.HasPrefix(line, "busybox") && strings.Contains(line, "latest") {
			latestLine = line
			break
		}
	}
	c.Assert(latestLine, checker.Not(checker.Equals), "", check.Commentf("no entry for busybox:latest found after pulling all tags"))
	splitLatest := strings.Fields(latestLine)
	splitCurrent := strings.Fields(splitOutImageCmd[1])

	// Clear relative creation times, since these can easily change between
	// two invocations of "docker images". Without this, the test can fail
	// like this:
	// ... obtained []string = []string{"busybox", "latest", "d9551b4026f0", "27", "minutes", "ago", "1.113", "MB"}
	// ... expected []string = []string{"busybox", "latest", "d9551b4026f0", "26", "minutes", "ago", "1.113", "MB"}
	splitLatest[3] = ""
	splitLatest[4] = ""
	splitLatest[5] = ""
	splitCurrent[3] = ""
	splitCurrent[4] = ""
	splitCurrent[5] = ""

	c.Assert(splitLatest, checker.DeepEquals, splitCurrent, check.Commentf("busybox:latest was changed after pulling all tags"))
}

// TestPullClientDisconnect kills the client during a pull operation and verifies that the operation
// gets cancelled.
//
// Ref: docker/docker#15589
func (s *DockerHubPullSuite) TestPullClientDisconnect(c *check.C) {
	testRequires(c, DaemonIsLinux)
	repoName := "hello-world:latest"

	pullCmd := s.MakeCmd("pull", repoName)
	stdout, err := pullCmd.StdoutPipe()
	c.Assert(err, checker.IsNil)
	err = pullCmd.Start()
	c.Assert(err, checker.IsNil)

	// Cancel as soon as we get some output.
	buf := make([]byte, 10)
	_, err = stdout.Read(buf)
	c.Assert(err, checker.IsNil)

	err = pullCmd.Process.Kill()
	c.Assert(err, checker.IsNil)

	time.Sleep(2 * time.Second)
	_, err = s.CmdWithError("inspect", repoName)
	c.Assert(err, checker.NotNil, check.Commentf("image was pulled after client disconnected"))
}

func (s *DockerRegistryAuthHtpasswdSuite) TestPullNoCredentialsNotFound(c *check.C) {
	// @TODO TestPullNoCredentialsNotFound expects docker to fall back to a v1 registry, so has to be updated for v17.12, when v1 registries are no longer supported
	s.d.StartWithBusybox(c, "--disable-legacy-registry=false")

	// we don't care about the actual image, we just want to see image not found
	// because that means v2 call returned 401 and we fell back to v1 which usually
	// gives a 404 (in this case the test registry doesn't handle v1 at all)
	out, err := s.d.Cmd("pull", privateRegistryURL+"/busybox")
	c.Assert(err, check.NotNil, check.Commentf(out))
	c.Assert(out, checker.Contains, "Error: image busybox:latest not found")
}

// Regression test for https://github.com/docker/docker/issues/26429
func (s *DockerSuite) TestPullLinuxImageFailsOnWindows(c *check.C) {
	testRequires(c, DaemonIsWindows, Network)
	_, _, err := dockerCmdWithError("pull", "ubuntu")
	c.Assert(err.Error(), checker.Contains, "cannot be used on this platform")
}

// Regression test for https://github.com/docker/docker/issues/28892
func (s *DockerSuite) TestPullWindowsImageFailsOnLinux(c *check.C) {
	testRequires(c, DaemonIsLinux, Network)
	_, _, err := dockerCmdWithError("pull", "microsoft/nanoserver")
	c.Assert(err.Error(), checker.Contains, "cannot be used on this platform")
}
