package main

import (
	"fmt"
	"strings"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/go-check/check"
)

// search for repos named  "registry" on the central registry
func (s *DockerSuite) TestSearchOnCentralRegistry(c *check.C) {
	testRequires(c, Network, DaemonIsLinux)

	out, _ := dockerCmd(c, "search", "busybox")
	c.Assert(out, checker.Contains, "Busybox base image.", check.Commentf("couldn't find any repository named (or containing) 'Busybox base image.'"))
}

func (s *DockerSuite) TestSearchStarsOptionWithWrongParameter(c *check.C) {
	out, _, err := dockerCmdWithError("search", "--filter", "stars=a", "busybox")
	c.Assert(err, check.NotNil, check.Commentf(out))
	c.Assert(out, checker.Contains, "Invalid filter", check.Commentf("couldn't find the invalid filter warning"))

	out, _, err = dockerCmdWithError("search", "-f", "stars=a", "busybox")
	c.Assert(err, check.NotNil, check.Commentf(out))
	c.Assert(out, checker.Contains, "Invalid filter", check.Commentf("couldn't find the invalid filter warning"))

	out, _, err = dockerCmdWithError("search", "-f", "is-automated=a", "busybox")
	c.Assert(err, check.NotNil, check.Commentf(out))
	c.Assert(out, checker.Contains, "Invalid filter", check.Commentf("couldn't find the invalid filter warning"))

	out, _, err = dockerCmdWithError("search", "-f", "is-official=a", "busybox")
	c.Assert(err, check.NotNil, check.Commentf(out))
	c.Assert(out, checker.Contains, "Invalid filter", check.Commentf("couldn't find the invalid filter warning"))

	// -s --stars deprecated since Docker 1.13
	out, _, err = dockerCmdWithError("search", "--stars=a", "busybox")
	c.Assert(err, check.NotNil, check.Commentf(out))
	c.Assert(out, checker.Contains, "invalid syntax", check.Commentf("couldn't find the invalid value warning"))

	// -s --stars deprecated since Docker 1.13
	out, _, err = dockerCmdWithError("search", "-s=-1", "busybox")
	c.Assert(err, check.NotNil, check.Commentf(out))
	c.Assert(out, checker.Contains, "invalid syntax", check.Commentf("couldn't find the invalid value warning"))
}

func (s *DockerSuite) TestSearchCmdOptions(c *check.C) {
	testRequires(c, Network, DaemonIsLinux)

	out, _ := dockerCmd(c, "search", "--help")
	c.Assert(out, checker.Contains, "Usage:\tdocker search [OPTIONS] TERM")

	outSearchCmd, _ := dockerCmd(c, "search", "busybox")
	outSearchCmdNotrunc, _ := dockerCmd(c, "search", "--no-trunc=true", "busybox")

	c.Assert(len(outSearchCmd) > len(outSearchCmdNotrunc), check.Equals, false, check.Commentf("The no-trunc option can't take effect."))

	outSearchCmdautomated, _ := dockerCmd(c, "search", "--filter", "is-automated=true", "busybox") //The busybox is a busybox base image, not an AUTOMATED image.
	outSearchCmdautomatedSlice := strings.Split(outSearchCmdautomated, "\n")
	for i := range outSearchCmdautomatedSlice {
		c.Assert(strings.HasPrefix(outSearchCmdautomatedSlice[i], "busybox "), check.Equals, false, check.Commentf("The busybox is not an AUTOMATED image: %s", outSearchCmdautomated))
	}

	outSearchCmdNotOfficial, _ := dockerCmd(c, "search", "--filter", "is-official=false", "busybox") //The busybox is a busybox base image, official image.
	outSearchCmdNotOfficialSlice := strings.Split(outSearchCmdNotOfficial, "\n")
	for i := range outSearchCmdNotOfficialSlice {
		c.Assert(strings.HasPrefix(outSearchCmdNotOfficialSlice[i], "busybox "), check.Equals, false, check.Commentf("The busybox is not an OFFICIAL image: %s", outSearchCmdNotOfficial))
	}

	outSearchCmdOfficial, _ := dockerCmd(c, "search", "--filter", "is-official=true", "busybox") //The busybox is a busybox base image, official image.
	outSearchCmdOfficialSlice := strings.Split(outSearchCmdOfficial, "\n")
	c.Assert(outSearchCmdOfficialSlice, checker.HasLen, 3) // 1 header, 1 line, 1 carriage return
	c.Assert(strings.HasPrefix(outSearchCmdOfficialSlice[1], "busybox "), check.Equals, true, check.Commentf("The busybox is an OFFICIAL image: %s", outSearchCmdNotOfficial))

	outSearchCmdStars, _ := dockerCmd(c, "search", "--filter", "stars=2", "busybox")
	c.Assert(strings.Count(outSearchCmdStars, "[OK]") > strings.Count(outSearchCmd, "[OK]"), check.Equals, false, check.Commentf("The quantity of images with stars should be less than that of all images: %s", outSearchCmdStars))

	dockerCmd(c, "search", "--filter", "is-automated=true", "--filter", "stars=2", "--no-trunc=true", "busybox")

	// --automated deprecated since Docker 1.13
	outSearchCmdautomated1, _ := dockerCmd(c, "search", "--automated=true", "busybox") //The busybox is a busybox base image, not an AUTOMATED image.
	outSearchCmdautomatedSlice1 := strings.Split(outSearchCmdautomated1, "\n")
	for i := range outSearchCmdautomatedSlice1 {
		c.Assert(strings.HasPrefix(outSearchCmdautomatedSlice1[i], "busybox "), check.Equals, false, check.Commentf("The busybox is not an AUTOMATED image: %s", outSearchCmdautomated))
	}

	// -s --stars deprecated since Docker 1.13
	outSearchCmdStars1, _ := dockerCmd(c, "search", "--stars=2", "busybox")
	c.Assert(strings.Count(outSearchCmdStars1, "[OK]") > strings.Count(outSearchCmd, "[OK]"), check.Equals, false, check.Commentf("The quantity of images with stars should be less than that of all images: %s", outSearchCmdStars1))

	// -s --stars deprecated since Docker 1.13
	dockerCmd(c, "search", "--stars=2", "--automated=true", "--no-trunc=true", "busybox")
}

// search for repos which start with "ubuntu-" on the central registry
func (s *DockerSuite) TestSearchOnCentralRegistryWithDash(c *check.C) {
	testRequires(c, Network, DaemonIsLinux)

	dockerCmd(c, "search", "ubuntu-")
}

// test case for #23055
func (s *DockerSuite) TestSearchWithLimit(c *check.C) {
	testRequires(c, Network, DaemonIsLinux)

	limit := 10
	out, _, err := dockerCmdWithError("search", fmt.Sprintf("--limit=%d", limit), "docker")
	c.Assert(err, checker.IsNil)
	outSlice := strings.Split(out, "\n")
	c.Assert(outSlice, checker.HasLen, limit+2) // 1 header, 1 carriage return

	limit = 50
	out, _, err = dockerCmdWithError("search", fmt.Sprintf("--limit=%d", limit), "docker")
	c.Assert(err, checker.IsNil)
	outSlice = strings.Split(out, "\n")
	c.Assert(outSlice, checker.HasLen, limit+2) // 1 header, 1 carriage return

	limit = 100
	out, _, err = dockerCmdWithError("search", fmt.Sprintf("--limit=%d", limit), "docker")
	c.Assert(err, checker.IsNil)
	outSlice = strings.Split(out, "\n")
	c.Assert(outSlice, checker.HasLen, limit+2) // 1 header, 1 carriage return

	limit = 0
	_, _, err = dockerCmdWithError("search", fmt.Sprintf("--limit=%d", limit), "docker")
	c.Assert(err, checker.Not(checker.IsNil))

	limit = 200
	_, _, err = dockerCmdWithError("search", fmt.Sprintf("--limit=%d", limit), "docker")
	c.Assert(err, checker.Not(checker.IsNil))
}
