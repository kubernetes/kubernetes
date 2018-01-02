package main

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/pkg/testutil"
	"github.com/docker/docker/runconfig"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestLinksPingUnlinkedContainers(c *check.C) {
	testRequires(c, DaemonIsLinux)
	_, exitCode, err := dockerCmdWithError("run", "--rm", "busybox", "sh", "-c", "ping -c 1 alias1 -W 1 && ping -c 1 alias2 -W 1")

	// run ping failed with error
	c.Assert(exitCode, checker.Equals, 1, check.Commentf("error: %v", err))
}

// Test for appropriate error when calling --link with an invalid target container
func (s *DockerSuite) TestLinksInvalidContainerTarget(c *check.C) {
	testRequires(c, DaemonIsLinux)
	out, _, err := dockerCmdWithError("run", "--link", "bogus:alias", "busybox", "true")

	// an invalid container target should produce an error
	c.Assert(err, checker.NotNil, check.Commentf("out: %s", out))
	// an invalid container target should produce an error
	c.Assert(out, checker.Contains, "Could not get container")
}

func (s *DockerSuite) TestLinksPingLinkedContainers(c *check.C) {
	testRequires(c, DaemonIsLinux)
	// Test with the three different ways of specifying the default network on Linux
	testLinkPingOnNetwork(c, "")
	testLinkPingOnNetwork(c, "default")
	testLinkPingOnNetwork(c, "bridge")
}

func testLinkPingOnNetwork(c *check.C, network string) {
	var postArgs []string
	if network != "" {
		postArgs = append(postArgs, []string{"--net", network}...)
	}
	postArgs = append(postArgs, []string{"busybox", "top"}...)
	runArgs1 := append([]string{"run", "-d", "--name", "container1", "--hostname", "fred"}, postArgs...)
	runArgs2 := append([]string{"run", "-d", "--name", "container2", "--hostname", "wilma"}, postArgs...)

	// Run the two named containers
	dockerCmd(c, runArgs1...)
	dockerCmd(c, runArgs2...)

	postArgs = []string{}
	if network != "" {
		postArgs = append(postArgs, []string{"--net", network}...)
	}
	postArgs = append(postArgs, []string{"busybox", "sh", "-c"}...)

	// Format a run for a container which links to the other two
	runArgs := append([]string{"run", "--rm", "--link", "container1:alias1", "--link", "container2:alias2"}, postArgs...)
	pingCmd := "ping -c 1 %s -W 1 && ping -c 1 %s -W 1"

	// test ping by alias, ping by name, and ping by hostname
	// 1. Ping by alias
	dockerCmd(c, append(runArgs, fmt.Sprintf(pingCmd, "alias1", "alias2"))...)
	// 2. Ping by container name
	dockerCmd(c, append(runArgs, fmt.Sprintf(pingCmd, "container1", "container2"))...)
	// 3. Ping by hostname
	dockerCmd(c, append(runArgs, fmt.Sprintf(pingCmd, "fred", "wilma"))...)

	// Clean for next round
	dockerCmd(c, "rm", "-f", "container1")
	dockerCmd(c, "rm", "-f", "container2")
}

func (s *DockerSuite) TestLinksPingLinkedContainersAfterRename(c *check.C) {
	testRequires(c, DaemonIsLinux)
	out, _ := dockerCmd(c, "run", "-d", "--name", "container1", "busybox", "top")
	idA := strings.TrimSpace(out)
	out, _ = dockerCmd(c, "run", "-d", "--name", "container2", "busybox", "top")
	idB := strings.TrimSpace(out)
	dockerCmd(c, "rename", "container1", "container_new")
	dockerCmd(c, "run", "--rm", "--link", "container_new:alias1", "--link", "container2:alias2", "busybox", "sh", "-c", "ping -c 1 alias1 -W 1 && ping -c 1 alias2 -W 1")
	dockerCmd(c, "kill", idA)
	dockerCmd(c, "kill", idB)

}

func (s *DockerSuite) TestLinksInspectLinksStarted(c *check.C) {
	testRequires(c, DaemonIsLinux)
	var (
		expected = map[string]struct{}{"/container1:/testinspectlink/alias1": {}, "/container2:/testinspectlink/alias2": {}}
		result   []string
	)
	dockerCmd(c, "run", "-d", "--name", "container1", "busybox", "top")
	dockerCmd(c, "run", "-d", "--name", "container2", "busybox", "top")
	dockerCmd(c, "run", "-d", "--name", "testinspectlink", "--link", "container1:alias1", "--link", "container2:alias2", "busybox", "top")
	links := inspectFieldJSON(c, "testinspectlink", "HostConfig.Links")

	err := json.Unmarshal([]byte(links), &result)
	c.Assert(err, checker.IsNil)

	output := testutil.ConvertSliceOfStringsToMap(result)

	c.Assert(output, checker.DeepEquals, expected)
}

func (s *DockerSuite) TestLinksInspectLinksStopped(c *check.C) {
	testRequires(c, DaemonIsLinux)
	var (
		expected = map[string]struct{}{"/container1:/testinspectlink/alias1": {}, "/container2:/testinspectlink/alias2": {}}
		result   []string
	)
	dockerCmd(c, "run", "-d", "--name", "container1", "busybox", "top")
	dockerCmd(c, "run", "-d", "--name", "container2", "busybox", "top")
	dockerCmd(c, "run", "-d", "--name", "testinspectlink", "--link", "container1:alias1", "--link", "container2:alias2", "busybox", "true")
	links := inspectFieldJSON(c, "testinspectlink", "HostConfig.Links")

	err := json.Unmarshal([]byte(links), &result)
	c.Assert(err, checker.IsNil)

	output := testutil.ConvertSliceOfStringsToMap(result)

	c.Assert(output, checker.DeepEquals, expected)
}

func (s *DockerSuite) TestLinksNotStartedParentNotFail(c *check.C) {
	testRequires(c, DaemonIsLinux)
	dockerCmd(c, "create", "--name=first", "busybox", "top")
	dockerCmd(c, "create", "--name=second", "--link=first:first", "busybox", "top")
	dockerCmd(c, "start", "first")

}

func (s *DockerSuite) TestLinksHostsFilesInject(c *check.C) {
	testRequires(c, DaemonIsLinux)
	testRequires(c, SameHostDaemon, ExecSupport)

	out, _ := dockerCmd(c, "run", "-itd", "--name", "one", "busybox", "top")
	idOne := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "run", "-itd", "--name", "two", "--link", "one:onetwo", "busybox", "top")
	idTwo := strings.TrimSpace(out)

	c.Assert(waitRun(idTwo), checker.IsNil)

	readContainerFileWithExec(c, idOne, "/etc/hosts")
	contentTwo := readContainerFileWithExec(c, idTwo, "/etc/hosts")
	// Host is not present in updated hosts file
	c.Assert(string(contentTwo), checker.Contains, "onetwo")
}

func (s *DockerSuite) TestLinksUpdateOnRestart(c *check.C) {
	testRequires(c, DaemonIsLinux)
	testRequires(c, SameHostDaemon, ExecSupport)
	dockerCmd(c, "run", "-d", "--name", "one", "busybox", "top")
	out, _ := dockerCmd(c, "run", "-d", "--name", "two", "--link", "one:onetwo", "--link", "one:one", "busybox", "top")
	id := strings.TrimSpace(string(out))

	realIP := inspectField(c, "one", "NetworkSettings.Networks.bridge.IPAddress")
	content := readContainerFileWithExec(c, id, "/etc/hosts")

	getIP := func(hosts []byte, hostname string) string {
		re := regexp.MustCompile(fmt.Sprintf(`(\S*)\t%s`, regexp.QuoteMeta(hostname)))
		matches := re.FindSubmatch(hosts)
		c.Assert(matches, checker.NotNil, check.Commentf("Hostname %s have no matches in hosts", hostname))
		return string(matches[1])
	}
	ip := getIP(content, "one")
	c.Assert(ip, checker.Equals, realIP)

	ip = getIP(content, "onetwo")
	c.Assert(ip, checker.Equals, realIP)

	dockerCmd(c, "restart", "one")
	realIP = inspectField(c, "one", "NetworkSettings.Networks.bridge.IPAddress")

	content = readContainerFileWithExec(c, id, "/etc/hosts")
	ip = getIP(content, "one")
	c.Assert(ip, checker.Equals, realIP)

	ip = getIP(content, "onetwo")
	c.Assert(ip, checker.Equals, realIP)
}

func (s *DockerSuite) TestLinksEnvs(c *check.C) {
	testRequires(c, DaemonIsLinux)
	dockerCmd(c, "run", "-d", "-e", "e1=", "-e", "e2=v2", "-e", "e3=v3=v3", "--name=first", "busybox", "top")
	out, _ := dockerCmd(c, "run", "--name=second", "--link=first:first", "busybox", "env")
	c.Assert(out, checker.Contains, "FIRST_ENV_e1=\n")
	c.Assert(out, checker.Contains, "FIRST_ENV_e2=v2")
	c.Assert(out, checker.Contains, "FIRST_ENV_e3=v3=v3")
}

func (s *DockerSuite) TestLinkShortDefinition(c *check.C) {
	testRequires(c, DaemonIsLinux)
	out, _ := dockerCmd(c, "run", "-d", "--name", "shortlinkdef", "busybox", "top")

	cid := strings.TrimSpace(out)
	c.Assert(waitRun(cid), checker.IsNil)

	out, _ = dockerCmd(c, "run", "-d", "--name", "link2", "--link", "shortlinkdef", "busybox", "top")

	cid2 := strings.TrimSpace(out)
	c.Assert(waitRun(cid2), checker.IsNil)

	links := inspectFieldJSON(c, cid2, "HostConfig.Links")
	c.Assert(links, checker.Equals, "[\"/shortlinkdef:/link2/shortlinkdef\"]")
}

func (s *DockerSuite) TestLinksNetworkHostContainer(c *check.C) {
	testRequires(c, DaemonIsLinux, NotUserNamespace)
	dockerCmd(c, "run", "-d", "--net", "host", "--name", "host_container", "busybox", "top")
	out, _, err := dockerCmdWithError("run", "--name", "should_fail", "--link", "host_container:tester", "busybox", "true")

	// Running container linking to a container with --net host should have failed
	c.Assert(err, checker.NotNil, check.Commentf("out: %s", out))
	// Running container linking to a container with --net host should have failed
	c.Assert(out, checker.Contains, runconfig.ErrConflictHostNetworkAndLinks.Error())
}

func (s *DockerSuite) TestLinksEtcHostsRegularFile(c *check.C) {
	testRequires(c, DaemonIsLinux, NotUserNamespace)
	out, _ := dockerCmd(c, "run", "--net=host", "busybox", "ls", "-la", "/etc/hosts")
	// /etc/hosts should be a regular file
	c.Assert(out, checker.Matches, "^-.+\n")
}

func (s *DockerSuite) TestLinksMultipleWithSameName(c *check.C) {
	testRequires(c, DaemonIsLinux)
	dockerCmd(c, "run", "-d", "--name=upstream-a", "busybox", "top")
	dockerCmd(c, "run", "-d", "--name=upstream-b", "busybox", "top")
	dockerCmd(c, "run", "--link", "upstream-a:upstream", "--link", "upstream-b:upstream", "busybox", "sh", "-c", "ping -c 1 upstream")
}
