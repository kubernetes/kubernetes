package main

import (
	"fmt"
	"github.com/go-check/check"
	"reflect"
	"regexp"
	"strings"
	"time"
)

func (s *DockerSuite) TestLinksPingUnlinkedContainers(c *check.C) {

	_, exitCode, err := dockerCmdWithError(c, "run", "--rm", "busybox", "sh", "-c", "ping -c 1 alias1 -W 1 && ping -c 1 alias2 -W 1")

	if exitCode == 0 {
		c.Fatal("run ping did not fail")
	} else if exitCode != 1 {
		c.Fatalf("run ping failed with errors: %v", err)
	}

}

// Test for appropriate error when calling --link with an invalid target container
func (s *DockerSuite) TestLinksInvalidContainerTarget(c *check.C) {

	out, _, err := dockerCmdWithError(c, "run", "--link", "bogus:alias", "busybox", "true")

	if err == nil {
		c.Fatal("an invalid container target should produce an error")
	}
	if !strings.Contains(out, "Could not get container") {
		c.Fatalf("error output expected 'Could not get container', but got %q instead; err: %v", out, err)
	}

}

func (s *DockerSuite) TestLinksPingLinkedContainers(c *check.C) {

	dockerCmd(c, "run", "-d", "--name", "container1", "--hostname", "fred", "busybox", "top")
	dockerCmd(c, "run", "-d", "--name", "container2", "--hostname", "wilma", "busybox", "top")

	runArgs := []string{"run", "--rm", "--link", "container1:alias1", "--link", "container2:alias2", "busybox", "sh", "-c"}
	pingCmd := "ping -c 1 %s -W 1 && ping -c 1 %s -W 1"

	// test ping by alias, ping by name, and ping by hostname
	// 1. Ping by alias
	dockerCmd(c, append(runArgs, fmt.Sprintf(pingCmd, "alias1", "alias2"))...)
	// 2. Ping by container name
	dockerCmd(c, append(runArgs, fmt.Sprintf(pingCmd, "container1", "container2"))...)
	// 3. Ping by hostname
	dockerCmd(c, append(runArgs, fmt.Sprintf(pingCmd, "fred", "wilma"))...)

}

func (s *DockerSuite) TestLinksPingLinkedContainersAfterRename(c *check.C) {

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
	var (
		expected = map[string]struct{}{"/container1:/testinspectlink/alias1": {}, "/container2:/testinspectlink/alias2": {}}
		result   []string
	)
	dockerCmd(c, "run", "-d", "--name", "container1", "busybox", "top")
	dockerCmd(c, "run", "-d", "--name", "container2", "busybox", "top")
	dockerCmd(c, "run", "-d", "--name", "testinspectlink", "--link", "container1:alias1", "--link", "container2:alias2", "busybox", "top")
	links, err := inspectFieldJSON("testinspectlink", "HostConfig.Links")
	if err != nil {
		c.Fatal(err)
	}

	err = unmarshalJSON([]byte(links), &result)
	if err != nil {
		c.Fatal(err)
	}

	output := convertSliceOfStringsToMap(result)

	equal := reflect.DeepEqual(output, expected)

	if !equal {
		c.Fatalf("Links %s, expected %s", result, expected)
	}
}

func (s *DockerSuite) TestLinksInspectLinksStopped(c *check.C) {
	var (
		expected = map[string]struct{}{"/container1:/testinspectlink/alias1": {}, "/container2:/testinspectlink/alias2": {}}
		result   []string
	)
	dockerCmd(c, "run", "-d", "--name", "container1", "busybox", "top")
	dockerCmd(c, "run", "-d", "--name", "container2", "busybox", "top")
	dockerCmd(c, "run", "-d", "--name", "testinspectlink", "--link", "container1:alias1", "--link", "container2:alias2", "busybox", "true")
	links, err := inspectFieldJSON("testinspectlink", "HostConfig.Links")
	if err != nil {
		c.Fatal(err)
	}

	err = unmarshalJSON([]byte(links), &result)
	if err != nil {
		c.Fatal(err)
	}

	output := convertSliceOfStringsToMap(result)

	equal := reflect.DeepEqual(output, expected)

	if !equal {
		c.Fatalf("Links %s, but expected %s", result, expected)
	}

}

func (s *DockerSuite) TestLinksNotStartedParentNotFail(c *check.C) {

	dockerCmd(c, "create", "--name=first", "busybox", "top")
	dockerCmd(c, "create", "--name=second", "--link=first:first", "busybox", "top")
	dockerCmd(c, "start", "first")

}

func (s *DockerSuite) TestLinksHostsFilesInject(c *check.C) {
	testRequires(c, SameHostDaemon, ExecSupport)

	out, _ := dockerCmd(c, "run", "-itd", "--name", "one", "busybox", "top")
	idOne := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "run", "-itd", "--name", "two", "--link", "one:onetwo", "busybox", "top")
	idTwo := strings.TrimSpace(out)

	time.Sleep(1 * time.Second)

	contentOne, err := readContainerFileWithExec(idOne, "/etc/hosts")
	if err != nil {
		c.Fatal(err, string(contentOne))
	}

	contentTwo, err := readContainerFileWithExec(idTwo, "/etc/hosts")
	if err != nil {
		c.Fatal(err, string(contentTwo))
	}

	if !strings.Contains(string(contentTwo), "onetwo") {
		c.Fatal("Host is not present in updated hosts file", string(contentTwo))
	}

}

func (s *DockerSuite) TestLinksUpdateOnRestart(c *check.C) {
	testRequires(c, SameHostDaemon, ExecSupport)
	dockerCmd(c, "run", "-d", "--name", "one", "busybox", "top")
	out, _ := dockerCmd(c, "run", "-d", "--name", "two", "--link", "one:onetwo", "--link", "one:one", "busybox", "top")
	id := strings.TrimSpace(string(out))

	realIP, err := inspectField("one", "NetworkSettings.IPAddress")
	if err != nil {
		c.Fatal(err)
	}
	content, err := readContainerFileWithExec(id, "/etc/hosts")
	if err != nil {
		c.Fatal(err, string(content))
	}
	getIP := func(hosts []byte, hostname string) string {
		re := regexp.MustCompile(fmt.Sprintf(`(\S*)\t%s`, regexp.QuoteMeta(hostname)))
		matches := re.FindSubmatch(hosts)
		if matches == nil {
			c.Fatalf("Hostname %s have no matches in hosts", hostname)
		}
		return string(matches[1])
	}
	if ip := getIP(content, "one"); ip != realIP {
		c.Fatalf("For 'one' alias expected IP: %s, got: %s", realIP, ip)
	}
	if ip := getIP(content, "onetwo"); ip != realIP {
		c.Fatalf("For 'onetwo' alias expected IP: %s, got: %s", realIP, ip)
	}
	dockerCmd(c, "restart", "one")
	realIP, err = inspectField("one", "NetworkSettings.IPAddress")
	if err != nil {
		c.Fatal(err)
	}
	content, err = readContainerFileWithExec(id, "/etc/hosts")
	if err != nil {
		c.Fatal(err, string(content))
	}
	if ip := getIP(content, "one"); ip != realIP {
		c.Fatalf("For 'one' alias expected IP: %s, got: %s", realIP, ip)
	}
	if ip := getIP(content, "onetwo"); ip != realIP {
		c.Fatalf("For 'onetwo' alias expected IP: %s, got: %s", realIP, ip)
	}
}

func (s *DockerSuite) TestLinksEnvs(c *check.C) {
	dockerCmd(c, "run", "-d", "-e", "e1=", "-e", "e2=v2", "-e", "e3=v3=v3", "--name=first", "busybox", "top")
	out, _ := dockerCmd(c, "run", "--name=second", "--link=first:first", "busybox", "env")
	if !strings.Contains(out, "FIRST_ENV_e1=\n") ||
		!strings.Contains(out, "FIRST_ENV_e2=v2") ||
		!strings.Contains(out, "FIRST_ENV_e3=v3=v3") {
		c.Fatalf("Incorrect output: %s", out)
	}
}

func (s *DockerSuite) TestLinkShortDefinition(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "--name", "shortlinkdef", "busybox", "top")

	cid := strings.TrimSpace(out)
	c.Assert(waitRun(cid), check.IsNil)

	out, _ = dockerCmd(c, "run", "-d", "--name", "link2", "--link", "shortlinkdef", "busybox", "top")

	cid2 := strings.TrimSpace(out)
	c.Assert(waitRun(cid2), check.IsNil)

	links, err := inspectFieldJSON(cid2, "HostConfig.Links")
	c.Assert(err, check.IsNil)
	c.Assert(links, check.Equals, "[\"/shortlinkdef:/link2/shortlinkdef\"]")
}
