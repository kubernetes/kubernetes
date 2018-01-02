package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/integration-cli/checker"
	icmd "github.com/docker/docker/pkg/testutil/cmd"
	"github.com/go-check/check"
)

func checkValidGraphDriver(c *check.C, name string) {
	if name != "devicemapper" && name != "overlay" && name != "vfs" && name != "zfs" && name != "btrfs" && name != "aufs" {
		c.Fatalf("%v is not a valid graph driver name", name)
	}
}

func (s *DockerSuite) TestInspectImage(c *check.C) {
	testRequires(c, DaemonIsLinux)
	imageTest := "emptyfs"
	// It is important that this ID remain stable. If a code change causes
	// it to be different, this is equivalent to a cache bust when pulling
	// a legacy-format manifest. If the check at the end of this function
	// fails, fix the difference in the image serialization instead of
	// updating this hash.
	imageTestID := "sha256:11f64303f0f7ffdc71f001788132bca5346831939a956e3e975c93267d89a16d"
	id := inspectField(c, imageTest, "Id")

	c.Assert(id, checker.Equals, imageTestID)
}

func (s *DockerSuite) TestInspectInt64(c *check.C) {
	dockerCmd(c, "run", "-d", "-m=300M", "--name", "inspectTest", "busybox", "true")
	inspectOut := inspectField(c, "inspectTest", "HostConfig.Memory")
	c.Assert(inspectOut, checker.Equals, "314572800")
}

func (s *DockerSuite) TestInspectDefault(c *check.C) {
	//Both the container and image are named busybox. docker inspect will fetch the container JSON.
	//If the container JSON is not available, it will go for the image JSON.

	out, _ := dockerCmd(c, "run", "--name=busybox", "-d", "busybox", "true")
	containerID := strings.TrimSpace(out)

	inspectOut := inspectField(c, "busybox", "Id")
	c.Assert(strings.TrimSpace(inspectOut), checker.Equals, containerID)
}

func (s *DockerSuite) TestInspectStatus(c *check.C) {
	out := runSleepingContainer(c, "-d")
	out = strings.TrimSpace(out)

	inspectOut := inspectField(c, out, "State.Status")
	c.Assert(inspectOut, checker.Equals, "running")

	// Windows does not support pause/unpause on Windows Server Containers.
	// (RS1 does for Hyper-V Containers, but production CI is not setup for that)
	if testEnv.DaemonPlatform() != "windows" {
		dockerCmd(c, "pause", out)
		inspectOut = inspectField(c, out, "State.Status")
		c.Assert(inspectOut, checker.Equals, "paused")

		dockerCmd(c, "unpause", out)
		inspectOut = inspectField(c, out, "State.Status")
		c.Assert(inspectOut, checker.Equals, "running")
	}

	dockerCmd(c, "stop", out)
	inspectOut = inspectField(c, out, "State.Status")
	c.Assert(inspectOut, checker.Equals, "exited")

}

func (s *DockerSuite) TestInspectTypeFlagContainer(c *check.C) {
	//Both the container and image are named busybox. docker inspect will fetch container
	//JSON State.Running field. If the field is true, it's a container.
	runSleepingContainer(c, "--name=busybox", "-d")

	formatStr := "--format={{.State.Running}}"
	out, _ := dockerCmd(c, "inspect", "--type=container", formatStr, "busybox")
	c.Assert(out, checker.Equals, "true\n") // not a container JSON
}

func (s *DockerSuite) TestInspectTypeFlagWithNoContainer(c *check.C) {
	//Run this test on an image named busybox. docker inspect will try to fetch container
	//JSON. Since there is no container named busybox and --type=container, docker inspect will
	//not try to get the image JSON. It will throw an error.

	dockerCmd(c, "run", "-d", "busybox", "true")

	_, _, err := dockerCmdWithError("inspect", "--type=container", "busybox")
	// docker inspect should fail, as there is no container named busybox
	c.Assert(err, checker.NotNil)
}

func (s *DockerSuite) TestInspectTypeFlagWithImage(c *check.C) {
	//Both the container and image are named busybox. docker inspect will fetch image
	//JSON as --type=image. if there is no image with name busybox, docker inspect
	//will throw an error.

	dockerCmd(c, "run", "--name=busybox", "-d", "busybox", "true")

	out, _ := dockerCmd(c, "inspect", "--type=image", "busybox")
	c.Assert(out, checker.Not(checker.Contains), "State") // not an image JSON
}

func (s *DockerSuite) TestInspectTypeFlagWithInvalidValue(c *check.C) {
	//Both the container and image are named busybox. docker inspect will fail
	//as --type=foobar is not a valid value for the flag.

	dockerCmd(c, "run", "--name=busybox", "-d", "busybox", "true")

	out, exitCode, err := dockerCmdWithError("inspect", "--type=foobar", "busybox")
	c.Assert(err, checker.NotNil, check.Commentf("%s", exitCode))
	c.Assert(exitCode, checker.Equals, 1, check.Commentf("%s", err))
	c.Assert(out, checker.Contains, "not a valid value for --type")
}

func (s *DockerSuite) TestInspectImageFilterInt(c *check.C) {
	testRequires(c, DaemonIsLinux)
	imageTest := "emptyfs"
	out := inspectField(c, imageTest, "Size")

	size, err := strconv.Atoi(out)
	c.Assert(err, checker.IsNil, check.Commentf("failed to inspect size of the image: %s, %v", out, err))

	//now see if the size turns out to be the same
	formatStr := fmt.Sprintf("--format={{eq .Size %d}}", size)
	out, _ = dockerCmd(c, "inspect", formatStr, imageTest)
	result, err := strconv.ParseBool(strings.TrimSuffix(out, "\n"))
	c.Assert(err, checker.IsNil)
	c.Assert(result, checker.Equals, true)
}

func (s *DockerSuite) TestInspectContainerFilterInt(c *check.C) {
	result := icmd.RunCmd(icmd.Cmd{
		Command: []string{dockerBinary, "run", "-i", "-a", "stdin", "busybox", "cat"},
		Stdin:   strings.NewReader("blahblah"),
	})
	result.Assert(c, icmd.Success)
	out := result.Stdout()
	id := strings.TrimSpace(out)

	out = inspectField(c, id, "State.ExitCode")

	exitCode, err := strconv.Atoi(out)
	c.Assert(err, checker.IsNil, check.Commentf("failed to inspect exitcode of the container: %s, %v", out, err))

	//now get the exit code to verify
	formatStr := fmt.Sprintf("--format={{eq .State.ExitCode %d}}", exitCode)
	out, _ = dockerCmd(c, "inspect", formatStr, id)
	inspectResult, err := strconv.ParseBool(strings.TrimSuffix(out, "\n"))
	c.Assert(err, checker.IsNil)
	c.Assert(inspectResult, checker.Equals, true)
}

func (s *DockerSuite) TestInspectImageGraphDriver(c *check.C) {
	testRequires(c, DaemonIsLinux, Devicemapper)
	imageTest := "emptyfs"
	name := inspectField(c, imageTest, "GraphDriver.Name")

	checkValidGraphDriver(c, name)

	deviceID := inspectField(c, imageTest, "GraphDriver.Data.DeviceId")

	_, err := strconv.Atoi(deviceID)
	c.Assert(err, checker.IsNil, check.Commentf("failed to inspect DeviceId of the image: %s, %v", deviceID, err))

	deviceSize := inspectField(c, imageTest, "GraphDriver.Data.DeviceSize")

	_, err = strconv.ParseUint(deviceSize, 10, 64)
	c.Assert(err, checker.IsNil, check.Commentf("failed to inspect DeviceSize of the image: %s, %v", deviceSize, err))
}

func (s *DockerSuite) TestInspectContainerGraphDriver(c *check.C) {
	testRequires(c, DaemonIsLinux, Devicemapper)

	out, _ := dockerCmd(c, "run", "-d", "busybox", "true")
	out = strings.TrimSpace(out)

	name := inspectField(c, out, "GraphDriver.Name")

	checkValidGraphDriver(c, name)

	imageDeviceID := inspectField(c, "busybox", "GraphDriver.Data.DeviceId")

	deviceID := inspectField(c, out, "GraphDriver.Data.DeviceId")

	c.Assert(imageDeviceID, checker.Not(checker.Equals), deviceID)

	_, err := strconv.Atoi(deviceID)
	c.Assert(err, checker.IsNil, check.Commentf("failed to inspect DeviceId of the image: %s, %v", deviceID, err))

	deviceSize := inspectField(c, out, "GraphDriver.Data.DeviceSize")

	_, err = strconv.ParseUint(deviceSize, 10, 64)
	c.Assert(err, checker.IsNil, check.Commentf("failed to inspect DeviceSize of the image: %s, %v", deviceSize, err))
}

func (s *DockerSuite) TestInspectBindMountPoint(c *check.C) {
	modifier := ",z"
	prefix, slash := getPrefixAndSlashFromDaemonPlatform()
	if testEnv.DaemonPlatform() == "windows" {
		modifier = ""
		// Linux creates the host directory if it doesn't exist. Windows does not.
		os.Mkdir(`c:\data`, os.ModeDir)
	}

	dockerCmd(c, "run", "-d", "--name", "test", "-v", prefix+slash+"data:"+prefix+slash+"data:ro"+modifier, "busybox", "cat")

	vol := inspectFieldJSON(c, "test", "Mounts")

	var mp []types.MountPoint
	err := json.Unmarshal([]byte(vol), &mp)
	c.Assert(err, checker.IsNil)

	// check that there is only one mountpoint
	c.Assert(mp, check.HasLen, 1)

	m := mp[0]

	c.Assert(m.Name, checker.Equals, "")
	c.Assert(m.Driver, checker.Equals, "")
	c.Assert(m.Source, checker.Equals, prefix+slash+"data")
	c.Assert(m.Destination, checker.Equals, prefix+slash+"data")
	if testEnv.DaemonPlatform() != "windows" { // Windows does not set mode
		c.Assert(m.Mode, checker.Equals, "ro"+modifier)
	}
	c.Assert(m.RW, checker.Equals, false)
}

func (s *DockerSuite) TestInspectNamedMountPoint(c *check.C) {
	prefix, slash := getPrefixAndSlashFromDaemonPlatform()

	dockerCmd(c, "run", "-d", "--name", "test", "-v", "data:"+prefix+slash+"data", "busybox", "cat")

	vol := inspectFieldJSON(c, "test", "Mounts")

	var mp []types.MountPoint
	err := json.Unmarshal([]byte(vol), &mp)
	c.Assert(err, checker.IsNil)

	// check that there is only one mountpoint
	c.Assert(mp, checker.HasLen, 1)

	m := mp[0]

	c.Assert(m.Name, checker.Equals, "data")
	c.Assert(m.Driver, checker.Equals, "local")
	c.Assert(m.Source, checker.Not(checker.Equals), "")
	c.Assert(m.Destination, checker.Equals, prefix+slash+"data")
	c.Assert(m.RW, checker.Equals, true)
}

// #14947
func (s *DockerSuite) TestInspectTimesAsRFC3339Nano(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "true")
	id := strings.TrimSpace(out)
	startedAt := inspectField(c, id, "State.StartedAt")
	finishedAt := inspectField(c, id, "State.FinishedAt")
	created := inspectField(c, id, "Created")

	_, err := time.Parse(time.RFC3339Nano, startedAt)
	c.Assert(err, checker.IsNil)
	_, err = time.Parse(time.RFC3339Nano, finishedAt)
	c.Assert(err, checker.IsNil)
	_, err = time.Parse(time.RFC3339Nano, created)
	c.Assert(err, checker.IsNil)

	created = inspectField(c, "busybox", "Created")

	_, err = time.Parse(time.RFC3339Nano, created)
	c.Assert(err, checker.IsNil)
}

// #15633
func (s *DockerSuite) TestInspectLogConfigNoType(c *check.C) {
	dockerCmd(c, "create", "--name=test", "--log-opt", "max-file=42", "busybox")
	var logConfig container.LogConfig

	out := inspectFieldJSON(c, "test", "HostConfig.LogConfig")

	err := json.NewDecoder(strings.NewReader(out)).Decode(&logConfig)
	c.Assert(err, checker.IsNil, check.Commentf("%v", out))

	c.Assert(logConfig.Type, checker.Equals, "json-file")
	c.Assert(logConfig.Config["max-file"], checker.Equals, "42", check.Commentf("%v", logConfig))
}

func (s *DockerSuite) TestInspectNoSizeFlagContainer(c *check.C) {

	//Both the container and image are named busybox. docker inspect will fetch container
	//JSON SizeRw and SizeRootFs field. If there is no flag --size/-s, there are no size fields.

	runSleepingContainer(c, "--name=busybox", "-d")

	formatStr := "--format={{.SizeRw}},{{.SizeRootFs}}"
	out, _ := dockerCmd(c, "inspect", "--type=container", formatStr, "busybox")
	c.Assert(strings.TrimSpace(out), check.Equals, "<nil>,<nil>", check.Commentf("Expected not to display size info: %s", out))
}

func (s *DockerSuite) TestInspectSizeFlagContainer(c *check.C) {
	runSleepingContainer(c, "--name=busybox", "-d")

	formatStr := "--format='{{.SizeRw}},{{.SizeRootFs}}'"
	out, _ := dockerCmd(c, "inspect", "-s", "--type=container", formatStr, "busybox")
	sz := strings.Split(out, ",")

	c.Assert(strings.TrimSpace(sz[0]), check.Not(check.Equals), "<nil>")
	c.Assert(strings.TrimSpace(sz[1]), check.Not(check.Equals), "<nil>")
}

func (s *DockerSuite) TestInspectTemplateError(c *check.C) {
	// Template parsing error for both the container and image.

	runSleepingContainer(c, "--name=container1", "-d")

	out, _, err := dockerCmdWithError("inspect", "--type=container", "--format='Format container: {{.ThisDoesNotExist}}'", "container1")
	c.Assert(err, check.Not(check.IsNil))
	c.Assert(out, checker.Contains, "Template parsing error")

	out, _, err = dockerCmdWithError("inspect", "--type=image", "--format='Format container: {{.ThisDoesNotExist}}'", "busybox")
	c.Assert(err, check.Not(check.IsNil))
	c.Assert(out, checker.Contains, "Template parsing error")
}

func (s *DockerSuite) TestInspectJSONFields(c *check.C) {
	runSleepingContainer(c, "--name=busybox", "-d")
	out, _, err := dockerCmdWithError("inspect", "--type=container", "--format={{.HostConfig.Dns}}", "busybox")

	c.Assert(err, check.IsNil)
	c.Assert(out, checker.Equals, "[]\n")
}

func (s *DockerSuite) TestInspectByPrefix(c *check.C) {
	id := inspectField(c, "busybox", "Id")
	c.Assert(id, checker.HasPrefix, "sha256:")

	id2 := inspectField(c, id[:12], "Id")
	c.Assert(id, checker.Equals, id2)

	id3 := inspectField(c, strings.TrimPrefix(id, "sha256:")[:12], "Id")
	c.Assert(id, checker.Equals, id3)
}

func (s *DockerSuite) TestInspectStopWhenNotFound(c *check.C) {
	runSleepingContainer(c, "--name=busybox1", "-d")
	runSleepingContainer(c, "--name=busybox2", "-d")
	result := dockerCmdWithResult("inspect", "--type=container", "--format='{{.Name}}'", "busybox1", "busybox2", "missing")

	c.Assert(result.Error, checker.Not(check.IsNil))
	c.Assert(result.Stdout(), checker.Contains, "busybox1")
	c.Assert(result.Stdout(), checker.Contains, "busybox2")
	c.Assert(result.Stderr(), checker.Contains, "Error: No such container: missing")

	// test inspect would not fast fail
	result = dockerCmdWithResult("inspect", "--type=container", "--format='{{.Name}}'", "missing", "busybox1", "busybox2")

	c.Assert(result.Error, checker.Not(check.IsNil))
	c.Assert(result.Stdout(), checker.Contains, "busybox1")
	c.Assert(result.Stdout(), checker.Contains, "busybox2")
	c.Assert(result.Stderr(), checker.Contains, "Error: No such container: missing")
}

func (s *DockerSuite) TestInspectHistory(c *check.C) {
	dockerCmd(c, "run", "--name=testcont", "busybox", "echo", "hello")
	dockerCmd(c, "commit", "-m", "test comment", "testcont", "testimg")
	out, _, err := dockerCmdWithError("inspect", "--format='{{.Comment}}'", "testimg")
	c.Assert(err, check.IsNil)
	c.Assert(out, checker.Contains, "test comment")
}

func (s *DockerSuite) TestInspectContainerNetworkDefault(c *check.C) {
	testRequires(c, DaemonIsLinux)

	contName := "test1"
	dockerCmd(c, "run", "--name", contName, "-d", "busybox", "top")
	netOut, _ := dockerCmd(c, "network", "inspect", "--format={{.ID}}", "bridge")
	out := inspectField(c, contName, "NetworkSettings.Networks")
	c.Assert(out, checker.Contains, "bridge")
	out = inspectField(c, contName, "NetworkSettings.Networks.bridge.NetworkID")
	c.Assert(strings.TrimSpace(out), checker.Equals, strings.TrimSpace(netOut))
}

func (s *DockerSuite) TestInspectContainerNetworkCustom(c *check.C) {
	testRequires(c, DaemonIsLinux)

	netOut, _ := dockerCmd(c, "network", "create", "net1")
	dockerCmd(c, "run", "--name=container1", "--net=net1", "-d", "busybox", "top")
	out := inspectField(c, "container1", "NetworkSettings.Networks")
	c.Assert(out, checker.Contains, "net1")
	out = inspectField(c, "container1", "NetworkSettings.Networks.net1.NetworkID")
	c.Assert(strings.TrimSpace(out), checker.Equals, strings.TrimSpace(netOut))
}

func (s *DockerSuite) TestInspectRootFS(c *check.C) {
	out, _, err := dockerCmdWithError("inspect", "busybox")
	c.Assert(err, check.IsNil)

	var imageJSON []types.ImageInspect
	err = json.Unmarshal([]byte(out), &imageJSON)
	c.Assert(err, checker.IsNil)

	c.Assert(len(imageJSON[0].RootFS.Layers), checker.GreaterOrEqualThan, 1)
}

func (s *DockerSuite) TestInspectAmpersand(c *check.C) {
	testRequires(c, DaemonIsLinux)

	name := "test"
	out, _ := dockerCmd(c, "run", "--name", name, "--env", `TEST_ENV="soanni&rtr"`, "busybox", "env")
	c.Assert(out, checker.Contains, `soanni&rtr`)
	out, _ = dockerCmd(c, "inspect", name)
	c.Assert(out, checker.Contains, `soanni&rtr`)
}

func (s *DockerSuite) TestInspectPlugin(c *check.C) {
	testRequires(c, DaemonIsLinux, IsAmd64, Network)
	_, _, err := dockerCmdWithError("plugin", "install", "--grant-all-permissions", pNameWithTag)
	c.Assert(err, checker.IsNil)

	out, _, err := dockerCmdWithError("inspect", "--type", "plugin", "--format", "{{.Name}}", pNameWithTag)
	c.Assert(err, checker.IsNil)
	c.Assert(strings.TrimSpace(out), checker.Equals, pNameWithTag)

	out, _, err = dockerCmdWithError("inspect", "--format", "{{.Name}}", pNameWithTag)
	c.Assert(err, checker.IsNil)
	c.Assert(strings.TrimSpace(out), checker.Equals, pNameWithTag)

	// Even without tag the inspect still work
	out, _, err = dockerCmdWithError("inspect", "--type", "plugin", "--format", "{{.Name}}", pNameWithTag)
	c.Assert(err, checker.IsNil)
	c.Assert(strings.TrimSpace(out), checker.Equals, pNameWithTag)

	out, _, err = dockerCmdWithError("inspect", "--format", "{{.Name}}", pNameWithTag)
	c.Assert(err, checker.IsNil)
	c.Assert(strings.TrimSpace(out), checker.Equals, pNameWithTag)

	_, _, err = dockerCmdWithError("plugin", "disable", pNameWithTag)
	c.Assert(err, checker.IsNil)

	out, _, err = dockerCmdWithError("plugin", "remove", pNameWithTag)
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Contains, pNameWithTag)
}

// Test case for 29185
func (s *DockerSuite) TestInspectUnknownObject(c *check.C) {
	// This test should work on both Windows and Linux
	out, _, err := dockerCmdWithError("inspect", "foobar")
	c.Assert(err, checker.NotNil)
	c.Assert(out, checker.Contains, "Error: No such object: foobar")
	c.Assert(err.Error(), checker.Contains, "Error: No such object: foobar")
}

func (s *DockerSuite) TestInspectInvalidReference(c *check.C) {
	// This test should work on both Windows and Linux
	out, _, err := dockerCmdWithError("inspect", "FooBar")
	c.Assert(err, checker.NotNil)
	c.Assert(out, checker.Contains, "Error: No such object: FooBar")
	c.Assert(err.Error(), checker.Contains, "Error: No such object: FooBar")
}
