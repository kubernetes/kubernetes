package main

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/cli/build"
	"github.com/docker/docker/integration-cli/request"
	icmd "github.com/docker/docker/pkg/testutil/cmd"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestVolumeCLICreate(c *check.C) {
	dockerCmd(c, "volume", "create")

	_, _, err := dockerCmdWithError("volume", "create", "-d", "nosuchdriver")
	c.Assert(err, check.NotNil)

	// test using hidden --name option
	out, _ := dockerCmd(c, "volume", "create", "--name=test")
	name := strings.TrimSpace(out)
	c.Assert(name, check.Equals, "test")

	out, _ = dockerCmd(c, "volume", "create", "test2")
	name = strings.TrimSpace(out)
	c.Assert(name, check.Equals, "test2")
}

func (s *DockerSuite) TestVolumeCLIInspect(c *check.C) {
	c.Assert(
		exec.Command(dockerBinary, "volume", "inspect", "doesnotexist").Run(),
		check.Not(check.IsNil),
		check.Commentf("volume inspect should error on non-existent volume"),
	)

	out, _ := dockerCmd(c, "volume", "create")
	name := strings.TrimSpace(out)
	out, _ = dockerCmd(c, "volume", "inspect", "--format={{ .Name }}", name)
	c.Assert(strings.TrimSpace(out), check.Equals, name)

	dockerCmd(c, "volume", "create", "test")
	out, _ = dockerCmd(c, "volume", "inspect", "--format={{ .Name }}", "test")
	c.Assert(strings.TrimSpace(out), check.Equals, "test")
}

func (s *DockerSuite) TestVolumeCLIInspectMulti(c *check.C) {
	dockerCmd(c, "volume", "create", "test1")
	dockerCmd(c, "volume", "create", "test2")
	dockerCmd(c, "volume", "create", "test3")

	result := dockerCmdWithResult("volume", "inspect", "--format={{ .Name }}", "test1", "test2", "doesnotexist", "test3")
	c.Assert(result, icmd.Matches, icmd.Expected{
		ExitCode: 1,
		Err:      "No such volume: doesnotexist",
	})

	out := result.Stdout()
	outArr := strings.Split(strings.TrimSpace(out), "\n")
	c.Assert(len(outArr), check.Equals, 3, check.Commentf("\n%s", out))

	c.Assert(out, checker.Contains, "test1")
	c.Assert(out, checker.Contains, "test2")
	c.Assert(out, checker.Contains, "test3")
}

func (s *DockerSuite) TestVolumeCLILs(c *check.C) {
	prefix, _ := getPrefixAndSlashFromDaemonPlatform()
	dockerCmd(c, "volume", "create", "aaa")

	dockerCmd(c, "volume", "create", "test")

	dockerCmd(c, "volume", "create", "soo")
	dockerCmd(c, "run", "-v", "soo:"+prefix+"/foo", "busybox", "ls", "/")

	out, _ := dockerCmd(c, "volume", "ls")
	outArr := strings.Split(strings.TrimSpace(out), "\n")
	c.Assert(len(outArr), check.Equals, 4, check.Commentf("\n%s", out))

	assertVolList(c, out, []string{"aaa", "soo", "test"})
}

func (s *DockerSuite) TestVolumeLsFormat(c *check.C) {
	dockerCmd(c, "volume", "create", "aaa")
	dockerCmd(c, "volume", "create", "test")
	dockerCmd(c, "volume", "create", "soo")

	out, _ := dockerCmd(c, "volume", "ls", "--format", "{{.Name}}")
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")

	expected := []string{"aaa", "soo", "test"}
	var names []string
	names = append(names, lines...)
	c.Assert(expected, checker.DeepEquals, names, check.Commentf("Expected array with truncated names: %v, got: %v", expected, names))
}

func (s *DockerSuite) TestVolumeLsFormatDefaultFormat(c *check.C) {
	dockerCmd(c, "volume", "create", "aaa")
	dockerCmd(c, "volume", "create", "test")
	dockerCmd(c, "volume", "create", "soo")

	config := `{
		"volumesFormat": "{{ .Name }} default"
}`
	d, err := ioutil.TempDir("", "integration-cli-")
	c.Assert(err, checker.IsNil)
	defer os.RemoveAll(d)

	err = ioutil.WriteFile(filepath.Join(d, "config.json"), []byte(config), 0644)
	c.Assert(err, checker.IsNil)

	out, _ := dockerCmd(c, "--config", d, "volume", "ls")
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")

	expected := []string{"aaa default", "soo default", "test default"}
	var names []string
	names = append(names, lines...)
	c.Assert(expected, checker.DeepEquals, names, check.Commentf("Expected array with truncated names: %v, got: %v", expected, names))
}

// assertVolList checks volume retrieved with ls command
// equals to expected volume list
// note: out should be `volume ls [option]` result
func assertVolList(c *check.C, out string, expectVols []string) {
	lines := strings.Split(out, "\n")
	var volList []string
	for _, line := range lines[1 : len(lines)-1] {
		volFields := strings.Fields(line)
		// wrap all volume name in volList
		volList = append(volList, volFields[1])
	}

	// volume ls should contains all expected volumes
	c.Assert(volList, checker.DeepEquals, expectVols)
}

func (s *DockerSuite) TestVolumeCLILsFilterDangling(c *check.C) {
	prefix, _ := getPrefixAndSlashFromDaemonPlatform()
	dockerCmd(c, "volume", "create", "testnotinuse1")
	dockerCmd(c, "volume", "create", "testisinuse1")
	dockerCmd(c, "volume", "create", "testisinuse2")

	// Make sure both "created" (but not started), and started
	// containers are included in reference counting
	dockerCmd(c, "run", "--name", "volume-test1", "-v", "testisinuse1:"+prefix+"/foo", "busybox", "true")
	dockerCmd(c, "create", "--name", "volume-test2", "-v", "testisinuse2:"+prefix+"/foo", "busybox", "true")

	out, _ := dockerCmd(c, "volume", "ls")

	// No filter, all volumes should show
	c.Assert(out, checker.Contains, "testnotinuse1\n", check.Commentf("expected volume 'testnotinuse1' in output"))
	c.Assert(out, checker.Contains, "testisinuse1\n", check.Commentf("expected volume 'testisinuse1' in output"))
	c.Assert(out, checker.Contains, "testisinuse2\n", check.Commentf("expected volume 'testisinuse2' in output"))

	out, _ = dockerCmd(c, "volume", "ls", "--filter", "dangling=false")

	// Explicitly disabling dangling
	c.Assert(out, check.Not(checker.Contains), "testnotinuse1\n", check.Commentf("expected volume 'testnotinuse1' in output"))
	c.Assert(out, checker.Contains, "testisinuse1\n", check.Commentf("expected volume 'testisinuse1' in output"))
	c.Assert(out, checker.Contains, "testisinuse2\n", check.Commentf("expected volume 'testisinuse2' in output"))

	out, _ = dockerCmd(c, "volume", "ls", "--filter", "dangling=true")

	// Filter "dangling" volumes; only "dangling" (unused) volumes should be in the output
	c.Assert(out, checker.Contains, "testnotinuse1\n", check.Commentf("expected volume 'testnotinuse1' in output"))
	c.Assert(out, check.Not(checker.Contains), "testisinuse1\n", check.Commentf("volume 'testisinuse1' in output, but not expected"))
	c.Assert(out, check.Not(checker.Contains), "testisinuse2\n", check.Commentf("volume 'testisinuse2' in output, but not expected"))

	out, _ = dockerCmd(c, "volume", "ls", "--filter", "dangling=1")
	// Filter "dangling" volumes; only "dangling" (unused) volumes should be in the output, dangling also accept 1
	c.Assert(out, checker.Contains, "testnotinuse1\n", check.Commentf("expected volume 'testnotinuse1' in output"))
	c.Assert(out, check.Not(checker.Contains), "testisinuse1\n", check.Commentf("volume 'testisinuse1' in output, but not expected"))
	c.Assert(out, check.Not(checker.Contains), "testisinuse2\n", check.Commentf("volume 'testisinuse2' in output, but not expected"))

	out, _ = dockerCmd(c, "volume", "ls", "--filter", "dangling=0")
	// dangling=0 is same as dangling=false case
	c.Assert(out, check.Not(checker.Contains), "testnotinuse1\n", check.Commentf("expected volume 'testnotinuse1' in output"))
	c.Assert(out, checker.Contains, "testisinuse1\n", check.Commentf("expected volume 'testisinuse1' in output"))
	c.Assert(out, checker.Contains, "testisinuse2\n", check.Commentf("expected volume 'testisinuse2' in output"))

	out, _ = dockerCmd(c, "volume", "ls", "--filter", "name=testisin")
	c.Assert(out, check.Not(checker.Contains), "testnotinuse1\n", check.Commentf("expected volume 'testnotinuse1' in output"))
	c.Assert(out, checker.Contains, "testisinuse1\n", check.Commentf("expected volume 'testisinuse1' in output"))
	c.Assert(out, checker.Contains, "testisinuse2\n", check.Commentf("expected volume 'testisinuse2' in output"))
}

func (s *DockerSuite) TestVolumeCLILsErrorWithInvalidFilterName(c *check.C) {
	out, _, err := dockerCmdWithError("volume", "ls", "-f", "FOO=123")
	c.Assert(err, checker.NotNil)
	c.Assert(out, checker.Contains, "Invalid filter")
}

func (s *DockerSuite) TestVolumeCLILsWithIncorrectFilterValue(c *check.C) {
	out, _, err := dockerCmdWithError("volume", "ls", "-f", "dangling=invalid")
	c.Assert(err, check.NotNil)
	c.Assert(out, checker.Contains, "Invalid filter")
}

func (s *DockerSuite) TestVolumeCLIRm(c *check.C) {
	prefix, _ := getPrefixAndSlashFromDaemonPlatform()
	out, _ := dockerCmd(c, "volume", "create")
	id := strings.TrimSpace(out)

	dockerCmd(c, "volume", "create", "test")
	dockerCmd(c, "volume", "rm", id)
	dockerCmd(c, "volume", "rm", "test")

	out, _ = dockerCmd(c, "volume", "ls")
	outArr := strings.Split(strings.TrimSpace(out), "\n")
	c.Assert(len(outArr), check.Equals, 1, check.Commentf("%s\n", out))

	volumeID := "testing"
	dockerCmd(c, "run", "-v", volumeID+":"+prefix+"/foo", "--name=test", "busybox", "sh", "-c", "echo hello > /foo/bar")

	icmd.RunCommand(dockerBinary, "volume", "rm", "testing").Assert(c, icmd.Expected{
		ExitCode: 1,
		Error:    "exit status 1",
	})

	out, _ = dockerCmd(c, "run", "--volumes-from=test", "--name=test2", "busybox", "sh", "-c", "cat /foo/bar")
	c.Assert(strings.TrimSpace(out), check.Equals, "hello")
	dockerCmd(c, "rm", "-fv", "test2")
	dockerCmd(c, "volume", "inspect", volumeID)
	dockerCmd(c, "rm", "-f", "test")

	out, _ = dockerCmd(c, "run", "--name=test2", "-v", volumeID+":"+prefix+"/foo", "busybox", "sh", "-c", "cat /foo/bar")
	c.Assert(strings.TrimSpace(out), check.Equals, "hello", check.Commentf("volume data was removed"))
	dockerCmd(c, "rm", "test2")

	dockerCmd(c, "volume", "rm", volumeID)
	c.Assert(
		exec.Command("volume", "rm", "doesnotexist").Run(),
		check.Not(check.IsNil),
		check.Commentf("volume rm should fail with non-existent volume"),
	)
}

// FIXME(vdemeester) should be a unit test in cli/command/volume package
func (s *DockerSuite) TestVolumeCLINoArgs(c *check.C) {
	out, _ := dockerCmd(c, "volume")
	// no args should produce the cmd usage output
	usage := "Usage:	docker volume COMMAND"
	c.Assert(out, checker.Contains, usage)

	// invalid arg should error and show the command usage on stderr
	icmd.RunCommand(dockerBinary, "volume", "somearg").Assert(c, icmd.Expected{
		ExitCode: 1,
		Error:    "exit status 1",
		Err:      usage,
	})

	// invalid flag should error and show the flag error and cmd usage
	result := icmd.RunCommand(dockerBinary, "volume", "--no-such-flag")
	result.Assert(c, icmd.Expected{
		ExitCode: 125,
		Error:    "exit status 125",
		Err:      usage,
	})
	c.Assert(result.Stderr(), checker.Contains, "unknown flag: --no-such-flag")
}

func (s *DockerSuite) TestVolumeCLIInspectTmplError(c *check.C) {
	out, _ := dockerCmd(c, "volume", "create")
	name := strings.TrimSpace(out)

	out, exitCode, err := dockerCmdWithError("volume", "inspect", "--format='{{ .FooBar }}'", name)
	c.Assert(err, checker.NotNil, check.Commentf("Output: %s", out))
	c.Assert(exitCode, checker.Equals, 1, check.Commentf("Output: %s", out))
	c.Assert(out, checker.Contains, "Template parsing error")
}

func (s *DockerSuite) TestVolumeCLICreateWithOpts(c *check.C) {
	testRequires(c, DaemonIsLinux)

	dockerCmd(c, "volume", "create", "-d", "local", "test", "--opt=type=tmpfs", "--opt=device=tmpfs", "--opt=o=size=1m,uid=1000")
	out, _ := dockerCmd(c, "run", "-v", "test:/foo", "busybox", "mount")

	mounts := strings.Split(out, "\n")
	var found bool
	for _, m := range mounts {
		if strings.Contains(m, "/foo") {
			found = true
			info := strings.Fields(m)
			// tmpfs on <path> type tmpfs (rw,relatime,size=1024k,uid=1000)
			c.Assert(info[0], checker.Equals, "tmpfs")
			c.Assert(info[2], checker.Equals, "/foo")
			c.Assert(info[4], checker.Equals, "tmpfs")
			c.Assert(info[5], checker.Contains, "uid=1000")
			c.Assert(info[5], checker.Contains, "size=1024k")
			break
		}
	}
	c.Assert(found, checker.Equals, true)
}

func (s *DockerSuite) TestVolumeCLICreateLabel(c *check.C) {
	testVol := "testvolcreatelabel"
	testLabel := "foo"
	testValue := "bar"

	out, _, err := dockerCmdWithError("volume", "create", "--label", testLabel+"="+testValue, testVol)
	c.Assert(err, check.IsNil)

	out, _ = dockerCmd(c, "volume", "inspect", "--format={{ .Labels."+testLabel+" }}", testVol)
	c.Assert(strings.TrimSpace(out), check.Equals, testValue)
}

func (s *DockerSuite) TestVolumeCLICreateLabelMultiple(c *check.C) {
	testVol := "testvolcreatelabel"

	testLabels := map[string]string{
		"foo": "bar",
		"baz": "foo",
	}

	args := []string{
		"volume",
		"create",
		testVol,
	}

	for k, v := range testLabels {
		args = append(args, "--label", k+"="+v)
	}

	out, _, err := dockerCmdWithError(args...)
	c.Assert(err, check.IsNil)

	for k, v := range testLabels {
		out, _ = dockerCmd(c, "volume", "inspect", "--format={{ .Labels."+k+" }}", testVol)
		c.Assert(strings.TrimSpace(out), check.Equals, v)
	}
}

func (s *DockerSuite) TestVolumeCLILsFilterLabels(c *check.C) {
	testVol1 := "testvolcreatelabel-1"
	out, _, err := dockerCmdWithError("volume", "create", "--label", "foo=bar1", testVol1)
	c.Assert(err, check.IsNil)

	testVol2 := "testvolcreatelabel-2"
	out, _, err = dockerCmdWithError("volume", "create", "--label", "foo=bar2", testVol2)
	c.Assert(err, check.IsNil)

	out, _ = dockerCmd(c, "volume", "ls", "--filter", "label=foo")

	// filter with label=key
	c.Assert(out, checker.Contains, "testvolcreatelabel-1\n", check.Commentf("expected volume 'testvolcreatelabel-1' in output"))
	c.Assert(out, checker.Contains, "testvolcreatelabel-2\n", check.Commentf("expected volume 'testvolcreatelabel-2' in output"))

	out, _ = dockerCmd(c, "volume", "ls", "--filter", "label=foo=bar1")

	// filter with label=key=value
	c.Assert(out, checker.Contains, "testvolcreatelabel-1\n", check.Commentf("expected volume 'testvolcreatelabel-1' in output"))
	c.Assert(out, check.Not(checker.Contains), "testvolcreatelabel-2\n", check.Commentf("expected volume 'testvolcreatelabel-2 in output"))

	out, _ = dockerCmd(c, "volume", "ls", "--filter", "label=non-exist")
	outArr := strings.Split(strings.TrimSpace(out), "\n")
	c.Assert(len(outArr), check.Equals, 1, check.Commentf("\n%s", out))

	out, _ = dockerCmd(c, "volume", "ls", "--filter", "label=foo=non-exist")
	outArr = strings.Split(strings.TrimSpace(out), "\n")
	c.Assert(len(outArr), check.Equals, 1, check.Commentf("\n%s", out))
}

func (s *DockerSuite) TestVolumeCLILsFilterDrivers(c *check.C) {
	// using default volume driver local to create volumes
	testVol1 := "testvol-1"
	out, _, err := dockerCmdWithError("volume", "create", testVol1)
	c.Assert(err, check.IsNil)

	testVol2 := "testvol-2"
	out, _, err = dockerCmdWithError("volume", "create", testVol2)
	c.Assert(err, check.IsNil)

	// filter with driver=local
	out, _ = dockerCmd(c, "volume", "ls", "--filter", "driver=local")
	c.Assert(out, checker.Contains, "testvol-1\n", check.Commentf("expected volume 'testvol-1' in output"))
	c.Assert(out, checker.Contains, "testvol-2\n", check.Commentf("expected volume 'testvol-2' in output"))

	// filter with driver=invaliddriver
	out, _ = dockerCmd(c, "volume", "ls", "--filter", "driver=invaliddriver")
	outArr := strings.Split(strings.TrimSpace(out), "\n")
	c.Assert(len(outArr), check.Equals, 1, check.Commentf("\n%s", out))

	// filter with driver=loca
	out, _ = dockerCmd(c, "volume", "ls", "--filter", "driver=loca")
	outArr = strings.Split(strings.TrimSpace(out), "\n")
	c.Assert(len(outArr), check.Equals, 1, check.Commentf("\n%s", out))

	// filter with driver=
	out, _ = dockerCmd(c, "volume", "ls", "--filter", "driver=")
	outArr = strings.Split(strings.TrimSpace(out), "\n")
	c.Assert(len(outArr), check.Equals, 1, check.Commentf("\n%s", out))
}

func (s *DockerSuite) TestVolumeCLIRmForceUsage(c *check.C) {
	out, _ := dockerCmd(c, "volume", "create")
	id := strings.TrimSpace(out)

	dockerCmd(c, "volume", "rm", "-f", id)
	dockerCmd(c, "volume", "rm", "--force", "nonexist")

	out, _ = dockerCmd(c, "volume", "ls")
	outArr := strings.Split(strings.TrimSpace(out), "\n")
	c.Assert(len(outArr), check.Equals, 1, check.Commentf("%s\n", out))
}

func (s *DockerSuite) TestVolumeCLIRmForce(c *check.C) {
	testRequires(c, SameHostDaemon, DaemonIsLinux)

	name := "test"
	out, _ := dockerCmd(c, "volume", "create", name)
	id := strings.TrimSpace(out)
	c.Assert(id, checker.Equals, name)

	out, _ = dockerCmd(c, "volume", "inspect", "--format", "{{.Mountpoint}}", name)
	c.Assert(strings.TrimSpace(out), checker.Not(checker.Equals), "")
	// Mountpoint is in the form of "/var/lib/docker/volumes/.../_data", removing `/_data`
	path := strings.TrimSuffix(strings.TrimSpace(out), "/_data")
	icmd.RunCommand("rm", "-rf", path).Assert(c, icmd.Success)

	dockerCmd(c, "volume", "rm", "-f", name)
	out, _ = dockerCmd(c, "volume", "ls")
	c.Assert(out, checker.Not(checker.Contains), name)
	dockerCmd(c, "volume", "create", name)
	out, _ = dockerCmd(c, "volume", "ls")
	c.Assert(out, checker.Contains, name)
}

// TestVolumeCLIRmForceInUse verifies that repeated `docker volume rm -f` calls does not remove a volume
// if it is in use. Test case for https://github.com/docker/docker/issues/31446
func (s *DockerSuite) TestVolumeCLIRmForceInUse(c *check.C) {
	name := "testvolume"
	out, _ := dockerCmd(c, "volume", "create", name)
	id := strings.TrimSpace(out)
	c.Assert(id, checker.Equals, name)

	prefix, slash := getPrefixAndSlashFromDaemonPlatform()
	out, e := dockerCmd(c, "create", "-v", "testvolume:"+prefix+slash+"foo", "busybox")
	cid := strings.TrimSpace(out)

	_, _, err := dockerCmdWithError("volume", "rm", "-f", name)
	c.Assert(err, check.NotNil)
	c.Assert(err.Error(), checker.Contains, "volume is in use")
	out, _ = dockerCmd(c, "volume", "ls")
	c.Assert(out, checker.Contains, name)

	// The original issue did not _remove_ the volume from the list
	// the first time. But a second call to `volume rm` removed it.
	// Calling `volume rm` a second time to confirm it's not removed
	// when calling twice.
	_, _, err = dockerCmdWithError("volume", "rm", "-f", name)
	c.Assert(err, check.NotNil)
	c.Assert(err.Error(), checker.Contains, "volume is in use")
	out, _ = dockerCmd(c, "volume", "ls")
	c.Assert(out, checker.Contains, name)

	// Verify removing the volume after the container is removed works
	_, e = dockerCmd(c, "rm", cid)
	c.Assert(e, check.Equals, 0)

	_, e = dockerCmd(c, "volume", "rm", "-f", name)
	c.Assert(e, check.Equals, 0)

	out, e = dockerCmd(c, "volume", "ls")
	c.Assert(e, check.Equals, 0)
	c.Assert(out, checker.Not(checker.Contains), name)
}

func (s *DockerSuite) TestVolumeCliInspectWithVolumeOpts(c *check.C) {
	testRequires(c, DaemonIsLinux)

	// Without options
	name := "test1"
	dockerCmd(c, "volume", "create", "-d", "local", name)
	out, _ := dockerCmd(c, "volume", "inspect", "--format={{ .Options }}", name)
	c.Assert(strings.TrimSpace(out), checker.Contains, "map[]")

	// With options
	name = "test2"
	k1, v1 := "type", "tmpfs"
	k2, v2 := "device", "tmpfs"
	k3, v3 := "o", "size=1m,uid=1000"
	dockerCmd(c, "volume", "create", "-d", "local", name, "--opt", fmt.Sprintf("%s=%s", k1, v1), "--opt", fmt.Sprintf("%s=%s", k2, v2), "--opt", fmt.Sprintf("%s=%s", k3, v3))
	out, _ = dockerCmd(c, "volume", "inspect", "--format={{ .Options }}", name)
	c.Assert(strings.TrimSpace(out), checker.Contains, fmt.Sprintf("%s:%s", k1, v1))
	c.Assert(strings.TrimSpace(out), checker.Contains, fmt.Sprintf("%s:%s", k2, v2))
	c.Assert(strings.TrimSpace(out), checker.Contains, fmt.Sprintf("%s:%s", k3, v3))
}

// Test case (1) for 21845: duplicate targets for --volumes-from
func (s *DockerSuite) TestDuplicateMountpointsForVolumesFrom(c *check.C) {
	testRequires(c, DaemonIsLinux)

	image := "vimage"
	buildImageSuccessfully(c, image, build.WithDockerfile(`
		FROM busybox
		VOLUME ["/tmp/data"]`))

	dockerCmd(c, "run", "--name=data1", image, "true")
	dockerCmd(c, "run", "--name=data2", image, "true")

	out, _ := dockerCmd(c, "inspect", "--format", "{{(index .Mounts 0).Name}}", "data1")
	data1 := strings.TrimSpace(out)
	c.Assert(data1, checker.Not(checker.Equals), "")

	out, _ = dockerCmd(c, "inspect", "--format", "{{(index .Mounts 0).Name}}", "data2")
	data2 := strings.TrimSpace(out)
	c.Assert(data2, checker.Not(checker.Equals), "")

	// Both volume should exist
	out, _ = dockerCmd(c, "volume", "ls", "-q")
	c.Assert(strings.TrimSpace(out), checker.Contains, data1)
	c.Assert(strings.TrimSpace(out), checker.Contains, data2)

	out, _, err := dockerCmdWithError("run", "--name=app", "--volumes-from=data1", "--volumes-from=data2", "-d", "busybox", "top")
	c.Assert(err, checker.IsNil, check.Commentf("Out: %s", out))

	// Only the second volume will be referenced, this is backward compatible
	out, _ = dockerCmd(c, "inspect", "--format", "{{(index .Mounts 0).Name}}", "app")
	c.Assert(strings.TrimSpace(out), checker.Equals, data2)

	dockerCmd(c, "rm", "-f", "-v", "app")
	dockerCmd(c, "rm", "-f", "-v", "data1")
	dockerCmd(c, "rm", "-f", "-v", "data2")

	// Both volume should not exist
	out, _ = dockerCmd(c, "volume", "ls", "-q")
	c.Assert(strings.TrimSpace(out), checker.Not(checker.Contains), data1)
	c.Assert(strings.TrimSpace(out), checker.Not(checker.Contains), data2)
}

// Test case (2) for 21845: duplicate targets for --volumes-from and -v (bind)
func (s *DockerSuite) TestDuplicateMountpointsForVolumesFromAndBind(c *check.C) {
	testRequires(c, DaemonIsLinux)

	image := "vimage"
	buildImageSuccessfully(c, image, build.WithDockerfile(`
                FROM busybox
                VOLUME ["/tmp/data"]`))

	dockerCmd(c, "run", "--name=data1", image, "true")
	dockerCmd(c, "run", "--name=data2", image, "true")

	out, _ := dockerCmd(c, "inspect", "--format", "{{(index .Mounts 0).Name}}", "data1")
	data1 := strings.TrimSpace(out)
	c.Assert(data1, checker.Not(checker.Equals), "")

	out, _ = dockerCmd(c, "inspect", "--format", "{{(index .Mounts 0).Name}}", "data2")
	data2 := strings.TrimSpace(out)
	c.Assert(data2, checker.Not(checker.Equals), "")

	// Both volume should exist
	out, _ = dockerCmd(c, "volume", "ls", "-q")
	c.Assert(strings.TrimSpace(out), checker.Contains, data1)
	c.Assert(strings.TrimSpace(out), checker.Contains, data2)

	// /tmp/data is automatically created, because we are not using the modern mount API here
	out, _, err := dockerCmdWithError("run", "--name=app", "--volumes-from=data1", "--volumes-from=data2", "-v", "/tmp/data:/tmp/data", "-d", "busybox", "top")
	c.Assert(err, checker.IsNil, check.Commentf("Out: %s", out))

	// No volume will be referenced (mount is /tmp/data), this is backward compatible
	out, _ = dockerCmd(c, "inspect", "--format", "{{(index .Mounts 0).Name}}", "app")
	c.Assert(strings.TrimSpace(out), checker.Not(checker.Contains), data1)
	c.Assert(strings.TrimSpace(out), checker.Not(checker.Contains), data2)

	dockerCmd(c, "rm", "-f", "-v", "app")
	dockerCmd(c, "rm", "-f", "-v", "data1")
	dockerCmd(c, "rm", "-f", "-v", "data2")

	// Both volume should not exist
	out, _ = dockerCmd(c, "volume", "ls", "-q")
	c.Assert(strings.TrimSpace(out), checker.Not(checker.Contains), data1)
	c.Assert(strings.TrimSpace(out), checker.Not(checker.Contains), data2)
}

// Test case (3) for 21845: duplicate targets for --volumes-from and `Mounts` (API only)
func (s *DockerSuite) TestDuplicateMountpointsForVolumesFromAndMounts(c *check.C) {
	testRequires(c, SameHostDaemon, DaemonIsLinux)

	image := "vimage"
	buildImageSuccessfully(c, image, build.WithDockerfile(`
                FROM busybox
                VOLUME ["/tmp/data"]`))

	dockerCmd(c, "run", "--name=data1", image, "true")
	dockerCmd(c, "run", "--name=data2", image, "true")

	out, _ := dockerCmd(c, "inspect", "--format", "{{(index .Mounts 0).Name}}", "data1")
	data1 := strings.TrimSpace(out)
	c.Assert(data1, checker.Not(checker.Equals), "")

	out, _ = dockerCmd(c, "inspect", "--format", "{{(index .Mounts 0).Name}}", "data2")
	data2 := strings.TrimSpace(out)
	c.Assert(data2, checker.Not(checker.Equals), "")

	// Both volume should exist
	out, _ = dockerCmd(c, "volume", "ls", "-q")
	c.Assert(strings.TrimSpace(out), checker.Contains, data1)
	c.Assert(strings.TrimSpace(out), checker.Contains, data2)

	err := os.MkdirAll("/tmp/data", 0755)
	c.Assert(err, checker.IsNil)
	// Mounts is available in API
	status, body, err := request.SockRequest("POST", "/containers/create?name=app", map[string]interface{}{
		"Image": "busybox",
		"Cmd":   []string{"top"},
		"HostConfig": map[string]interface{}{
			"VolumesFrom": []string{
				"data1",
				"data2",
			},
			"Mounts": []map[string]interface{}{
				{
					"Type":   "bind",
					"Source": "/tmp/data",
					"Target": "/tmp/data",
				},
			}},
	}, daemonHost())

	c.Assert(err, checker.IsNil, check.Commentf(string(body)))
	c.Assert(status, checker.Equals, http.StatusCreated, check.Commentf(string(body)))

	// No volume will be referenced (mount is /tmp/data), this is backward compatible
	out, _ = dockerCmd(c, "inspect", "--format", "{{(index .Mounts 0).Name}}", "app")
	c.Assert(strings.TrimSpace(out), checker.Not(checker.Contains), data1)
	c.Assert(strings.TrimSpace(out), checker.Not(checker.Contains), data2)

	dockerCmd(c, "rm", "-f", "-v", "app")
	dockerCmd(c, "rm", "-f", "-v", "data1")
	dockerCmd(c, "rm", "-f", "-v", "data2")

	// Both volume should not exist
	out, _ = dockerCmd(c, "volume", "ls", "-q")
	c.Assert(strings.TrimSpace(out), checker.Not(checker.Contains), data1)
	c.Assert(strings.TrimSpace(out), checker.Not(checker.Contains), data2)
}
