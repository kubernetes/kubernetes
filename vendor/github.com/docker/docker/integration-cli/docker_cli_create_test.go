package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"reflect"
	"strings"
	"time"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/cli"
	"github.com/docker/docker/integration-cli/cli/build"
	"github.com/docker/docker/integration-cli/cli/build/fakecontext"
	"github.com/docker/docker/pkg/stringid"
	icmd "github.com/docker/docker/pkg/testutil/cmd"
	"github.com/docker/go-connections/nat"
	"github.com/go-check/check"
)

// Make sure we can create a simple container with some args
func (s *DockerSuite) TestCreateArgs(c *check.C) {
	// Intentionally clear entrypoint, as the Windows busybox image needs an entrypoint, which breaks this test
	out, _ := dockerCmd(c, "create", "--entrypoint=", "busybox", "command", "arg1", "arg2", "arg with space", "-c", "flags")

	cleanedContainerID := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "inspect", cleanedContainerID)

	containers := []struct {
		ID      string
		Created time.Time
		Path    string
		Args    []string
		Image   string
	}{}

	err := json.Unmarshal([]byte(out), &containers)
	c.Assert(err, check.IsNil, check.Commentf("Error inspecting the container: %s", err))
	c.Assert(containers, checker.HasLen, 1)

	cont := containers[0]
	c.Assert(string(cont.Path), checker.Equals, "command", check.Commentf("Unexpected container path. Expected command, received: %s", cont.Path))

	b := false
	expected := []string{"arg1", "arg2", "arg with space", "-c", "flags"}
	for i, arg := range expected {
		if arg != cont.Args[i] {
			b = true
			break
		}
	}
	if len(cont.Args) != len(expected) || b {
		c.Fatalf("Unexpected args. Expected %v, received: %v", expected, cont.Args)
	}

}

// Make sure we can grow the container's rootfs at creation time.
func (s *DockerSuite) TestCreateGrowRootfs(c *check.C) {
	// Windows and Devicemapper support growing the rootfs
	if testEnv.DaemonPlatform() != "windows" {
		testRequires(c, Devicemapper)
	}
	out, _ := dockerCmd(c, "create", "--storage-opt", "size=120G", "busybox")

	cleanedContainerID := strings.TrimSpace(out)

	inspectOut := inspectField(c, cleanedContainerID, "HostConfig.StorageOpt")
	c.Assert(inspectOut, checker.Equals, "map[size:120G]")
}

// Make sure we cannot shrink the container's rootfs at creation time.
func (s *DockerSuite) TestCreateShrinkRootfs(c *check.C) {
	testRequires(c, Devicemapper)

	// Ensure this fails because of the defaultBaseFsSize is 10G
	out, _, err := dockerCmdWithError("create", "--storage-opt", "size=5G", "busybox")
	c.Assert(err, check.NotNil, check.Commentf(out))
	c.Assert(out, checker.Contains, "Container size cannot be smaller than")
}

// Make sure we can set hostconfig options too
func (s *DockerSuite) TestCreateHostConfig(c *check.C) {
	out, _ := dockerCmd(c, "create", "-P", "busybox", "echo")

	cleanedContainerID := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "inspect", cleanedContainerID)

	containers := []struct {
		HostConfig *struct {
			PublishAllPorts bool
		}
	}{}

	err := json.Unmarshal([]byte(out), &containers)
	c.Assert(err, check.IsNil, check.Commentf("Error inspecting the container: %s", err))
	c.Assert(containers, checker.HasLen, 1)

	cont := containers[0]
	c.Assert(cont.HostConfig, check.NotNil, check.Commentf("Expected HostConfig, got none"))
	c.Assert(cont.HostConfig.PublishAllPorts, check.NotNil, check.Commentf("Expected PublishAllPorts, got false"))
}

func (s *DockerSuite) TestCreateWithPortRange(c *check.C) {
	out, _ := dockerCmd(c, "create", "-p", "3300-3303:3300-3303/tcp", "busybox", "echo")

	cleanedContainerID := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "inspect", cleanedContainerID)

	containers := []struct {
		HostConfig *struct {
			PortBindings map[nat.Port][]nat.PortBinding
		}
	}{}
	err := json.Unmarshal([]byte(out), &containers)
	c.Assert(err, check.IsNil, check.Commentf("Error inspecting the container: %s", err))
	c.Assert(containers, checker.HasLen, 1)

	cont := containers[0]

	c.Assert(cont.HostConfig, check.NotNil, check.Commentf("Expected HostConfig, got none"))
	c.Assert(cont.HostConfig.PortBindings, checker.HasLen, 4, check.Commentf("Expected 4 ports bindings, got %d", len(cont.HostConfig.PortBindings)))

	for k, v := range cont.HostConfig.PortBindings {
		c.Assert(v, checker.HasLen, 1, check.Commentf("Expected 1 ports binding, for the port  %s but found %s", k, v))
		c.Assert(k.Port(), checker.Equals, v[0].HostPort, check.Commentf("Expected host port %s to match published port %s", k.Port(), v[0].HostPort))

	}

}

func (s *DockerSuite) TestCreateWithLargePortRange(c *check.C) {
	out, _ := dockerCmd(c, "create", "-p", "1-65535:1-65535/tcp", "busybox", "echo")

	cleanedContainerID := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "inspect", cleanedContainerID)

	containers := []struct {
		HostConfig *struct {
			PortBindings map[nat.Port][]nat.PortBinding
		}
	}{}

	err := json.Unmarshal([]byte(out), &containers)
	c.Assert(err, check.IsNil, check.Commentf("Error inspecting the container: %s", err))
	c.Assert(containers, checker.HasLen, 1)

	cont := containers[0]
	c.Assert(cont.HostConfig, check.NotNil, check.Commentf("Expected HostConfig, got none"))
	c.Assert(cont.HostConfig.PortBindings, checker.HasLen, 65535)

	for k, v := range cont.HostConfig.PortBindings {
		c.Assert(v, checker.HasLen, 1)
		c.Assert(k.Port(), checker.Equals, v[0].HostPort, check.Commentf("Expected host port %s to match published port %s", k.Port(), v[0].HostPort))
	}

}

// "test123" should be printed by docker create + start
func (s *DockerSuite) TestCreateEchoStdout(c *check.C) {
	out, _ := dockerCmd(c, "create", "busybox", "echo", "test123")

	cleanedContainerID := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "start", "-ai", cleanedContainerID)
	c.Assert(out, checker.Equals, "test123\n", check.Commentf("container should've printed 'test123', got %q", out))

}

func (s *DockerSuite) TestCreateVolumesCreated(c *check.C) {
	testRequires(c, SameHostDaemon)
	prefix, slash := getPrefixAndSlashFromDaemonPlatform()

	name := "test_create_volume"
	dockerCmd(c, "create", "--name", name, "-v", prefix+slash+"foo", "busybox")

	dir, err := inspectMountSourceField(name, prefix+slash+"foo")
	c.Assert(err, check.IsNil, check.Commentf("Error getting volume host path: %q", err))

	if _, err := os.Stat(dir); err != nil && os.IsNotExist(err) {
		c.Fatalf("Volume was not created")
	}
	if err != nil {
		c.Fatalf("Error statting volume host path: %q", err)
	}

}

func (s *DockerSuite) TestCreateLabels(c *check.C) {
	name := "test_create_labels"
	expected := map[string]string{"k1": "v1", "k2": "v2"}
	dockerCmd(c, "create", "--name", name, "-l", "k1=v1", "--label", "k2=v2", "busybox")

	actual := make(map[string]string)
	inspectFieldAndUnmarshall(c, name, "Config.Labels", &actual)

	if !reflect.DeepEqual(expected, actual) {
		c.Fatalf("Expected %s got %s", expected, actual)
	}
}

func (s *DockerSuite) TestCreateLabelFromImage(c *check.C) {
	imageName := "testcreatebuildlabel"
	buildImageSuccessfully(c, imageName, build.WithDockerfile(`FROM busybox
		LABEL k1=v1 k2=v2`))

	name := "test_create_labels_from_image"
	expected := map[string]string{"k2": "x", "k3": "v3", "k1": "v1"}
	dockerCmd(c, "create", "--name", name, "-l", "k2=x", "--label", "k3=v3", imageName)

	actual := make(map[string]string)
	inspectFieldAndUnmarshall(c, name, "Config.Labels", &actual)

	if !reflect.DeepEqual(expected, actual) {
		c.Fatalf("Expected %s got %s", expected, actual)
	}
}

func (s *DockerSuite) TestCreateHostnameWithNumber(c *check.C) {
	image := "busybox"
	// Busybox on Windows does not implement hostname command
	if testEnv.DaemonPlatform() == "windows" {
		image = testEnv.MinimalBaseImage()
	}
	out, _ := dockerCmd(c, "run", "-h", "web.0", image, "hostname")
	c.Assert(strings.TrimSpace(out), checker.Equals, "web.0", check.Commentf("hostname not set, expected `web.0`, got: %s", out))

}

func (s *DockerSuite) TestCreateRM(c *check.C) {
	// Test to make sure we can 'rm' a new container that is in
	// "Created" state, and has ever been run. Test "rm -f" too.

	// create a container
	out, _ := dockerCmd(c, "create", "busybox")
	cID := strings.TrimSpace(out)

	dockerCmd(c, "rm", cID)

	// Now do it again so we can "rm -f" this time
	out, _ = dockerCmd(c, "create", "busybox")

	cID = strings.TrimSpace(out)
	dockerCmd(c, "rm", "-f", cID)
}

func (s *DockerSuite) TestCreateModeIpcContainer(c *check.C) {
	// Uses Linux specific functionality (--ipc)
	testRequires(c, DaemonIsLinux, SameHostDaemon)

	out, _ := dockerCmd(c, "create", "busybox")
	id := strings.TrimSpace(out)

	dockerCmd(c, "create", fmt.Sprintf("--ipc=container:%s", id), "busybox")
}

func (s *DockerSuite) TestCreateByImageID(c *check.C) {
	imageName := "testcreatebyimageid"
	buildImageSuccessfully(c, imageName, build.WithDockerfile(`FROM busybox
		MAINTAINER dockerio`))
	imageID := getIDByName(c, imageName)
	truncatedImageID := stringid.TruncateID(imageID)

	dockerCmd(c, "create", imageID)
	dockerCmd(c, "create", truncatedImageID)
	dockerCmd(c, "create", fmt.Sprintf("%s:%s", imageName, truncatedImageID))

	// Ensure this fails
	out, exit, _ := dockerCmdWithError("create", fmt.Sprintf("%s:%s", imageName, imageID))
	if exit == 0 {
		c.Fatalf("expected non-zero exit code; received %d", exit)
	}

	if expected := "invalid reference format"; !strings.Contains(out, expected) {
		c.Fatalf(`Expected %q in output; got: %s`, expected, out)
	}

	out, exit, _ = dockerCmdWithError("create", fmt.Sprintf("%s:%s", "wrongimage", truncatedImageID))
	if exit == 0 {
		c.Fatalf("expected non-zero exit code; received %d", exit)
	}

	if expected := "Unable to find image"; !strings.Contains(out, expected) {
		c.Fatalf(`Expected %q in output; got: %s`, expected, out)
	}
}

func (s *DockerTrustSuite) TestTrustedCreate(c *check.C) {
	repoName := s.setupTrustedImage(c, "trusted-create")

	// Try create
	cli.Docker(cli.Args("create", repoName), trustedCmd).Assert(c, SuccessTagging)
	cli.DockerCmd(c, "rmi", repoName)

	// Try untrusted create to ensure we pushed the tag to the registry
	cli.Docker(cli.Args("create", "--disable-content-trust=true", repoName)).Assert(c, SuccessDownloadedOnStderr)
}

func (s *DockerTrustSuite) TestUntrustedCreate(c *check.C) {
	repoName := fmt.Sprintf("%v/dockercliuntrusted/createtest", privateRegistryURL)
	withTagName := fmt.Sprintf("%s:latest", repoName)
	// tag the image and upload it to the private registry
	cli.DockerCmd(c, "tag", "busybox", withTagName)
	cli.DockerCmd(c, "push", withTagName)
	cli.DockerCmd(c, "rmi", withTagName)

	// Try trusted create on untrusted tag
	cli.Docker(cli.Args("create", withTagName), trustedCmd).Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      fmt.Sprintf("does not have trust data for %s", repoName),
	})
}

func (s *DockerTrustSuite) TestTrustedIsolatedCreate(c *check.C) {
	repoName := s.setupTrustedImage(c, "trusted-isolated-create")

	// Try create
	cli.Docker(cli.Args("--config", "/tmp/docker-isolated-create", "create", repoName), trustedCmd).Assert(c, SuccessTagging)
	defer os.RemoveAll("/tmp/docker-isolated-create")

	cli.DockerCmd(c, "rmi", repoName)
}

func (s *DockerTrustSuite) TestTrustedCreateFromBadTrustServer(c *check.C) {
	repoName := fmt.Sprintf("%v/dockerclievilcreate/trusted:latest", privateRegistryURL)
	evilLocalConfigDir, err := ioutil.TempDir("", "evilcreate-local-config-dir")
	c.Assert(err, check.IsNil)

	// tag the image and upload it to the private registry
	cli.DockerCmd(c, "tag", "busybox", repoName)
	cli.Docker(cli.Args("push", repoName), trustedCmd).Assert(c, SuccessSigningAndPushing)
	cli.DockerCmd(c, "rmi", repoName)

	// Try create
	cli.Docker(cli.Args("create", repoName), trustedCmd).Assert(c, SuccessTagging)
	cli.DockerCmd(c, "rmi", repoName)

	// Kill the notary server, start a new "evil" one.
	s.not.Close()
	s.not, err = newTestNotary(c)
	c.Assert(err, check.IsNil)

	// In order to make an evil server, lets re-init a client (with a different trust dir) and push new data.
	// tag an image and upload it to the private registry
	cli.DockerCmd(c, "--config", evilLocalConfigDir, "tag", "busybox", repoName)

	// Push up to the new server
	cli.Docker(cli.Args("--config", evilLocalConfigDir, "push", repoName), trustedCmd).Assert(c, SuccessSigningAndPushing)

	// Now, try creating with the original client from this new trust server. This should fail because the new root is invalid.
	cli.Docker(cli.Args("create", repoName), trustedCmd).Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      "could not rotate trust to a new trusted root",
	})
}

func (s *DockerSuite) TestCreateStopSignal(c *check.C) {
	name := "test_create_stop_signal"
	dockerCmd(c, "create", "--name", name, "--stop-signal", "9", "busybox")

	res := inspectFieldJSON(c, name, "Config.StopSignal")
	c.Assert(res, checker.Contains, "9")

}

func (s *DockerSuite) TestCreateWithWorkdir(c *check.C) {
	name := "foo"

	prefix, slash := getPrefixAndSlashFromDaemonPlatform()
	dir := prefix + slash + "home" + slash + "foo" + slash + "bar"

	dockerCmd(c, "create", "--name", name, "-w", dir, "busybox")
	// Windows does not create the workdir until the container is started
	if testEnv.DaemonPlatform() == "windows" {
		dockerCmd(c, "start", name)
	}
	dockerCmd(c, "cp", fmt.Sprintf("%s:%s", name, dir), prefix+slash+"tmp")
}

func (s *DockerSuite) TestCreateWithInvalidLogOpts(c *check.C) {
	name := "test-invalidate-log-opts"
	out, _, err := dockerCmdWithError("create", "--name", name, "--log-opt", "invalid=true", "busybox")
	c.Assert(err, checker.NotNil)
	c.Assert(out, checker.Contains, "unknown log opt")

	out, _ = dockerCmd(c, "ps", "-a")
	c.Assert(out, checker.Not(checker.Contains), name)
}

// #20972
func (s *DockerSuite) TestCreate64ByteHexID(c *check.C) {
	out := inspectField(c, "busybox", "Id")
	imageID := strings.TrimPrefix(strings.TrimSpace(string(out)), "sha256:")

	dockerCmd(c, "create", imageID)
}

// Test case for #23498
func (s *DockerSuite) TestCreateUnsetEntrypoint(c *check.C) {
	name := "test-entrypoint"
	dockerfile := `FROM busybox
ADD entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD echo foobar`

	ctx := fakecontext.New(c, "",
		fakecontext.WithDockerfile(dockerfile),
		fakecontext.WithFiles(map[string]string{
			"entrypoint.sh": `#!/bin/sh
echo "I am an entrypoint"
exec "$@"`,
		}))
	defer ctx.Close()

	cli.BuildCmd(c, name, build.WithExternalBuildContext(ctx))

	out := cli.DockerCmd(c, "create", "--entrypoint=", name, "echo", "foo").Combined()
	id := strings.TrimSpace(out)
	c.Assert(id, check.Not(check.Equals), "")
	out = cli.DockerCmd(c, "start", "-a", id).Combined()
	c.Assert(strings.TrimSpace(out), check.Equals, "foo")
}

// #22471
func (s *DockerSuite) TestCreateStopTimeout(c *check.C) {
	name1 := "test_create_stop_timeout_1"
	dockerCmd(c, "create", "--name", name1, "--stop-timeout", "15", "busybox")

	res := inspectFieldJSON(c, name1, "Config.StopTimeout")
	c.Assert(res, checker.Contains, "15")

	name2 := "test_create_stop_timeout_2"
	dockerCmd(c, "create", "--name", name2, "busybox")

	res = inspectFieldJSON(c, name2, "Config.StopTimeout")
	c.Assert(res, checker.Contains, "null")
}
