package main

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/cli"
	"github.com/docker/docker/integration-cli/daemon"
	"github.com/docker/docker/integration-cli/fixtures/plugin"
	"github.com/docker/docker/integration-cli/request"
	icmd "github.com/docker/docker/pkg/testutil/cmd"
	"github.com/go-check/check"
	"golang.org/x/net/context"
)

var (
	pluginProcessName = "sample-volume-plugin"
	pName             = "tiborvass/sample-volume-plugin"
	npName            = "tiborvass/test-docker-netplugin"
	pTag              = "latest"
	pNameWithTag      = pName + ":" + pTag
	npNameWithTag     = npName + ":" + pTag
)

func (ps *DockerPluginSuite) TestPluginBasicOps(c *check.C) {
	plugin := ps.getPluginRepoWithTag()
	_, _, err := dockerCmdWithError("plugin", "install", "--grant-all-permissions", plugin)
	c.Assert(err, checker.IsNil)

	out, _, err := dockerCmdWithError("plugin", "ls")
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Contains, plugin)
	c.Assert(out, checker.Contains, "true")

	id, _, err := dockerCmdWithError("plugin", "inspect", "-f", "{{.Id}}", plugin)
	id = strings.TrimSpace(id)
	c.Assert(err, checker.IsNil)

	out, _, err = dockerCmdWithError("plugin", "remove", plugin)
	c.Assert(err, checker.NotNil)
	c.Assert(out, checker.Contains, "is enabled")

	_, _, err = dockerCmdWithError("plugin", "disable", plugin)
	c.Assert(err, checker.IsNil)

	out, _, err = dockerCmdWithError("plugin", "remove", plugin)
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Contains, plugin)

	_, err = os.Stat(filepath.Join(testEnv.DockerBasePath(), "plugins", id))
	if !os.IsNotExist(err) {
		c.Fatal(err)
	}
}

func (ps *DockerPluginSuite) TestPluginForceRemove(c *check.C) {
	pNameWithTag := ps.getPluginRepoWithTag()

	out, _, err := dockerCmdWithError("plugin", "install", "--grant-all-permissions", pNameWithTag)
	c.Assert(err, checker.IsNil)

	out, _, err = dockerCmdWithError("plugin", "remove", pNameWithTag)
	c.Assert(out, checker.Contains, "is enabled")

	out, _, err = dockerCmdWithError("plugin", "remove", "--force", pNameWithTag)
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Contains, pNameWithTag)
}

func (s *DockerSuite) TestPluginActive(c *check.C) {
	testRequires(c, DaemonIsLinux, IsAmd64, Network)

	_, _, err := dockerCmdWithError("plugin", "install", "--grant-all-permissions", pNameWithTag)
	c.Assert(err, checker.IsNil)

	_, _, err = dockerCmdWithError("volume", "create", "-d", pNameWithTag, "--name", "testvol1")
	c.Assert(err, checker.IsNil)

	out, _, err := dockerCmdWithError("plugin", "disable", pNameWithTag)
	c.Assert(out, checker.Contains, "in use")

	_, _, err = dockerCmdWithError("volume", "rm", "testvol1")
	c.Assert(err, checker.IsNil)

	_, _, err = dockerCmdWithError("plugin", "disable", pNameWithTag)
	c.Assert(err, checker.IsNil)

	out, _, err = dockerCmdWithError("plugin", "remove", pNameWithTag)
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Contains, pNameWithTag)
}

func (s *DockerSuite) TestPluginActiveNetwork(c *check.C) {
	testRequires(c, DaemonIsLinux, IsAmd64, Network)
	out, _, err := dockerCmdWithError("plugin", "install", "--grant-all-permissions", npNameWithTag)
	c.Assert(err, checker.IsNil)

	out, _, err = dockerCmdWithError("network", "create", "-d", npNameWithTag, "test")
	c.Assert(err, checker.IsNil)

	nID := strings.TrimSpace(out)

	out, _, err = dockerCmdWithError("plugin", "remove", npNameWithTag)
	c.Assert(out, checker.Contains, "is in use")

	_, _, err = dockerCmdWithError("network", "rm", nID)
	c.Assert(err, checker.IsNil)

	out, _, err = dockerCmdWithError("plugin", "remove", npNameWithTag)
	c.Assert(out, checker.Contains, "is enabled")

	_, _, err = dockerCmdWithError("plugin", "disable", npNameWithTag)
	c.Assert(err, checker.IsNil)

	out, _, err = dockerCmdWithError("plugin", "remove", npNameWithTag)
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Contains, npNameWithTag)
}

func (ps *DockerPluginSuite) TestPluginInstallDisable(c *check.C) {
	pName := ps.getPluginRepoWithTag()

	out, _, err := dockerCmdWithError("plugin", "install", "--grant-all-permissions", "--disable", pName)
	c.Assert(err, checker.IsNil)
	c.Assert(strings.TrimSpace(out), checker.Contains, pName)

	out, _, err = dockerCmdWithError("plugin", "ls")
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Contains, "false")

	out, _, err = dockerCmdWithError("plugin", "enable", pName)
	c.Assert(err, checker.IsNil)
	c.Assert(strings.TrimSpace(out), checker.Contains, pName)

	out, _, err = dockerCmdWithError("plugin", "disable", pName)
	c.Assert(err, checker.IsNil)
	c.Assert(strings.TrimSpace(out), checker.Contains, pName)

	out, _, err = dockerCmdWithError("plugin", "remove", pName)
	c.Assert(err, checker.IsNil)
	c.Assert(strings.TrimSpace(out), checker.Contains, pName)
}

func (s *DockerSuite) TestPluginInstallDisableVolumeLs(c *check.C) {
	testRequires(c, DaemonIsLinux, IsAmd64, Network)
	out, _, err := dockerCmdWithError("plugin", "install", "--grant-all-permissions", "--disable", pName)
	c.Assert(err, checker.IsNil)
	c.Assert(strings.TrimSpace(out), checker.Contains, pName)

	dockerCmd(c, "volume", "ls")
}

func (ps *DockerPluginSuite) TestPluginSet(c *check.C) {
	// Create a new plugin with extra settings
	client, err := request.NewClient()
	c.Assert(err, checker.IsNil, check.Commentf("failed to create test client"))

	name := "test"
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	initialValue := "0"
	err = plugin.Create(ctx, client, name, func(cfg *plugin.Config) {
		cfg.Env = []types.PluginEnv{{Name: "DEBUG", Value: &initialValue, Settable: []string{"value"}}}
	})
	c.Assert(err, checker.IsNil, check.Commentf("failed to create test plugin"))

	env, _ := dockerCmd(c, "plugin", "inspect", "-f", "{{.Settings.Env}}", name)
	c.Assert(strings.TrimSpace(env), checker.Equals, "[DEBUG=0]")

	dockerCmd(c, "plugin", "set", name, "DEBUG=1")

	env, _ = dockerCmd(c, "plugin", "inspect", "-f", "{{.Settings.Env}}", name)
	c.Assert(strings.TrimSpace(env), checker.Equals, "[DEBUG=1]")
}

func (ps *DockerPluginSuite) TestPluginInstallArgs(c *check.C) {
	pName := path.Join(ps.registryHost(), "plugin", "testplugininstallwithargs")
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	plugin.CreateInRegistry(ctx, pName, nil, func(cfg *plugin.Config) {
		cfg.Env = []types.PluginEnv{{Name: "DEBUG", Settable: []string{"value"}}}
	})

	out, _ := dockerCmd(c, "plugin", "install", "--grant-all-permissions", "--disable", pName, "DEBUG=1")
	c.Assert(strings.TrimSpace(out), checker.Contains, pName)

	env, _ := dockerCmd(c, "plugin", "inspect", "-f", "{{.Settings.Env}}", pName)
	c.Assert(strings.TrimSpace(env), checker.Equals, "[DEBUG=1]")
}

func (ps *DockerPluginSuite) TestPluginInstallImage(c *check.C) {
	testRequires(c, IsAmd64)

	repoName := fmt.Sprintf("%v/dockercli/busybox", privateRegistryURL)
	// tag the image to upload it to the private registry
	dockerCmd(c, "tag", "busybox", repoName)
	// push the image to the registry
	dockerCmd(c, "push", repoName)

	out, _, err := dockerCmdWithError("plugin", "install", repoName)
	c.Assert(err, checker.NotNil)
	c.Assert(out, checker.Contains, `Encountered remote "application/vnd.docker.container.image.v1+json"(image) when fetching`)
}

func (ps *DockerPluginSuite) TestPluginEnableDisableNegative(c *check.C) {
	pName := ps.getPluginRepoWithTag()

	out, _, err := dockerCmdWithError("plugin", "install", "--grant-all-permissions", pName)
	c.Assert(err, checker.IsNil)
	c.Assert(strings.TrimSpace(out), checker.Contains, pName)

	out, _, err = dockerCmdWithError("plugin", "enable", pName)
	c.Assert(err, checker.NotNil)
	c.Assert(strings.TrimSpace(out), checker.Contains, "already enabled")

	_, _, err = dockerCmdWithError("plugin", "disable", pName)
	c.Assert(err, checker.IsNil)

	out, _, err = dockerCmdWithError("plugin", "disable", pName)
	c.Assert(err, checker.NotNil)
	c.Assert(strings.TrimSpace(out), checker.Contains, "already disabled")

	_, _, err = dockerCmdWithError("plugin", "remove", pName)
	c.Assert(err, checker.IsNil)
}

func (ps *DockerPluginSuite) TestPluginCreate(c *check.C) {
	name := "foo/bar-driver"
	temp, err := ioutil.TempDir("", "foo")
	c.Assert(err, checker.IsNil)
	defer os.RemoveAll(temp)

	data := `{"description": "foo plugin"}`
	err = ioutil.WriteFile(filepath.Join(temp, "config.json"), []byte(data), 0644)
	c.Assert(err, checker.IsNil)

	err = os.MkdirAll(filepath.Join(temp, "rootfs"), 0700)
	c.Assert(err, checker.IsNil)

	out, _, err := dockerCmdWithError("plugin", "create", name, temp)
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Contains, name)

	out, _, err = dockerCmdWithError("plugin", "ls")
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Contains, name)

	out, _, err = dockerCmdWithError("plugin", "create", name, temp)
	c.Assert(err, checker.NotNil)
	c.Assert(out, checker.Contains, "already exist")

	out, _, err = dockerCmdWithError("plugin", "ls")
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Contains, name)
	// The output will consists of one HEADER line and one line of foo/bar-driver
	c.Assert(len(strings.Split(strings.TrimSpace(out), "\n")), checker.Equals, 2)
}

func (ps *DockerPluginSuite) TestPluginInspect(c *check.C) {
	pNameWithTag := ps.getPluginRepoWithTag()

	_, _, err := dockerCmdWithError("plugin", "install", "--grant-all-permissions", pNameWithTag)
	c.Assert(err, checker.IsNil)

	out, _, err := dockerCmdWithError("plugin", "ls")
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Contains, pNameWithTag)
	c.Assert(out, checker.Contains, "true")

	// Find the ID first
	out, _, err = dockerCmdWithError("plugin", "inspect", "-f", "{{.Id}}", pNameWithTag)
	c.Assert(err, checker.IsNil)
	id := strings.TrimSpace(out)
	c.Assert(id, checker.Not(checker.Equals), "")

	// Long form
	out, _, err = dockerCmdWithError("plugin", "inspect", "-f", "{{.Id}}", id)
	c.Assert(err, checker.IsNil)
	c.Assert(strings.TrimSpace(out), checker.Equals, id)

	// Short form
	out, _, err = dockerCmdWithError("plugin", "inspect", "-f", "{{.Id}}", id[:5])
	c.Assert(err, checker.IsNil)
	c.Assert(strings.TrimSpace(out), checker.Equals, id)

	// Name with tag form
	out, _, err = dockerCmdWithError("plugin", "inspect", "-f", "{{.Id}}", pNameWithTag)
	c.Assert(err, checker.IsNil)
	c.Assert(strings.TrimSpace(out), checker.Equals, id)

	// Name without tag form
	out, _, err = dockerCmdWithError("plugin", "inspect", "-f", "{{.Id}}", ps.getPluginRepo())
	c.Assert(err, checker.IsNil)
	c.Assert(strings.TrimSpace(out), checker.Equals, id)

	_, _, err = dockerCmdWithError("plugin", "disable", pNameWithTag)
	c.Assert(err, checker.IsNil)

	out, _, err = dockerCmdWithError("plugin", "remove", pNameWithTag)
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Contains, pNameWithTag)

	// After remove nothing should be found
	_, _, err = dockerCmdWithError("plugin", "inspect", "-f", "{{.Id}}", id[:5])
	c.Assert(err, checker.NotNil)
}

// Test case for https://github.com/docker/docker/pull/29186#discussion_r91277345
func (s *DockerSuite) TestPluginInspectOnWindows(c *check.C) {
	// This test should work on Windows only
	testRequires(c, DaemonIsWindows)

	out, _, err := dockerCmdWithError("plugin", "inspect", "foobar")
	c.Assert(err, checker.NotNil)
	c.Assert(out, checker.Contains, "plugins are not supported on this platform")
	c.Assert(err.Error(), checker.Contains, "plugins are not supported on this platform")
}

func (s *DockerTrustSuite) TestPluginTrustedInstall(c *check.C) {
	testRequires(c, DaemonIsLinux, IsAmd64, Network)

	trustedName := s.setupTrustedplugin(c, pNameWithTag, "trusted-plugin-install")

	cli.Docker(cli.Args("plugin", "install", "--grant-all-permissions", trustedName), trustedCmd).Assert(c, icmd.Expected{
		Out: trustedName,
	})

	out := cli.DockerCmd(c, "plugin", "ls").Combined()
	c.Assert(out, checker.Contains, "true")

	out = cli.DockerCmd(c, "plugin", "disable", trustedName).Combined()
	c.Assert(strings.TrimSpace(out), checker.Contains, trustedName)

	out = cli.DockerCmd(c, "plugin", "enable", trustedName).Combined()
	c.Assert(strings.TrimSpace(out), checker.Contains, trustedName)

	out = cli.DockerCmd(c, "plugin", "rm", "-f", trustedName).Combined()
	c.Assert(strings.TrimSpace(out), checker.Contains, trustedName)

	// Try untrusted pull to ensure we pushed the tag to the registry
	cli.Docker(cli.Args("plugin", "install", "--disable-content-trust=true", "--grant-all-permissions", trustedName), trustedCmd).Assert(c, SuccessDownloaded)

	out = cli.DockerCmd(c, "plugin", "ls").Combined()
	c.Assert(out, checker.Contains, "true")

}

func (s *DockerTrustSuite) TestPluginUntrustedInstall(c *check.C) {
	testRequires(c, DaemonIsLinux, IsAmd64, Network)

	pluginName := fmt.Sprintf("%v/dockercliuntrusted/plugintest:latest", privateRegistryURL)
	// install locally and push to private registry
	cli.DockerCmd(c, "plugin", "install", "--grant-all-permissions", "--alias", pluginName, pNameWithTag)
	cli.DockerCmd(c, "plugin", "push", pluginName)
	cli.DockerCmd(c, "plugin", "rm", "-f", pluginName)

	// Try trusted install on untrusted plugin
	cli.Docker(cli.Args("plugin", "install", "--grant-all-permissions", pluginName), trustedCmd).Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      "Error: remote trust data does not exist",
	})
}

func (ps *DockerPluginSuite) TestPluginIDPrefix(c *check.C) {
	name := "test"
	client, err := request.NewClient()
	c.Assert(err, checker.IsNil, check.Commentf("error creating test client"))

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	initialValue := "0"
	err = plugin.Create(ctx, client, name, func(cfg *plugin.Config) {
		cfg.Env = []types.PluginEnv{{Name: "DEBUG", Value: &initialValue, Settable: []string{"value"}}}
	})
	cancel()

	c.Assert(err, checker.IsNil, check.Commentf("failed to create test plugin"))

	// Find ID first
	id, _, err := dockerCmdWithError("plugin", "inspect", "-f", "{{.Id}}", name)
	id = strings.TrimSpace(id)
	c.Assert(err, checker.IsNil)

	// List current state
	out, _, err := dockerCmdWithError("plugin", "ls")
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Contains, name)
	c.Assert(out, checker.Contains, "false")

	env, _ := dockerCmd(c, "plugin", "inspect", "-f", "{{.Settings.Env}}", id[:5])
	c.Assert(strings.TrimSpace(env), checker.Equals, "[DEBUG=0]")

	dockerCmd(c, "plugin", "set", id[:5], "DEBUG=1")

	env, _ = dockerCmd(c, "plugin", "inspect", "-f", "{{.Settings.Env}}", id[:5])
	c.Assert(strings.TrimSpace(env), checker.Equals, "[DEBUG=1]")

	// Enable
	_, _, err = dockerCmdWithError("plugin", "enable", id[:5])
	c.Assert(err, checker.IsNil)
	out, _, err = dockerCmdWithError("plugin", "ls")
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Contains, name)
	c.Assert(out, checker.Contains, "true")

	// Disable
	_, _, err = dockerCmdWithError("plugin", "disable", id[:5])
	c.Assert(err, checker.IsNil)
	out, _, err = dockerCmdWithError("plugin", "ls")
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Contains, name)
	c.Assert(out, checker.Contains, "false")

	// Remove
	out, _, err = dockerCmdWithError("plugin", "remove", id[:5])
	c.Assert(err, checker.IsNil)
	// List returns none
	out, _, err = dockerCmdWithError("plugin", "ls")
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Not(checker.Contains), name)
}

func (ps *DockerPluginSuite) TestPluginListDefaultFormat(c *check.C) {
	config, err := ioutil.TempDir("", "config-file-")
	c.Assert(err, check.IsNil)
	defer os.RemoveAll(config)

	err = ioutil.WriteFile(filepath.Join(config, "config.json"), []byte(`{"pluginsFormat": "raw"}`), 0644)
	c.Assert(err, check.IsNil)

	name := "test:latest"
	client, err := request.NewClient()
	c.Assert(err, checker.IsNil, check.Commentf("error creating test client"))

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	err = plugin.Create(ctx, client, name, func(cfg *plugin.Config) {
		cfg.Description = "test plugin"
	})
	c.Assert(err, checker.IsNil, check.Commentf("failed to create test plugin"))

	out, _ := dockerCmd(c, "plugin", "inspect", "--format", "{{.ID}}", name)
	id := strings.TrimSpace(out)

	// We expect the format to be in `raw + --no-trunc`
	expectedOutput := fmt.Sprintf(`plugin_id: %s
name: %s
description: test plugin
enabled: false`, id, name)

	out, _ = dockerCmd(c, "--config", config, "plugin", "ls", "--no-trunc")
	c.Assert(strings.TrimSpace(out), checker.Contains, expectedOutput)
}

func (s *DockerSuite) TestPluginUpgrade(c *check.C) {
	testRequires(c, DaemonIsLinux, Network, SameHostDaemon, IsAmd64)
	plugin := "cpuguy83/docker-volume-driver-plugin-local:latest"
	pluginV2 := "cpuguy83/docker-volume-driver-plugin-local:v2"

	dockerCmd(c, "plugin", "install", "--grant-all-permissions", plugin)
	dockerCmd(c, "volume", "create", "--driver", plugin, "bananas")
	dockerCmd(c, "run", "--rm", "-v", "bananas:/apple", "busybox", "sh", "-c", "touch /apple/core")

	out, _, err := dockerCmdWithError("plugin", "upgrade", "--grant-all-permissions", plugin, pluginV2)
	c.Assert(err, checker.NotNil, check.Commentf(out))
	c.Assert(out, checker.Contains, "disabled before upgrading")

	out, _ = dockerCmd(c, "plugin", "inspect", "--format={{.ID}}", plugin)
	id := strings.TrimSpace(out)

	// make sure "v2" does not exists
	_, err = os.Stat(filepath.Join(testEnv.DockerBasePath(), "plugins", id, "rootfs", "v2"))
	c.Assert(os.IsNotExist(err), checker.True, check.Commentf(out))

	dockerCmd(c, "plugin", "disable", "-f", plugin)
	dockerCmd(c, "plugin", "upgrade", "--grant-all-permissions", "--skip-remote-check", plugin, pluginV2)

	// make sure "v2" file exists
	_, err = os.Stat(filepath.Join(testEnv.DockerBasePath(), "plugins", id, "rootfs", "v2"))
	c.Assert(err, checker.IsNil)

	dockerCmd(c, "plugin", "enable", plugin)
	dockerCmd(c, "volume", "inspect", "bananas")
	dockerCmd(c, "run", "--rm", "-v", "bananas:/apple", "busybox", "sh", "-c", "ls -lh /apple/core")
}

func (s *DockerSuite) TestPluginMetricsCollector(c *check.C) {
	testRequires(c, DaemonIsLinux, Network, SameHostDaemon, IsAmd64)
	d := daemon.New(c, dockerBinary, dockerdBinary, daemon.Config{})
	d.Start(c)
	defer d.Stop(c)

	name := "cpuguy83/docker-metrics-plugin-test:latest"
	r := cli.Docker(cli.Args("plugin", "install", "--grant-all-permissions", name), cli.Daemon(d))
	c.Assert(r.Error, checker.IsNil, check.Commentf(r.Combined()))

	// plugin lisens on localhost:19393 and proxies the metrics
	resp, err := http.Get("http://localhost:19393/metrics")
	c.Assert(err, checker.IsNil)
	defer resp.Body.Close()

	b, err := ioutil.ReadAll(resp.Body)
	c.Assert(err, checker.IsNil)
	// check that a known metric is there... don't expect this metric to change over time.. probably safe
	c.Assert(string(b), checker.Contains, "container_actions")
}
