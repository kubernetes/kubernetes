package main

import (
	"context"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/docker/docker/api/types"
	cliconfig "github.com/docker/docker/cli/config"
	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/cli"
	"github.com/docker/docker/integration-cli/fixtures/plugin"
	"github.com/docker/docker/integration-cli/request"
	icmd "github.com/docker/docker/pkg/testutil/cmd"
	"github.com/docker/go-connections/tlsconfig"
	"github.com/go-check/check"
)

var notaryBinary = "notary"
var notaryServerBinary = "notary-server"

type keyPair struct {
	Public  string
	Private string
}

type testNotary struct {
	cmd  *exec.Cmd
	dir  string
	keys []keyPair
}

const notaryHost = "localhost:4443"
const notaryURL = "https://" + notaryHost

var SuccessTagging = icmd.Expected{
	Out: "Tagging",
}

var SuccessSigningAndPushing = icmd.Expected{
	Out: "Signing and pushing trust metadata",
}

var SuccessDownloaded = icmd.Expected{
	Out: "Status: Downloaded",
}

var SuccessDownloadedOnStderr = icmd.Expected{
	Err: "Status: Downloaded",
}

func newTestNotary(c *check.C) (*testNotary, error) {
	// generate server config
	template := `{
	"server": {
		"http_addr": "%s",
		"tls_key_file": "%s",
		"tls_cert_file": "%s"
	},
	"trust_service": {
		"type": "local",
		"hostname": "",
		"port": "",
		"key_algorithm": "ed25519"
	},
	"logging": {
		"level": "debug"
	},
	"storage": {
        "backend": "memory"
    }
}`
	tmp, err := ioutil.TempDir("", "notary-test-")
	if err != nil {
		return nil, err
	}
	confPath := filepath.Join(tmp, "config.json")
	config, err := os.Create(confPath)
	if err != nil {
		return nil, err
	}
	defer config.Close()

	workingDir, err := os.Getwd()
	if err != nil {
		return nil, err
	}
	if _, err := fmt.Fprintf(config, template, notaryHost, filepath.Join(workingDir, "fixtures/notary/localhost.key"), filepath.Join(workingDir, "fixtures/notary/localhost.cert")); err != nil {
		os.RemoveAll(tmp)
		return nil, err
	}

	// generate client config
	clientConfPath := filepath.Join(tmp, "client-config.json")
	clientConfig, err := os.Create(clientConfPath)
	if err != nil {
		return nil, err
	}
	defer clientConfig.Close()

	template = `{
	"trust_dir" : "%s",
	"remote_server": {
		"url": "%s",
		"skipTLSVerify": true
	}
}`
	if _, err = fmt.Fprintf(clientConfig, template, filepath.Join(cliconfig.Dir(), "trust"), notaryURL); err != nil {
		os.RemoveAll(tmp)
		return nil, err
	}

	// load key fixture filenames
	var keys []keyPair
	for i := 1; i < 5; i++ {
		keys = append(keys, keyPair{
			Public:  filepath.Join(workingDir, fmt.Sprintf("fixtures/notary/delgkey%v.crt", i)),
			Private: filepath.Join(workingDir, fmt.Sprintf("fixtures/notary/delgkey%v.key", i)),
		})
	}

	// run notary-server
	cmd := exec.Command(notaryServerBinary, "-config", confPath)
	if err := cmd.Start(); err != nil {
		os.RemoveAll(tmp)
		if os.IsNotExist(err) {
			c.Skip(err.Error())
		}
		return nil, err
	}

	testNotary := &testNotary{
		cmd:  cmd,
		dir:  tmp,
		keys: keys,
	}

	// Wait for notary to be ready to serve requests.
	for i := 1; i <= 20; i++ {
		if err = testNotary.Ping(); err == nil {
			break
		}
		time.Sleep(10 * time.Millisecond * time.Duration(i*i))
	}

	if err != nil {
		c.Fatalf("Timeout waiting for test notary to become available: %s", err)
	}

	return testNotary, nil
}

func (t *testNotary) Ping() error {
	tlsConfig := tlsconfig.ClientDefault()
	tlsConfig.InsecureSkipVerify = true
	client := http.Client{
		Transport: &http.Transport{
			Proxy: http.ProxyFromEnvironment,
			Dial: (&net.Dialer{
				Timeout:   30 * time.Second,
				KeepAlive: 30 * time.Second,
			}).Dial,
			TLSHandshakeTimeout: 10 * time.Second,
			TLSClientConfig:     tlsConfig,
		},
	}
	resp, err := client.Get(fmt.Sprintf("%s/v2/", notaryURL))
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("notary ping replied with an unexpected status code %d", resp.StatusCode)
	}
	return nil
}

func (t *testNotary) Close() {
	t.cmd.Process.Kill()
	t.cmd.Process.Wait()
	os.RemoveAll(t.dir)
}

func trustedCmd(cmd *icmd.Cmd) func() {
	pwd := "12345678"
	cmd.Env = append(cmd.Env, trustEnv(notaryURL, pwd, pwd)...)
	return nil
}

func trustedCmdWithServer(server string) func(*icmd.Cmd) func() {
	return func(cmd *icmd.Cmd) func() {
		pwd := "12345678"
		cmd.Env = append(cmd.Env, trustEnv(server, pwd, pwd)...)
		return nil
	}
}

func trustedCmdWithPassphrases(rootPwd, repositoryPwd string) func(*icmd.Cmd) func() {
	return func(cmd *icmd.Cmd) func() {
		cmd.Env = append(cmd.Env, trustEnv(notaryURL, rootPwd, repositoryPwd)...)
		return nil
	}
}

func trustEnv(server, rootPwd, repositoryPwd string) []string {
	env := append(os.Environ(), []string{
		"DOCKER_CONTENT_TRUST=1",
		fmt.Sprintf("DOCKER_CONTENT_TRUST_SERVER=%s", server),
		fmt.Sprintf("DOCKER_CONTENT_TRUST_ROOT_PASSPHRASE=%s", rootPwd),
		fmt.Sprintf("DOCKER_CONTENT_TRUST_REPOSITORY_PASSPHRASE=%s", repositoryPwd),
	}...)
	return env
}

func (s *DockerTrustSuite) setupTrustedImage(c *check.C, name string) string {
	repoName := fmt.Sprintf("%v/dockercli/%s:latest", privateRegistryURL, name)
	// tag the image and upload it to the private registry
	cli.DockerCmd(c, "tag", "busybox", repoName)
	cli.Docker(cli.Args("push", repoName), trustedCmd).Assert(c, SuccessSigningAndPushing)
	cli.DockerCmd(c, "rmi", repoName)
	return repoName
}

func (s *DockerTrustSuite) setupTrustedplugin(c *check.C, source, name string) string {
	repoName := fmt.Sprintf("%v/dockercli/%s:latest", privateRegistryURL, name)

	client, err := request.NewClient()
	c.Assert(err, checker.IsNil, check.Commentf("could not create test client"))

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	err = plugin.Create(ctx, client, repoName)
	cancel()
	c.Assert(err, checker.IsNil, check.Commentf("could not create test plugin"))

	// tag the image and upload it to the private registry
	// TODO: shouldn't need to use the CLI to do trust
	cli.Docker(cli.Args("plugin", "push", repoName), trustedCmd).Assert(c, SuccessSigningAndPushing)

	ctx, cancel = context.WithTimeout(context.Background(), 60*time.Second)
	err = client.PluginRemove(ctx, repoName, types.PluginRemoveOptions{Force: true})
	cancel()
	c.Assert(err, checker.IsNil, check.Commentf("failed to cleanup test plugin for trust suite"))
	return repoName
}

func (s *DockerTrustSuite) notaryCmd(c *check.C, args ...string) string {
	pwd := "12345678"
	env := []string{
		fmt.Sprintf("NOTARY_ROOT_PASSPHRASE=%s", pwd),
		fmt.Sprintf("NOTARY_TARGETS_PASSPHRASE=%s", pwd),
		fmt.Sprintf("NOTARY_SNAPSHOT_PASSPHRASE=%s", pwd),
		fmt.Sprintf("NOTARY_DELEGATION_PASSPHRASE=%s", pwd),
	}
	result := icmd.RunCmd(icmd.Cmd{
		Command: append([]string{notaryBinary, "-c", filepath.Join(s.not.dir, "client-config.json")}, args...),
		Env:     append(os.Environ(), env...),
	})
	result.Assert(c, icmd.Success)
	return result.Combined()
}

func (s *DockerTrustSuite) notaryInitRepo(c *check.C, repoName string) {
	s.notaryCmd(c, "init", repoName)
}

func (s *DockerTrustSuite) notaryCreateDelegation(c *check.C, repoName, role string, pubKey string, paths ...string) {
	pathsArg := "--all-paths"
	if len(paths) > 0 {
		pathsArg = "--paths=" + strings.Join(paths, ",")
	}

	s.notaryCmd(c, "delegation", "add", repoName, role, pubKey, pathsArg)
}

func (s *DockerTrustSuite) notaryPublish(c *check.C, repoName string) {
	s.notaryCmd(c, "publish", repoName)
}

func (s *DockerTrustSuite) notaryImportKey(c *check.C, repoName, role string, privKey string) {
	s.notaryCmd(c, "key", "import", privKey, "-g", repoName, "-r", role)
}

func (s *DockerTrustSuite) notaryListTargetsInRole(c *check.C, repoName, role string) map[string]string {
	out := s.notaryCmd(c, "list", repoName, "-r", role)

	// should look something like:
	//    NAME                                 DIGEST                                SIZE (BYTES)    ROLE
	// ------------------------------------------------------------------------------------------------------
	//   latest   24a36bbc059b1345b7e8be0df20f1b23caa3602e85d42fff7ecd9d0bd255de56   1377           targets

	targets := make(map[string]string)

	// no target
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) == 1 && strings.Contains(out, "No targets present in this repository.") {
		return targets
	}

	// otherwise, there is at least one target
	c.Assert(len(lines), checker.GreaterOrEqualThan, 3)

	for _, line := range lines[2:] {
		tokens := strings.Fields(line)
		c.Assert(tokens, checker.HasLen, 4)
		targets[tokens[0]] = tokens[3]
	}

	return targets
}

func (s *DockerTrustSuite) assertTargetInRoles(c *check.C, repoName, target string, roles ...string) {
	// check all the roles
	for _, role := range roles {
		targets := s.notaryListTargetsInRole(c, repoName, role)
		roleName, ok := targets[target]
		c.Assert(ok, checker.True)
		c.Assert(roleName, checker.Equals, role)
	}
}

func (s *DockerTrustSuite) assertTargetNotInRoles(c *check.C, repoName, target string, roles ...string) {
	targets := s.notaryListTargetsInRole(c, repoName, "targets")

	roleName, ok := targets[target]
	if ok {
		for _, role := range roles {
			c.Assert(roleName, checker.Not(checker.Equals), role)
		}
	}
}
