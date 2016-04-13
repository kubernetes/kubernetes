package main

import (
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"

	"github.com/docker/docker/autogen/dockerversion"
	"github.com/docker/docker/pkg/homedir"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestConfigHttpHeader(c *check.C) {
	testRequires(c, UnixCli) // Can't set/unset HOME on windows right now
	// We either need a level of Go that supports Unsetenv (for cases
	// when HOME/USERPROFILE isn't set), or we need to be able to use
	// os/user but user.Current() only works if we aren't statically compiling

	var headers map[string][]string

	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			headers = r.Header
		}))
	defer server.Close()

	homeKey := homedir.Key()
	homeVal := homedir.Get()
	tmpDir, _ := ioutil.TempDir("", "fake-home")
	defer os.RemoveAll(tmpDir)

	dotDocker := filepath.Join(tmpDir, ".docker")
	os.Mkdir(dotDocker, 0600)
	tmpCfg := filepath.Join(dotDocker, "config.json")

	defer func() { os.Setenv(homeKey, homeVal) }()
	os.Setenv(homeKey, tmpDir)

	data := `{
		"HttpHeaders": { "MyHeader": "MyValue" }
	}`

	err := ioutil.WriteFile(tmpCfg, []byte(data), 0600)
	if err != nil {
		c.Fatalf("Err creating file(%s): %v", tmpCfg, err)
	}

	cmd := exec.Command(dockerBinary, "-H="+server.URL[7:], "ps")
	out, _, _ := runCommandWithOutput(cmd)

	if headers["User-Agent"] == nil {
		c.Fatalf("Missing User-Agent: %q\nout:%v", headers, out)
	}

	if headers["User-Agent"][0] != "Docker-Client/"+dockerversion.VERSION+" ("+runtime.GOOS+")" {
		c.Fatalf("Badly formatted User-Agent: %q\nout:%v", headers, out)
	}

	if headers["Myheader"] == nil || headers["Myheader"][0] != "MyValue" {
		c.Fatalf("Missing/bad header: %q\nout:%v", headers, out)
	}
}

func (s *DockerSuite) TestConfigDir(c *check.C) {
	cDir, _ := ioutil.TempDir("", "fake-home")

	// First make sure pointing to empty dir doesn't generate an error
	out, rc := dockerCmd(c, "--config", cDir, "ps")

	if rc != 0 {
		c.Fatalf("ps1 didn't work:\nrc:%d\nout%s", rc, out)
	}

	// Test with env var too
	cmd := exec.Command(dockerBinary, "ps")
	cmd.Env = append(os.Environ(), "DOCKER_CONFIG="+cDir)
	out, rc, err := runCommandWithOutput(cmd)

	if rc != 0 || err != nil {
		c.Fatalf("ps2 didn't work:\nrc:%d\nout%s\nerr:%v", rc, out, err)
	}

	// Start a server so we can check to see if the config file was
	// loaded properly
	var headers map[string][]string

	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			headers = r.Header
		}))
	defer server.Close()

	// Create a dummy config file in our new config dir
	data := `{
		"HttpHeaders": { "MyHeader": "MyValue" }
	}`

	tmpCfg := filepath.Join(cDir, "config.json")
	err = ioutil.WriteFile(tmpCfg, []byte(data), 0600)
	if err != nil {
		c.Fatalf("Err creating file(%s): %v", tmpCfg, err)
	}

	cmd = exec.Command(dockerBinary, "--config", cDir, "-H="+server.URL[7:], "ps")
	out, _, _ = runCommandWithOutput(cmd)

	if headers["Myheader"] == nil || headers["Myheader"][0] != "MyValue" {
		c.Fatalf("ps3 - Missing header: %q\nout:%v", headers, out)
	}

	// Reset headers and try again using env var this time
	headers = map[string][]string{}
	cmd = exec.Command(dockerBinary, "-H="+server.URL[7:], "ps")
	cmd.Env = append(os.Environ(), "DOCKER_CONFIG="+cDir)
	out, _, _ = runCommandWithOutput(cmd)

	if headers["Myheader"] == nil || headers["Myheader"][0] != "MyValue" {
		c.Fatalf("ps4 - Missing header: %q\nout:%v", headers, out)
	}

	// Reset headers and make sure flag overrides the env var
	headers = map[string][]string{}
	cmd = exec.Command(dockerBinary, "--config", cDir, "-H="+server.URL[7:], "ps")
	cmd.Env = append(os.Environ(), "DOCKER_CONFIG=MissingDir")
	out, _, _ = runCommandWithOutput(cmd)

	if headers["Myheader"] == nil || headers["Myheader"][0] != "MyValue" {
		c.Fatalf("ps5 - Missing header: %q\nout:%v", headers, out)
	}

	// Reset headers and make sure flag overrides the env var.
	// Almost same as previous but make sure the "MissingDir" isn't
	// ignore - we don't want to default back to the env var.
	headers = map[string][]string{}
	cmd = exec.Command(dockerBinary, "--config", "MissingDir", "-H="+server.URL[7:], "ps")
	cmd.Env = append(os.Environ(), "DOCKER_CONFIG="+cDir)
	out, _, _ = runCommandWithOutput(cmd)

	if headers["Myheader"] != nil {
		c.Fatalf("ps6 - Headers are there but shouldn't be: %q\nout:%v", headers, out)
	}

}
