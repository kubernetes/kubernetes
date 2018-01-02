package fakestorage

import (
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"sync"

	"github.com/docker/docker/integration-cli/cli"
)

var ensureHTTPServerOnce sync.Once

func ensureHTTPServerImage(t testingT) {
	var doIt bool
	ensureHTTPServerOnce.Do(func() {
		doIt = true
	})

	if !doIt {
		return
	}

	defer testEnv.ProtectImage(t, "httpserver:latest")

	tmp, err := ioutil.TempDir("", "docker-http-server-test")
	if err != nil {
		t.Fatalf("could not build http server: %v", err)
	}
	defer os.RemoveAll(tmp)

	goos := testEnv.DaemonPlatform()
	if goos == "" {
		goos = "linux"
	}
	goarch := os.Getenv("DOCKER_ENGINE_GOARCH")
	if goarch == "" {
		goarch = "amd64"
	}

	goCmd, lookErr := exec.LookPath("go")
	if lookErr != nil {
		t.Fatalf("could not build http server: %v", lookErr)
	}

	cmd := exec.Command(goCmd, "build", "-o", filepath.Join(tmp, "httpserver"), "github.com/docker/docker/contrib/httpserver")
	cmd.Env = append(os.Environ(), []string{
		"CGO_ENABLED=0",
		"GOOS=" + goos,
		"GOARCH=" + goarch,
	}...)
	var out []byte
	if out, err = cmd.CombinedOutput(); err != nil {
		t.Fatalf("could not build http server: %s", string(out))
	}

	cpCmd, lookErr := exec.LookPath("cp")
	if lookErr != nil {
		t.Fatalf("could not build http server: %v", lookErr)
	}
	if out, err = exec.Command(cpCmd, "../contrib/httpserver/Dockerfile", filepath.Join(tmp, "Dockerfile")).CombinedOutput(); err != nil {
		t.Fatalf("could not build http server: %v", string(out))
	}

	cli.DockerCmd(t, "build", "-q", "-t", "httpserver", tmp)
}
