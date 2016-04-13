package main

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/go-check/check"
)

const v2binary = "registry-v2"

type testRegistryV2 struct {
	cmd *exec.Cmd
	dir string
}

func newTestRegistryV2(c *check.C) (*testRegistryV2, error) {
	template := `version: 0.1
loglevel: debug
storage:
    filesystem:
        rootdirectory: %s
http:
    addr: %s`
	tmp, err := ioutil.TempDir("", "registry-test-")
	if err != nil {
		return nil, err
	}
	confPath := filepath.Join(tmp, "config.yaml")
	config, err := os.Create(confPath)
	if err != nil {
		return nil, err
	}
	if _, err := fmt.Fprintf(config, template, tmp, privateRegistryURL); err != nil {
		os.RemoveAll(tmp)
		return nil, err
	}

	cmd := exec.Command(v2binary, confPath)
	if err := cmd.Start(); err != nil {
		os.RemoveAll(tmp)
		if os.IsNotExist(err) {
			c.Skip(err.Error())
		}
		return nil, err
	}
	return &testRegistryV2{
		cmd: cmd,
		dir: tmp,
	}, nil
}

func (t *testRegistryV2) Ping() error {
	// We always ping through HTTP for our test registry.
	resp, err := http.Get(fmt.Sprintf("http://%s/v2/", privateRegistryURL))
	if err != nil {
		return err
	}
	if resp.StatusCode != 200 {
		return fmt.Errorf("registry ping replied with an unexpected status code %d", resp.StatusCode)
	}
	return nil
}

func (r *testRegistryV2) Close() {
	r.cmd.Process.Kill()
	os.RemoveAll(r.dir)
}
