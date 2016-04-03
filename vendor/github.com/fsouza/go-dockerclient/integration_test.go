// Copyright 2015 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build docker_integration

package docker

import (
	"bytes"
	"os"
	"testing"
)

var dockerEndpoint string

func init() {
	dockerEndpoint = os.Getenv("DOCKER_HOST")
	if dockerEndpoint == "" {
		dockerEndpoint = "unix:///var/run/docker.sock"
	}
}

func TestIntegrationPullCreateStartLogs(t *testing.T) {
	imageName := pullImage(t)
	client := getClient()
	hostConfig := HostConfig{PublishAllPorts: true}
	createOpts := CreateContainerOptions{
		Config: &Config{
			Image: imageName,
			Cmd:   []string{"cat", "/home/gopher/file.txt"},
			User:  "gopher",
		},
		HostConfig: &hostConfig,
	}
	container, err := client.CreateContainer(createOpts)
	if err != nil {
		t.Fatal(err)
	}
	err = client.StartContainer(container.ID, &hostConfig)
	if err != nil {
		t.Fatal(err)
	}
	status, err := client.WaitContainer(container.ID)
	if err != nil {
		t.Error(err)
	}
	if status != 0 {
		t.Error("WaitContainer(%q): wrong status. Want 0. Got %d", container.ID, status)
	}
	var stdout, stderr bytes.Buffer
	logsOpts := LogsOptions{
		Container:    container.ID,
		OutputStream: &stdout,
		ErrorStream:  &stderr,
		Stdout:       true,
		Stderr:       true,
	}
	err = client.Logs(logsOpts)
	if err != nil {
		t.Error(err)
	}
	if stderr.String() != "" {
		t.Errorf("Got unexpected stderr from logs: %q", stderr.String())
	}
	expected := `Welcome to reality, wake up and rejoice
Welcome to reality, you've made the right choice
Welcome to reality, and let them hear your voice, shout it out!
`
	if stdout.String() != expected {
		t.Errorf("Got wrong stdout from logs.\nWant:\n%#v.\n\nGot:\n%#v.", expected, stdout.String())
	}
}

func pullImage(t *testing.T) string {
	imageName := "fsouza/go-dockerclient-integration"
	var buf bytes.Buffer
	pullOpts := PullImageOptions{
		Repository:   imageName,
		OutputStream: &buf,
	}
	client := getClient()
	err := client.PullImage(pullOpts, AuthConfiguration{})
	if err != nil {
		t.Logf("Pull output: %s", buf.String())
		t.Fatal(err)
	}
	return imageName
}

func getClient() *Client {
	client, _ := NewClient(dockerEndpoint)
	return client
}
