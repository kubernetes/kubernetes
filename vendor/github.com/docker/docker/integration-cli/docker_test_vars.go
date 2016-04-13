package main

import (
	"fmt"
	"os"
	"os/exec"
)

var (
	// the docker binary to use
	dockerBinary = "docker"

	// the private registry image to use for tests involving the registry
	registryImageName = "registry"

	// the private registry to use for tests
	privateRegistryURL = "127.0.0.1:5000"

	dockerBasePath       = "/var/lib/docker"
	volumesConfigPath    = dockerBasePath + "/volumes"
	containerStoragePath = dockerBasePath + "/containers"

	runtimePath    = "/var/run/docker"
	execDriverPath = runtimePath + "/execdriver/native"

	workingDirectory string
)

func init() {
	if dockerBin := os.Getenv("DOCKER_BINARY"); dockerBin != "" {
		dockerBinary = dockerBin
	}
	var err error
	dockerBinary, err = exec.LookPath(dockerBinary)
	if err != nil {
		fmt.Printf("ERROR: couldn't resolve full path to the Docker binary (%v)", err)
		os.Exit(1)
	}
	if registryImage := os.Getenv("REGISTRY_IMAGE"); registryImage != "" {
		registryImageName = registryImage
	}
	if registry := os.Getenv("REGISTRY_URL"); registry != "" {
		privateRegistryURL = registry
	}
	workingDirectory, _ = os.Getwd()
}
