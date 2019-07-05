// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Provides global docker information.
package docker

import (
	"fmt"
	"regexp"
	"strconv"

	dockertypes "github.com/docker/docker/api/types"
	"golang.org/x/net/context"

	"time"

	"github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/machine"
)

var dockerTimeout = 10 * time.Second

func defaultContext() context.Context {
	ctx, _ := context.WithTimeout(context.Background(), dockerTimeout)
	return ctx
}

func SetTimeout(timeout time.Duration) {
	dockerTimeout = timeout
}

func Status() (v1.DockerStatus, error) {
	return StatusWithContext(defaultContext())
}

func StatusWithContext(ctx context.Context) (v1.DockerStatus, error) {
	client, err := Client()
	if err != nil {
		return v1.DockerStatus{}, fmt.Errorf("unable to communicate with docker daemon: %v", err)
	}
	dockerInfo, err := client.Info(ctx)
	if err != nil {
		return v1.DockerStatus{}, err
	}
	return StatusFromDockerInfo(dockerInfo)
}

func StatusFromDockerInfo(dockerInfo dockertypes.Info) (v1.DockerStatus, error) {
	out := v1.DockerStatus{}
	out.KernelVersion = machine.KernelVersion()
	out.OS = dockerInfo.OperatingSystem
	out.Hostname = dockerInfo.Name
	out.RootDir = dockerInfo.DockerRootDir
	out.Driver = dockerInfo.Driver
	out.NumImages = dockerInfo.Images
	out.NumContainers = dockerInfo.Containers
	out.DriverStatus = make(map[string]string, len(dockerInfo.DriverStatus))
	for _, v := range dockerInfo.DriverStatus {
		out.DriverStatus[v[0]] = v[1]
	}
	var err error
	ver, err := VersionString()
	if err != nil {
		return out, err
	}
	out.Version = ver
	ver, err = APIVersionString()
	if err != nil {
		return out, err
	}
	out.APIVersion = ver
	return out, nil
}

func Images() ([]v1.DockerImage, error) {
	client, err := Client()
	if err != nil {
		return nil, fmt.Errorf("unable to communicate with docker daemon: %v", err)
	}
	images, err := client.ImageList(defaultContext(), dockertypes.ImageListOptions{All: false})
	if err != nil {
		return nil, err
	}

	out := []v1.DockerImage{}
	const unknownTag = "<none>:<none>"
	for _, image := range images {
		if len(image.RepoTags) == 1 && image.RepoTags[0] == unknownTag {
			// images with repo or tags are uninteresting.
			continue
		}
		di := v1.DockerImage{
			ID:          image.ID,
			RepoTags:    image.RepoTags,
			Created:     image.Created,
			VirtualSize: image.VirtualSize,
			Size:        image.Size,
		}
		out = append(out, di)
	}
	return out, nil

}

// Checks whether the dockerInfo reflects a valid docker setup, and returns it if it does, or an
// error otherwise.
func ValidateInfo() (*dockertypes.Info, error) {
	client, err := Client()
	if err != nil {
		return nil, fmt.Errorf("unable to communicate with docker daemon: %v", err)
	}

	dockerInfo, err := client.Info(defaultContext())
	if err != nil {
		return nil, fmt.Errorf("failed to detect Docker info: %v", err)
	}

	// Fall back to version API if ServerVersion is not set in info.
	if dockerInfo.ServerVersion == "" {
		version, err := client.ServerVersion(defaultContext())
		if err != nil {
			return nil, fmt.Errorf("unable to get docker version: %v", err)
		}
		dockerInfo.ServerVersion = version.Version
	}
	version, err := parseVersion(dockerInfo.ServerVersion, version_re, 3)
	if err != nil {
		return nil, err
	}

	if version[0] < 1 {
		return nil, fmt.Errorf("cAdvisor requires docker version %v or above but we have found version %v reported as %q", []int{1, 0, 0}, version, dockerInfo.ServerVersion)
	}

	if dockerInfo.Driver == "" {
		return nil, fmt.Errorf("failed to find docker storage driver")
	}

	return &dockerInfo, nil
}

func Version() ([]int, error) {
	ver, err := VersionString()
	if err != nil {
		return nil, err
	}
	return parseVersion(ver, version_re, 3)
}

func APIVersion() ([]int, error) {
	ver, err := APIVersionString()
	if err != nil {
		return nil, err
	}
	return parseVersion(ver, apiversion_re, 2)
}

func VersionString() (string, error) {
	docker_version := "Unknown"
	client, err := Client()
	if err == nil {
		version, err := client.ServerVersion(defaultContext())
		if err == nil {
			docker_version = version.Version
		}
	}
	return docker_version, err
}

func APIVersionString() (string, error) {
	docker_api_version := "Unknown"
	client, err := Client()
	if err == nil {
		version, err := client.ServerVersion(defaultContext())
		if err == nil {
			docker_api_version = version.APIVersion
		}
	}
	return docker_api_version, err
}

func parseVersion(version_string string, regex *regexp.Regexp, length int) ([]int, error) {
	matches := regex.FindAllStringSubmatch(version_string, -1)
	if len(matches) != 1 {
		return nil, fmt.Errorf("version string \"%v\" doesn't match expected regular expression: \"%v\"", version_string, regex.String())
	}
	version_string_array := matches[0][1:]
	version_array := make([]int, length)
	for index, version_str := range version_string_array {
		version, err := strconv.Atoi(version_str)
		if err != nil {
			return nil, fmt.Errorf("error while parsing \"%v\" in \"%v\"", version_str, version_string)
		}
		version_array[index] = version
	}
	return version_array, nil
}
