/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package system

import (
	"context"
	"regexp"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/client"
	"github.com/pkg/errors"
)

var _ Validator = &DockerValidator{}

// DockerValidator validates docker configuration.
type DockerValidator struct {
	Reporter Reporter
}

func (d *DockerValidator) Name() string {
	return "docker"
}

const (
	dockerConfigPrefix           = "DOCKER_"
	latestValidatedDockerVersion = "18.09"
)

// TODO(random-liu): Add more validating items.
func (d *DockerValidator) Validate(spec SysSpec) (error, error) {
	if spec.RuntimeSpec.DockerSpec == nil {
		// If DockerSpec is not specified, assume current runtime is not
		// docker, skip the docker configuration validation.
		return nil, nil
	}

	c, err := client.NewClient(dockerEndpoint, "", nil, nil)
	if err != nil {
		return nil, errors.Wrap(err, "failed to create docker client")
	}
	info, err := c.Info(context.Background())
	if err != nil {
		return nil, errors.Wrap(err, "failed to get docker info")
	}
	return d.validateDockerInfo(spec.RuntimeSpec.DockerSpec, info)
}

func (d *DockerValidator) validateDockerInfo(spec *DockerSpec, info types.Info) (error, error) {
	// Validate docker version.
	matched := false
	for _, v := range spec.Version {
		r := regexp.MustCompile(v)
		if r.MatchString(info.ServerVersion) {
			d.Reporter.Report(dockerConfigPrefix+"VERSION", info.ServerVersion, good)
			matched = true
		}
	}
	if !matched {
		// If it's of the new Docker version scheme but didn't match above, it
		// must be a newer version than the most recently validated one.
		ver := `\d{2}\.\d+\.\d+(?:-[a-z]{2})?`
		r := regexp.MustCompile(ver)
		if r.MatchString(info.ServerVersion) {
			d.Reporter.Report(dockerConfigPrefix+"VERSION", info.ServerVersion, good)
			w := errors.Errorf(
				"this Docker version is not on the list of validated versions: %s. Latest validated version: %s",
				info.ServerVersion,
				latestValidatedDockerVersion,
			)
			return w, nil
		}
		d.Reporter.Report(dockerConfigPrefix+"VERSION", info.ServerVersion, bad)
		return nil, errors.Errorf("unsupported docker version: %s", info.ServerVersion)
	}
	// Validate graph driver.
	item := dockerConfigPrefix + "GRAPH_DRIVER"
	for _, gd := range spec.GraphDriver {
		if info.Driver == gd {
			d.Reporter.Report(item, info.Driver, good)
			return nil, nil
		}
	}
	d.Reporter.Report(item, info.Driver, bad)
	return nil, errors.Errorf("unsupported graph driver: %s", info.Driver)
}
