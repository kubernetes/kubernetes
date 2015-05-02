/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package rkt

import (
	"fmt"
	"os"
	"path"
	"strings"

	"github.com/docker/docker/pkg/parsers"
	docker "github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
)

const (
	authDir            = "auth.d"
	dockerAuthTemplate = `{"rktKind":"dockerAuth","rktVersion":"v1","registries":[%q],"credentials":{"user":%q,"password":%q}}`
)

// writeDockerAuthConfig writes the docker credentials to rkt auth config files.
// This enables rkt to pull docker images from docker registry with credentials.
func (r *Runtime) writeDockerAuthConfig(image string, creds docker.AuthConfiguration) error {
	registry := "index.docker.io"
	// Image spec: [<registry>/]<repository>/<image>[:<version]
	explicitRegistry := (strings.Count(image, "/") == 2)
	if explicitRegistry {
		registry = strings.Split(image, "/")[0]
	}

	localConfigDir := rktLocalConfigDir
	if r.config.LocalConfigDir != "" {
		localConfigDir = r.config.LocalConfigDir
	}
	authDir := path.Join(localConfigDir, "auth.d")
	if _, err := os.Stat(authDir); os.IsNotExist(err) {
		if err := os.Mkdir(authDir, 0600); err != nil {
			glog.Errorf("rkt: Cannot create auth dir: %v", err)
			return err
		}
	}
	f, err := os.Create(path.Join(localConfigDir, authDir, registry+".json"))
	if err != nil {
		glog.Errorf("rkt: Cannot create docker auth config file: %v", err)
		return err
	}
	defer f.Close()
	config := fmt.Sprintf(dockerAuthTemplate, registry, creds.Username, creds.Password)
	if _, err := f.Write([]byte(config)); err != nil {
		glog.Errorf("rkt: Cannot write docker auth config file: %v", err)
		return err
	}
	return nil
}

// PullImage invokes 'rkt fetch' to download an aci.
func (r *Runtime) PullImage(img string) error {
	// Use credentials for docker images. This string operation can be cleaned up
	// once the format of image is landed, see:
	// https://github.com/GoogleCloudPlatform/kubernetes/issues/7203
	//
	if strings.HasPrefix(img, dockerPrefix) {
		repoToPull, tag := parsers.ParseRepositoryTag(img)
		// If no tag was specified, use the default "latest".
		if len(tag) == 0 {
			tag = "latest"
		}

		creds, ok := r.dockerKeyring.Lookup(repoToPull)
		if !ok {
			glog.V(1).Infof("Pulling image %s without credentials", img)
		}

		// Let's update a json.
		// TODO(yifan): Find a way to feed this to rkt.
		if err := r.writeDockerAuthConfig(img, creds); err != nil {
			return err
		}
	}

	output, err := r.runCommand("fetch", img)
	if err != nil {
		return fmt.Errorf("rkt: Failed to fetch image: %v:", output)
	}
	return nil
}

// IsImagePresent returns true if the image is available on the machine.
// TODO(yifan): 'rkt image' is now landed on master, use that once we bump up
// the rkt version.
func (r *Runtime) IsImagePresent(img string) (bool, error) {
	if _, err := r.runCommand("prepare", "--local=true", img); err != nil {
		return false, nil
	}
	return true, nil
}
