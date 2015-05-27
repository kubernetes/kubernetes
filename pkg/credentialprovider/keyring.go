/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package credentialprovider

import (
	"encoding/json"
	"net/url"
	"sort"
	"strings"

	docker "github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// DockerKeyring tracks a set of docker registry credentials, maintaining a
// reverse index across the registry endpoints. A registry endpoint is made
// up of a host (e.g. registry.example.com), but it may also contain a path
// (e.g. registry.example.com/foo) This index is important for two reasons:
// - registry endpoints may overlap, and when this happens we must find the
//   most specific match for a given image
// - iterating a map does not yield predictable results
type DockerKeyring interface {
	Lookup(image string) ([]docker.AuthConfiguration, bool)
}

// BasicDockerKeyring is a trivial map-backed implementation of DockerKeyring
type BasicDockerKeyring struct {
	index []string
	creds map[string][]docker.AuthConfiguration
}

// lazyDockerKeyring is an implementation of DockerKeyring that lazily
// materializes its dockercfg based on a set of dockerConfigProviders.
type lazyDockerKeyring struct {
	Providers []DockerConfigProvider
}

func (dk *BasicDockerKeyring) Add(cfg DockerConfig) {
	if dk.index == nil {
		dk.index = make([]string, 0)
		dk.creds = make(map[string][]docker.AuthConfiguration)
	}
	for loc, ident := range cfg {

		creds := docker.AuthConfiguration{
			Username: ident.Username,
			Password: ident.Password,
			Email:    ident.Email,
		}

		parsed, err := url.Parse(loc)
		if err != nil {
			glog.Errorf("Entry %q in dockercfg invalid (%v), ignoring", loc, err)
			continue
		}

		// The docker client allows exact matches:
		//    foo.bar.com/namespace
		// Or hostname matches:
		//    foo.bar.com
		// See ResolveAuthConfig in docker/registry/auth.go.
		if parsed.Host != "" {
			// NOTE: foo.bar.com comes through as Path.
			dk.creds[parsed.Host] = append(dk.creds[parsed.Host], creds)
			dk.index = append(dk.index, parsed.Host)
		}
		if (len(parsed.Path) > 0) && (parsed.Path != "/") {
			key := parsed.Host + parsed.Path
			dk.creds[key] = append(dk.creds[key], creds)
			dk.index = append(dk.index, key)
		}
	}

	eliminateDupes := util.NewStringSet(dk.index...)
	dk.index = eliminateDupes.List()

	// Update the index used to identify which credentials to use for a given
	// image. The index is reverse-sorted so more specific paths are matched
	// first. For example, if for the given image "quay.io/coreos/etcd",
	// credentials for "quay.io/coreos" should match before "quay.io".
	sort.Sort(sort.Reverse(sort.StringSlice(dk.index)))
}

const defaultRegistryHost = "index.docker.io/v1/"

// isDefaultRegistryMatch determines whether the given image will
// pull from the default registry (DockerHub) based on the
// characteristics of its name.
func isDefaultRegistryMatch(image string) bool {
	parts := strings.SplitN(image, "/", 2)

	if len(parts[0]) == 0 {
		return false
	}

	if len(parts) == 1 {
		// e.g. library/ubuntu
		return true
	}

	if parts[0] == "docker.io" || parts[0] == "index.docker.io" {
		// resolve docker.io/image and index.docker.io/image as default registry
		return true
	}

	// From: http://blog.docker.com/2013/07/how-to-use-your-own-registry/
	// Docker looks for either a “.” (domain separator) or “:” (port separator)
	// to learn that the first part of the repository name is a location and not
	// a user name.
	return !strings.ContainsAny(parts[0], ".:")
}

// Lookup implements the DockerKeyring method for fetching credentials based on image name.
// Multiple credentials may be returned if there are multiple potentially valid credentials
// available.  This allows for rotation.
func (dk *BasicDockerKeyring) Lookup(image string) ([]docker.AuthConfiguration, bool) {
	// range over the index as iterating over a map does not provide a predictable ordering
	ret := []docker.AuthConfiguration{}
	for _, k := range dk.index {
		// NOTE: prefix is a sufficient check because while scheme is allowed,
		// it is stripped as part of 'Add'
		if !strings.HasPrefix(image, k) {
			continue
		}

		ret = append(ret, dk.creds[k]...)
	}

	if len(ret) > 0 {
		return ret, true
	}

	// Use credentials for the default registry if provided, and appropriate
	if auth, ok := dk.creds[defaultRegistryHost]; ok && isDefaultRegistryMatch(image) {
		return auth, true
	}

	return []docker.AuthConfiguration{}, false
}

// Lookup implements the DockerKeyring method for fetching credentials
// based on image name.
func (dk *lazyDockerKeyring) Lookup(image string) ([]docker.AuthConfiguration, bool) {
	keyring := &BasicDockerKeyring{}

	for _, p := range dk.Providers {
		keyring.Add(p.Provide())
	}

	return keyring.Lookup(image)
}

type FakeKeyring struct {
	auth []docker.AuthConfiguration
	ok   bool
}

func (f *FakeKeyring) Lookup(image string) ([]docker.AuthConfiguration, bool) {
	return f.auth, f.ok
}

// unionDockerKeyring delegates to a set of keyrings.
type unionDockerKeyring struct {
	keyrings []DockerKeyring
}

func (k *unionDockerKeyring) Lookup(image string) ([]docker.AuthConfiguration, bool) {
	authConfigs := []docker.AuthConfiguration{}

	for _, subKeyring := range k.keyrings {
		if subKeyring == nil {
			continue
		}

		currAuthResults, _ := subKeyring.Lookup(image)
		authConfigs = append(authConfigs, currAuthResults...)
	}

	return authConfigs, (len(authConfigs) > 0)
}

// MakeDockerKeyring inspects the passedSecrets to see if they contain any DockerConfig secrets.  If they do,
// then a DockerKeyring is built based on every hit and unioned with the defaultKeyring.
// If they do not, then the default keyring is returned
func MakeDockerKeyring(passedSecrets []api.Secret, defaultKeyring DockerKeyring) (DockerKeyring, error) {
	passedCredentials := []DockerConfig{}
	for _, passedSecret := range passedSecrets {
		if dockercfgBytes, dockercfgExists := passedSecret.Data[api.DockerConfigKey]; (passedSecret.Type == api.SecretTypeDockercfg) && dockercfgExists && (len(dockercfgBytes) > 0) {
			dockercfg := DockerConfig{}
			if err := json.Unmarshal(dockercfgBytes, &dockercfg); err != nil {
				return nil, err
			}

			passedCredentials = append(passedCredentials, dockercfg)
		}
	}

	if len(passedCredentials) > 0 {
		basicKeyring := &BasicDockerKeyring{}
		for _, currCredentials := range passedCredentials {
			basicKeyring.Add(currCredentials)
		}
		return &unionDockerKeyring{[]DockerKeyring{basicKeyring, defaultKeyring}}, nil
	}

	return defaultKeyring, nil
}
