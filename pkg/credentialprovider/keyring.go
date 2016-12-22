/*
Copyright 2014 The Kubernetes Authors.

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
	"net"
	"net/url"
	"path/filepath"
	"sort"
	"strings"

	"github.com/golang/glog"

	dockertypes "github.com/docker/engine-api/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/sets"
)

// DockerKeyring tracks a set of docker registry credentials, maintaining a
// reverse index across the registry endpoints. A registry endpoint is made
// up of a host (e.g. registry.example.com), but it may also contain a path
// (e.g. registry.example.com/foo) This index is important for two reasons:
// - registry endpoints may overlap, and when this happens we must find the
//   most specific match for a given image
// - iterating a map does not yield predictable results
type DockerKeyring interface {
	Lookup(image string) ([]LazyAuthConfiguration, bool)
}

// BasicDockerKeyring is a trivial map-backed implementation of DockerKeyring
type BasicDockerKeyring struct {
	index []string
	creds map[string][]LazyAuthConfiguration
}

// lazyDockerKeyring is an implementation of DockerKeyring that lazily
// materializes its dockercfg based on a set of dockerConfigProviders.
type lazyDockerKeyring struct {
	Providers []DockerConfigProvider
}

// LazyAuthConfiguration wraps dockertypes.AuthConfig, potentially deferring its
// binding. If Provider is non-nil, it will be used to obtain new credentials
// by calling LazyProvide() on it.
type LazyAuthConfiguration struct {
	dockertypes.AuthConfig
	Provider DockerConfigProvider
}

func DockerConfigEntryToLazyAuthConfiguration(ident DockerConfigEntry) LazyAuthConfiguration {
	return LazyAuthConfiguration{
		AuthConfig: dockertypes.AuthConfig{
			Username: ident.Username,
			Password: ident.Password,
			Email:    ident.Email,
		},
	}
}

func (dk *BasicDockerKeyring) Add(cfg DockerConfig) {
	if dk.index == nil {
		dk.index = make([]string, 0)
		dk.creds = make(map[string][]LazyAuthConfiguration)
	}
	for loc, ident := range cfg {

		var creds LazyAuthConfiguration
		if ident.Provider != nil {
			creds = LazyAuthConfiguration{
				Provider: ident.Provider,
			}
		} else {
			creds = DockerConfigEntryToLazyAuthConfiguration(ident)
		}

		value := loc
		if !strings.HasPrefix(value, "https://") && !strings.HasPrefix(value, "http://") {
			value = "https://" + value
		}
		parsed, err := url.Parse(value)
		if err != nil {
			glog.Errorf("Entry %q in dockercfg invalid (%v), ignoring", loc, err)
			continue
		}

		// The docker client allows exact matches:
		//    foo.bar.com/namespace
		// Or hostname matches:
		//    foo.bar.com
		// It also considers /v2/  and /v1/ equivalent to the hostname
		// See ResolveAuthConfig in docker/registry/auth.go.
		effectivePath := parsed.Path
		if strings.HasPrefix(effectivePath, "/v2/") || strings.HasPrefix(effectivePath, "/v1/") {
			effectivePath = effectivePath[3:]
		}
		var key string
		if (len(effectivePath) > 0) && (effectivePath != "/") {
			key = parsed.Host + effectivePath
		} else {
			key = parsed.Host
		}
		dk.creds[key] = append(dk.creds[key], creds)
		dk.index = append(dk.index, key)
	}

	eliminateDupes := sets.NewString(dk.index...)
	dk.index = eliminateDupes.List()

	// Update the index used to identify which credentials to use for a given
	// image. The index is reverse-sorted so more specific paths are matched
	// first. For example, if for the given image "quay.io/coreos/etcd",
	// credentials for "quay.io/coreos" should match before "quay.io".
	sort.Sort(sort.Reverse(sort.StringSlice(dk.index)))
}

const (
	defaultRegistryHost = "index.docker.io"
	defaultRegistryKey  = defaultRegistryHost + "/v1/"
)

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

// url.Parse require a scheme, but ours don't have schemes.  Adding a
// scheme to make url.Parse happy, then clear out the resulting scheme.
func parseSchemelessUrl(schemelessUrl string) (*url.URL, error) {
	parsed, err := url.Parse("https://" + schemelessUrl)
	if err != nil {
		return nil, err
	}
	// clear out the resulting scheme
	parsed.Scheme = ""
	return parsed, nil
}

// split the host name into parts, as well as the port
func splitUrl(url *url.URL) (parts []string, port string) {
	host, port, err := net.SplitHostPort(url.Host)
	if err != nil {
		// could not parse port
		host, port = url.Host, ""
	}
	return strings.Split(host, "."), port
}

// overloaded version of urlsMatch, operating on strings instead of URLs.
func urlsMatchStr(glob string, target string) (bool, error) {
	globUrl, err := parseSchemelessUrl(glob)
	if err != nil {
		return false, err
	}
	targetUrl, err := parseSchemelessUrl(target)
	if err != nil {
		return false, err
	}
	return urlsMatch(globUrl, targetUrl)
}

// check whether the given target url matches the glob url, which may have
// glob wild cards in the host name.
//
// Examples:
//    globUrl=*.docker.io, targetUrl=blah.docker.io => match
//    globUrl=*.docker.io, targetUrl=not.right.io   => no match
//
// Note that we don't support wildcards in ports and paths yet.
func urlsMatch(globUrl *url.URL, targetUrl *url.URL) (bool, error) {
	globUrlParts, globPort := splitUrl(globUrl)
	targetUrlParts, targetPort := splitUrl(targetUrl)
	if globPort != targetPort {
		// port doesn't match
		return false, nil
	}
	if len(globUrlParts) != len(targetUrlParts) {
		// host name does not have the same number of parts
		return false, nil
	}
	if !strings.HasPrefix(targetUrl.Path, globUrl.Path) {
		// the path of the credential must be a prefix
		return false, nil
	}
	for k, globUrlPart := range globUrlParts {
		targetUrlPart := targetUrlParts[k]
		matched, err := filepath.Match(globUrlPart, targetUrlPart)
		if err != nil {
			return false, err
		}
		if !matched {
			// glob mismatch for some part
			return false, nil
		}
	}
	// everything matches
	return true, nil
}

// Lookup implements the DockerKeyring method for fetching credentials based on image name.
// Multiple credentials may be returned if there are multiple potentially valid credentials
// available.  This allows for rotation.
func (dk *BasicDockerKeyring) Lookup(image string) ([]LazyAuthConfiguration, bool) {
	// range over the index as iterating over a map does not provide a predictable ordering
	ret := []LazyAuthConfiguration{}
	for _, k := range dk.index {
		// both k and image are schemeless URLs because even though schemes are allowed
		// in the credential configurations, we remove them in Add.
		if matched, _ := urlsMatchStr(k, image); !matched {
			continue
		}

		ret = append(ret, dk.creds[k]...)
	}

	if len(ret) > 0 {
		return ret, true
	}

	// Use credentials for the default registry if provided, and appropriate
	if isDefaultRegistryMatch(image) {
		if auth, ok := dk.creds[defaultRegistryHost]; ok {
			return auth, true
		}
	}

	return []LazyAuthConfiguration{}, false
}

// Lookup implements the DockerKeyring method for fetching credentials
// based on image name.
func (dk *lazyDockerKeyring) Lookup(image string) ([]LazyAuthConfiguration, bool) {
	keyring := &BasicDockerKeyring{}

	for _, p := range dk.Providers {
		keyring.Add(p.Provide())
	}

	return keyring.Lookup(image)
}

type FakeKeyring struct {
	auth []LazyAuthConfiguration
	ok   bool
}

func (f *FakeKeyring) Lookup(image string) ([]LazyAuthConfiguration, bool) {
	return f.auth, f.ok
}

// unionDockerKeyring delegates to a set of keyrings.
type unionDockerKeyring struct {
	keyrings []DockerKeyring
}

func (k *unionDockerKeyring) Lookup(image string) ([]LazyAuthConfiguration, bool) {
	authConfigs := []LazyAuthConfiguration{}
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
func MakeDockerKeyring(passedSecrets []v1.Secret, defaultKeyring DockerKeyring) (DockerKeyring, error) {
	passedCredentials := []DockerConfig{}
	for _, passedSecret := range passedSecrets {
		if dockerConfigJsonBytes, dockerConfigJsonExists := passedSecret.Data[v1.DockerConfigJsonKey]; (passedSecret.Type == v1.SecretTypeDockerConfigJson) && dockerConfigJsonExists && (len(dockerConfigJsonBytes) > 0) {
			dockerConfigJson := DockerConfigJson{}
			if err := json.Unmarshal(dockerConfigJsonBytes, &dockerConfigJson); err != nil {
				return nil, err
			}

			passedCredentials = append(passedCredentials, dockerConfigJson.Auths)
		} else if dockercfgBytes, dockercfgExists := passedSecret.Data[v1.DockerConfigKey]; (passedSecret.Type == v1.SecretTypeDockercfg) && dockercfgExists && (len(dockercfgBytes) > 0) {
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
