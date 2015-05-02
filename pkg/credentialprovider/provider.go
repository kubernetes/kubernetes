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
	"os"
	"reflect"
	"sync"
	"time"

	"github.com/golang/glog"
)

// DockerConfigProvider is the interface that registered extensions implement
// to materialize 'dockercfg' credentials.
type DockerConfigProvider interface {
	Enabled() bool
	Provide() DockerConfig
}

// A DockerConfigProvider that simply reads the .dockercfg file
type defaultDockerConfigProvider struct{}

// init registers our default provider, which simply reads the .dockercfg file.
func init() {
	RegisterCredentialProvider(".dockercfg",
		&CachingDockerConfigProvider{
			Provider: &defaultDockerConfigProvider{},
			Lifetime: 5 * time.Minute,
		})
}

// CachingDockerConfigProvider implements DockerConfigProvider by composing
// with another DockerConfigProvider and caching the DockerConfig it provides
// for a pre-specified lifetime.
type CachingDockerConfigProvider struct {
	Provider DockerConfigProvider
	Lifetime time.Duration

	// cache fields
	cacheDockerConfig DockerConfig
	expiration        time.Time
	mu                sync.Mutex
}

// Enabled implements dockerConfigProvider
func (d *defaultDockerConfigProvider) Enabled() bool {
	return true
}

// Provide implements dockerConfigProvider
func (d *defaultDockerConfigProvider) Provide() DockerConfig {
	// Read the standard Docker credentials from .dockercfg
	if cfg, err := ReadDockerConfigFile(); err == nil {
		return cfg
	} else if !os.IsNotExist(err) {
		glog.V(4).Infof("Unable to parse Docker config file: %v", err)
	}
	return DockerConfig{}
}

// Enabled implements dockerConfigProvider
func (d *CachingDockerConfigProvider) Enabled() bool {
	return d.Provider.Enabled()
}

// Provide implements dockerConfigProvider
func (d *CachingDockerConfigProvider) Provide() DockerConfig {
	d.mu.Lock()
	defer d.mu.Unlock()

	// If the cache hasn't expired, return our cache
	if time.Now().Before(d.expiration) {
		return d.cacheDockerConfig
	}

	glog.Infof("Refreshing cache for provider: %v", reflect.TypeOf(d.Provider).String())
	d.cacheDockerConfig = d.Provider.Provide()
	d.expiration = time.Now().Add(d.Lifetime)
	return d.cacheDockerConfig
}
