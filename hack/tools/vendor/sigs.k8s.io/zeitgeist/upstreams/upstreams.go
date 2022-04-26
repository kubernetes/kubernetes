/*
Copyright 2020 The Kubernetes Authors.

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

// Package upstreams defines how to check version info in upstream repositories.
//
// Upstream types are identified by their _flavour_, represented as a string (see UpstreamFlavour).
//
// Different Upstream types can have their own parameters, but they must:
//
//	- Include the BaseUpstream type
//	- Define a LatestVersion() function that returns the latest available version as a string
package upstreams

import (
	"github.com/pkg/errors"
)

// UpstreamBase only contains a flavour. "Concrete" upstreams each implement their own fields.
type UpstreamBase struct {
	Flavour UpstreamFlavour `yaml:"flavour"`
}

// LatestVersion will always return an error.
// UpstreamBase is only used to determine which actual upstream needs to be called, so it cannot return a sensible value
func (u *UpstreamBase) LatestVersion() (string, error) {
	return "", errors.New("cannot determine latest version for UpstreamBase")
}

// UpstreamFlavour is an enum of all supported upstreams and their string representation
type UpstreamFlavour string

const (
	// GithubFlavour is for Github releases
	GithubFlavour UpstreamFlavour = "github"
	// GitLabFlavour is for GitLab releases
	GitLabFlavour UpstreamFlavour = "gitlab"
	// AMIFlavour is for Amazon Machine Images
	AMIFlavour UpstreamFlavour = "ami"
	// DummyFlavour is for testing
	DummyFlavour UpstreamFlavour = "dummy"
)
