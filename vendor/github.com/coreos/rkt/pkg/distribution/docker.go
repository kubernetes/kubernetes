// Copyright 2016 The rkt Authors
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

package distribution

import (
	"fmt"
	"net/url"
	"path"
	"strings"

	d2acommon "github.com/appc/docker2aci/lib/common"
)

const (
	distDockerVersion = 0

	// TypeDocker represents the Docker distribution type
	TypeDocker Type = "docker"

	defaultIndexURL   = "registry-1.docker.io"
	defaultTag        = "latest"
	defaultRepoPrefix = "library/"
)

func init() {
	Register(TypeDocker, NewDocker)
}

// Docker defines a distribution using a docker registry.
// The format after the docker distribution type prefix (cimd:docker:v=0:) is the same
// as the docker image string format (man docker-pull):
// cimd:docker:v=0:[REGISTRY_HOST[:REGISTRY_PORT]/]NAME[:TAG|@DIGEST]
// Examples:
// cimd:docker:v=0:busybox
// cimd:docker:v=0:busybox:latest
// cimd:docker:v=0:registry-1.docker.io/library/busybox@sha256:a59906e33509d14c036c8678d687bd4eec81ed7c4b8ce907b888c607f6a1e0e6
type Docker struct {
	url       string // a valid docker reference URL
	parsedURL *d2acommon.ParsedDockerURL

	full   string // the full string representation for equals operations
	simple string // the user friendly (simple) string representation
}

// NewDocker creates a new docker distribution from the provided distribution uri string
func NewDocker(u *url.URL) (Distribution, error) {
	dp, err := parseCIMD(u)
	if err != nil {
		return nil, fmt.Errorf("cannot parse URI: %q: %v", u.String(), err)
	}
	if dp.Type != TypeDocker {
		return nil, fmt.Errorf("wrong distribution type: %q", dp.Type)
	}

	parsed, err := d2acommon.ParseDockerURL(dp.Data)
	if err != nil {
		return nil, fmt.Errorf("bad docker URL %q: %v", dp.Data, err)
	}

	return &Docker{
		url:       dp.Data,
		parsedURL: parsed,
		simple:    SimpleDockerRef(parsed),
		full:      FullDockerRef(parsed),
	}, nil
}

// NewDockerFromDockerString creates a new docker distribution from the provided
// docker string (like "busybox", "busybox:1.0", "myregistry.example.com:4000/busybox" etc...)
func NewDockerFromString(ds string) (Distribution, error) {
	urlStr := NewCIMDString(TypeDocker, distDockerVersion, ds)
	u, err := url.Parse(urlStr)
	if err != nil {
		return nil, err
	}
	return NewDocker(u)
}

func (d *Docker) CIMD() *url.URL {
	uriStr := NewCIMDString(TypeDocker, distDockerVersion, d.url)
	// Create a copy of the URL
	u, err := url.Parse(uriStr)
	if err != nil {
		panic(err)
	}
	return u
}

func (d *Docker) String() string {
	return d.simple
}

func (d *Docker) Equals(dist Distribution) bool {
	d2, ok := dist.(*Docker)
	if !ok {
		return false
	}

	return d.full == d2.full
}

// ReferenceURL returns the docker reference URL.
func (d *Docker) ReferenceURL() string {
	return d.url
}

// SimpleDockerRef returns a simplyfied docker reference. This means removing
// the index url if it's the default docker registry (registry-1.docker.io),
// removing the default repo (library) when using the default docker registry
func SimpleDockerRef(p *d2acommon.ParsedDockerURL) string {
	var sds string
	if p.IndexURL != defaultIndexURL {
		sds += p.IndexURL
	}

	imageName := p.ImageName
	if p.IndexURL == defaultIndexURL && strings.HasPrefix(p.ImageName, defaultRepoPrefix) {
		imageName = strings.TrimPrefix(p.ImageName, defaultRepoPrefix)
	}

	if sds == "" {
		sds = imageName
	} else {
		sds = path.Join(sds, imageName)
	}

	digest := p.Digest
	tag := p.Tag
	if digest != "" {
		sds += "@" + digest
	} else {
		if tag != defaultTag {
			sds += ":" + tag
		}
	}

	return sds
}

// FullDockerRef return the docker reference populated with all the default values.
// References like "busybox" or "registry-1.docker.io/library/busybox:latest"
// will become the same docker reference.
func FullDockerRef(p *d2acommon.ParsedDockerURL) string {
	fds := path.Join(p.IndexURL, p.ImageName)
	digest := p.Digest
	tag := p.Tag

	if digest != "" {
		fds += "@" + digest
	} else {
		fds += ":" + tag
	}

	return fds
}
