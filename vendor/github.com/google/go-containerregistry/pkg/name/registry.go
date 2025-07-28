// Copyright 2018 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package name

import (
	"encoding"
	"encoding/json"
	"net"
	"net/url"
	"path"
	"regexp"
	"strings"
)

// Detect more complex forms of local references.
var reLocal = regexp.MustCompile(`.*\.local(?:host)?(?::\d{1,5})?$`)

// Detect the loopback IP (127.0.0.1)
var reLoopback = regexp.MustCompile(regexp.QuoteMeta("127.0.0.1"))

// Detect the loopback IPV6 (::1)
var reipv6Loopback = regexp.MustCompile(regexp.QuoteMeta("::1"))

// Registry stores a docker registry name in a structured form.
type Registry struct {
	insecure bool
	registry string
}

var _ encoding.TextMarshaler = (*Registry)(nil)
var _ encoding.TextUnmarshaler = (*Registry)(nil)
var _ json.Marshaler = (*Registry)(nil)
var _ json.Unmarshaler = (*Registry)(nil)

// RegistryStr returns the registry component of the Registry.
func (r Registry) RegistryStr() string {
	return r.registry
}

// Name returns the name from which the Registry was derived.
func (r Registry) Name() string {
	return r.RegistryStr()
}

func (r Registry) String() string {
	return r.Name()
}

// Repo returns a Repository in the Registry with the given name.
func (r Registry) Repo(repo ...string) Repository {
	return Repository{Registry: r, repository: path.Join(repo...)}
}

// Scope returns the scope required to access the registry.
func (r Registry) Scope(string) string {
	// The only resource under 'registry' is 'catalog'. http://goo.gl/N9cN9Z
	return "registry:catalog:*"
}

func (r Registry) isRFC1918() bool {
	ipStr := strings.Split(r.Name(), ":")[0]
	ip := net.ParseIP(ipStr)
	if ip == nil {
		return false
	}
	for _, cidr := range []string{"10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"} {
		_, block, _ := net.ParseCIDR(cidr)
		if block.Contains(ip) {
			return true
		}
	}
	return false
}

// Scheme returns https scheme for all the endpoints except localhost or when explicitly defined.
func (r Registry) Scheme() string {
	if r.insecure {
		return "http"
	}
	if r.isRFC1918() {
		return "http"
	}
	if strings.HasPrefix(r.Name(), "localhost:") {
		return "http"
	}
	if reLocal.MatchString(r.Name()) {
		return "http"
	}
	if reLoopback.MatchString(r.Name()) {
		return "http"
	}
	if reipv6Loopback.MatchString(r.Name()) {
		return "http"
	}
	return "https"
}

func checkRegistry(name string) error {
	// Per RFC 3986, registries (authorities) are required to be prefixed with "//"
	// url.Host == hostname[:port] == authority
	if url, err := url.Parse("//" + name); err != nil || url.Host != name {
		return newErrBadName("registries must be valid RFC 3986 URI authorities: %s", name)
	}
	return nil
}

// NewRegistry returns a Registry based on the given name.
// Strict validation requires explicit, valid RFC 3986 URI authorities to be given.
func NewRegistry(name string, opts ...Option) (Registry, error) {
	opt := makeOptions(opts...)
	if opt.strict && len(name) == 0 {
		return Registry{}, newErrBadName("strict validation requires the registry to be explicitly defined")
	}

	if err := checkRegistry(name); err != nil {
		return Registry{}, err
	}

	if name == "" {
		name = opt.defaultRegistry
	}
	// Rewrite "docker.io" to "index.docker.io".
	// See: https://github.com/google/go-containerregistry/issues/68
	if name == defaultRegistryAlias {
		name = DefaultRegistry
	}

	return Registry{registry: name, insecure: opt.insecure}, nil
}

// NewInsecureRegistry returns an Insecure Registry based on the given name.
//
// Deprecated: Use the Insecure Option with NewRegistry instead.
func NewInsecureRegistry(name string, opts ...Option) (Registry, error) {
	opts = append(opts, Insecure)
	return NewRegistry(name, opts...)
}

// MarshalJSON formats the Registry into a string for JSON serialization.
func (r Registry) MarshalJSON() ([]byte, error) { return json.Marshal(r.String()) }

// UnmarshalJSON parses a JSON string into a Registry.
func (r *Registry) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	n, err := NewRegistry(s)
	if err != nil {
		return err
	}
	*r = n
	return nil
}

// MarshalText formats the registry into a string for text serialization.
func (r Registry) MarshalText() ([]byte, error) { return []byte(r.String()), nil }

// UnmarshalText parses a text string into a Registry.
func (r *Registry) UnmarshalText(data []byte) error {
	n, err := NewRegistry(string(data))
	if err != nil {
		return err
	}
	*r = n
	return nil
}
