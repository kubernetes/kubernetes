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
)

// Type represents the Distribution type
type Type string

const (
	// Scheme represents the Distribution URI scheme
	Scheme = "cimd"
)

// Distribution is the interface that represents a distribution point.
type Distribution interface {
	// URI returns a copy of the Distribution CIMD URI.
	CIMD() *url.URL
	// String returns a user friendly representation of the distribution point.
	String() string
	// Equals returns true if this distribution equals the given distribution, else false.
	Equals(Distribution) bool
}

type newDistribution func(*url.URL) (Distribution, error)

var distributions = make(map[Type]newDistribution)

// Register registers a function that returns a new instance of the given
// distribution. This is intended to be called from the init function in
// packages that implement distribution functions.
func Register(distType Type, f newDistribution) {
	if _, ok := distributions[distType]; ok {
		panic(fmt.Errorf("distribution %q already registered", distType))
	}
	distributions[distType] = f
}

// Get returns a Distribution from the input URI.
// It returns an error if the uri string is wrong or referencing an unknown
// distribution type.
func Get(u *url.URL) (Distribution, error) {
	c, err := parseCIMD(u)
	if err != nil {
		return nil, fmt.Errorf("malformed distribution uri %q: %v", u.String(), err)
	}
	if u.Scheme != Scheme {
		return nil, fmt.Errorf("malformed distribution uri %q", u.String())
	}
	if _, ok := distributions[c.Type]; !ok {
		return nil, fmt.Errorf("unknown distribution type: %q", c.Type)
	}
	return distributions[c.Type](u)
}

// Parse parses the provided distribution URI string and returns a
// Distribution.
func Parse(rawuri string) (Distribution, error) {
	u, err := url.Parse(rawuri)
	if err != nil {
		return nil, fmt.Errorf("cannot parse uri: %q: %v", rawuri, err)
	}
	return Get(u)
}
