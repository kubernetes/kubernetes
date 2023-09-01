/*
Copyright 2022 The KCP Authors.

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

package logicalcluster

import (
	"encoding/json"
	"path"
	"regexp"
	"strings"
)

var (
	// Wildcard is the path indicating a requests that spans many logical clusters.
	Wildcard = Path{value: "*"}

	// None represents an unset path.
	None = Path{}

	// TODO is a value created by automated refactoring tools that should be replaced by a real path.
	TODO = None
)

const (
	separator = ":"
)

// Path represents a colon separated list of words describing a path in a logical cluster hierarchy,
// like a file path in a file-system.
//
// For instance, in the following hierarchy:
//
// root/                    (62208dab)
// ├── accounting           (c8a942c5)
// │   └── us-west          (33bab531)
// │       └── invoices     (f5865fce)
// └── management           (e7e08986)
//     └── us-west-invoices (f5865fce)
//
// the following would all be valid paths:
//
//   - root:accounting:us-west:invoices
//   - 62208dab:accounting:us-west:invoices
//   - c8a942c5:us-west:invoices
//   - 33bab531:invoices
//   - f5865fce
//   - root:management:us-west-invoices
//   - 62208dab:management:us-west-invoices
//   - e7e08986:us-west-invoices
type Path struct {
	value string
}

// NewPath returns a new Path.
func NewPath(value string) Path {
	return Path{value}
}

// NewValidatedPath returns a Path and whether it is valid.
func NewValidatedPath(value string) (Path, bool) {
	p := Path{value}
	return p, p.IsValid()
}

// Empty returns true if the stored path is unset.
// It is a convenience method for checking against an empty value.
func (p Path) Empty() bool {
	return p.value == ""
}

// Name return a new Name object from the stored path and whether it can be created.
// A convenience method for working with methods which accept a Name type.
func (p Path) Name() (Name, bool) {
	if strings.HasPrefix(p.value, "system:") {
		return Name(p.value), true
	}
	if _, hasParent := p.Parent(); hasParent {
		return "", false
	}
	return Name(p.value), true
}

// RequestPath returns a URL path segment used to access API for the stored path.
func (p Path) RequestPath() string {
	return path.Join("/clusters", p.value)
}

// String returns string representation of the stored value.
// Satisfies the Stringer interface.
func (p Path) String() string {
	return p.value
}

// Parent returns a new path with all but the last element of the stored path.
func (p Path) Parent() (Path, bool) {
	parent, _ := p.Split()
	return parent, parent.value != ""
}

// Split splits the path immediately following the final colon,
// separating it into a new path and a logical cluster name component.
// If there is no colon in the path,
// Split returns an empty path and a name set to the path.
func (p Path) Split() (parent Path, name string) {
	if strings.HasPrefix(p.value, "system:") {
		return Path{p.value}, ""
	}
	i := strings.LastIndex(p.value, separator)
	if i < 0 {
		return Path{}, p.value
	}
	return Path{p.value[:i]}, p.value[i+1:]
}

// Base returns the last element of the path.
func (p Path) Base() string {
	_, name := p.Split()
	return name
}

// Join returns a new path by adding the given path segment
// into already existing path and separating it with a colon.
func (p Path) Join(name string) Path {
	if p.value == "" {
		return Path{name}
	}
	return Path{p.value + separator + name}
}

// MarshalJSON satisfies the Marshaler interface
// for encoding the path into JSON.
func (p Path) MarshalJSON() ([]byte, error) {
	return json.Marshal(&p.value)
}

// UnmarshalJSON satisfies the Unmarshaler interface implemented by types
// for decoding a JSON encoded path.
func (p *Path) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	p.value = s
	return nil
}

// HasPrefix tests whether the path begins with the other path.
func (p Path) HasPrefix(other Path) bool {
	if p == other || other.Empty() {
		return true
	}
	if strings.HasSuffix(other.String(), separator) {
		// this is not a valid path, but we should have a defined behaviour
		return strings.HasPrefix(p.value, other.value)
	}
	return strings.HasPrefix(p.value, other.value+separator)
}

// Equal checks if the path is the same as the other path.
func (p Path) Equal(other Path) bool {
	return p.value == other.value
}

const lclusterNameFmt string = "[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?"

var lclusterRegExp = regexp.MustCompile("^" + lclusterNameFmt + "(:" + lclusterNameFmt + ")*$")

// IsValid returns true if the path is a Wildcard or a colon separated list of words where each word
// starts with a lower-case letter and contains only lower-case letters, digits and hyphens.
func (p Path) IsValid() bool {
	return p == Wildcard || lclusterRegExp.MatchString(p.value)
}
