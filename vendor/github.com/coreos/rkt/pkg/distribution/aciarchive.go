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
	"path/filepath"

	"github.com/PuerkitoBio/purell"
	"github.com/appc/spec/schema"
	"github.com/hashicorp/errwrap"
)

const (
	distACIArchiveVersion = 0

	// TypeACIArchive represents the ACIArchive distribution type
	TypeACIArchive Type = "aci-archive"
)

func init() {
	Register(TypeACIArchive, NewACIArchive)
}

// ACIArchive defines a distribution using an ACI file
// The format is:
// cimd:aci-archive:v=0:ArchiveURL?query...
// The distribution type is "archive"
// ArchiveURL must be query escaped
// Examples:
// cimd:aci-archive:v=0:file%3A%2F%2Fabsolute%2Fpath%2Fto%2Ffile
// cimd:aci-archive:v=0:https%3A%2F%2Fexample.com%2Fapp.aci
type ACIArchive struct {
	cimdURL      *url.URL // the cimd URL as explained in the examples
	transportURL *url.URL // the transport URL of the target ACI archive, i.e. https://example.com/app.aci

	str string // the string representation
}

// NewACIArchive creates a new aci-archive distribution from the provided distribution uri.
func NewACIArchive(u *url.URL) (Distribution, error) {
	c, err := parseCIMD(u)
	if err != nil {
		return nil, fmt.Errorf("cannot parse URI: %q: %v", u.String(), err)
	}
	if c.Type != TypeACIArchive {
		return nil, fmt.Errorf("illegal ACI archive distribution type: %q", c.Type)
	}

	// This should be a valid URL
	data, err := url.QueryUnescape(c.Data)
	if err != nil {
		return nil, errwrap.Wrap(fmt.Errorf("error unescaping url %q", c.Data), err)
	}
	aciu, err := url.Parse(data)
	if err != nil {
		return nil, errwrap.Wrap(fmt.Errorf("error parsing url %q", c.Data), err)
	}

	// save the URI as sorted to make it ready for comparison
	purell.NormalizeURL(u, purell.FlagSortQuery)

	str := u.String()
	if path := aciu.String(); filepath.Ext(path) == schema.ACIExtension {
		str = path
	}

	return &ACIArchive{
		cimdURL:      u,
		transportURL: aciu,
		str:          str,
	}, nil
}

// NewACIArchiveFromTransportURL creates a new aci-archive distribution from the provided transport URL
// Example: file:///full/path/to/aci/file.aci -> cimd:aci-archive:v=0:file%3A%2F%2F%2Ffull%2Fpath%2Fto%2Faci%2Ffile.aci
func NewACIArchiveFromTransportURL(u *url.URL) (Distribution, error) {
	urlStr := NewCIMDString(TypeACIArchive, distACIArchiveVersion, url.QueryEscape(u.String()))
	u, err := url.Parse(urlStr)
	if err != nil {
		return nil, err
	}
	return NewACIArchive(u)
}

func (a *ACIArchive) CIMD() *url.URL {
	// Create a copy of the URL.
	u, err := url.Parse(a.cimdURL.String())
	if err != nil {
		panic(err)
	}
	return u
}

func (a *ACIArchive) Equals(d Distribution) bool {
	a2, ok := d.(*ACIArchive)
	if !ok {
		return false
	}

	return a.CIMD().String() == a2.CIMD().String()
}

func (a *ACIArchive) String() string {
	return a.str
}

// TransportURL returns a copy of the transport URL.
func (a *ACIArchive) TransportURL() *url.URL {
	// Create a copy of the transport URL
	tu, err := url.Parse(a.transportURL.String())
	if err != nil {
		panic(fmt.Errorf("invalid transport URL: %v", err))
	}

	return tu
}
