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
	"strings"

	"github.com/PuerkitoBio/purell"
	"github.com/appc/spec/discovery"
	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/common/labelsort"
)

const (
	distAppcVersion = 0

	// TypeAppc represents the Appc distribution type
	TypeAppc Type = "appc"
)

func init() {
	Register(TypeAppc, NewAppc)
}

// Appc defines a distribution using appc image discovery
// Its format is cimd:appc:v=0:name?label01=....&label02=....
// The distribution type is "appc"
// The labels values must be Query escaped
// Example cimd:appc:v=0:coreos.com/etcd?version=v3.0.3&os=linux&arch=amd64
type Appc struct {
	cimd *url.URL
	app  *discovery.App

	str string // the string representation
}

// NewAppc returns an Appc distribution from an Appc distribution URI
func NewAppc(u *url.URL) (Distribution, error) {
	c, err := parseCIMD(u)
	if err != nil {
		return nil, fmt.Errorf("cannot parse URI: %q: %v", u.String(), err)
	}
	if c.Type != TypeAppc {
		return nil, fmt.Errorf("wrong distribution type: %q", c.Type)
	}

	appcStr := c.Data
	for n, v := range u.Query() {
		appcStr += fmt.Sprintf(",%s=%s", n, v[0])
	}
	app, err := discovery.NewAppFromString(appcStr)
	if err != nil {
		return nil, fmt.Errorf("wrong appc image string %q: %v", u.String(), err)
	}

	return NewAppcFromApp(app), nil
}

// NewAppcFromApp returns an Appc distribution from an appc App discovery string
func NewAppcFromApp(app *discovery.App) Distribution {
	rawuri := NewCIMDString(TypeAppc, distAppcVersion, url.QueryEscape(app.Name.String()))

	var version string
	labels := types.Labels{}
	for n, v := range app.Labels {
		if n == "version" {
			version = v
		}

		labels = append(labels, types.Label{Name: n, Value: v})
	}

	if len(labels) > 0 {
		queries := make([]string, len(labels))
		rawuri += "?"
		for i, l := range labels {
			queries[i] = fmt.Sprintf("%s=%s", l.Name, url.QueryEscape(l.Value))
		}
		rawuri += strings.Join(queries, "&")
	}

	u, err := url.Parse(rawuri)
	if err != nil {
		panic(fmt.Errorf("cannot parse URI %q: %v", rawuri, err))
	}

	// save the URI as sorted to make it ready for comparison
	purell.NormalizeURL(u, purell.FlagSortQuery)

	str := app.Name.String()
	if version != "" {
		str += fmt.Sprintf(":%s", version)
	}

	labelsort.By(labelsort.RankedName).Sort(labels)
	for _, l := range labels {
		if l.Name != "version" {
			str += fmt.Sprintf(",%s=%s", l.Name, l.Value)
		}
	}

	return &Appc{
		cimd: u,
		app:  app.Copy(),
		str:  str,
	}
}

func (a *Appc) CIMD() *url.URL {
	// Create a copy of the URL
	u, err := url.Parse(a.cimd.String())
	if err != nil {
		panic(err)
	}
	return u
}

// String returns the simplest appc image format string from the Appc
// distribution.
// To avoid random label position they are lexically ordered giving the
// precedence to version, os, arch (in this order)
func (d *Appc) String() string {
	return d.str
}

func (a *Appc) Equals(d Distribution) bool {
	a2, ok := d.(*Appc)
	if !ok {
		return false
	}

	return a.cimd.String() == a2.cimd.String()
}

// App returns the discovery.App for an Appc distribution. It'll fail
// if the distribution is not of type Appc.
func (a *Appc) App() *discovery.App {
	return a.app.Copy()
}
