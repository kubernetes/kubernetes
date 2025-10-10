/*
 *
 * Copyright 2021 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package xdsresource

import (
	"net/url"
	"sort"
	"strings"
)

// FederationScheme is the scheme of a federation resource name.
const FederationScheme = "xdstp"

// Name contains the parsed component of an xDS resource name.
//
// An xDS resource name is in the format of
// xdstp://[{authority}]/{resource type}/{id/*}?{context parameters}{#processing directive,*}
//
// See
// https://github.com/cncf/xds/blob/main/proposals/TP1-xds-transport-next.md#uri-based-xds-resource-names
// for details, and examples.
type Name struct {
	Scheme    string
	Authority string
	Type      string
	ID        string

	ContextParams map[string]string

	processingDirective string
}

// ParseName splits the name and returns a struct representation of the Name.
//
// If the name isn't a valid new-style xDS name, field ID is set to the input.
// Note that this is not an error, because we still support the old-style
// resource names (those not starting with "xdstp:").
//
// The caller can tell if the parsing is successful by checking the returned
// Scheme.
func ParseName(name string) *Name {
	if !strings.Contains(name, "://") {
		// Only the long form URL, with ://, is valid.
		return &Name{ID: name}
	}
	parsed, err := url.Parse(name)
	if err != nil {
		return &Name{ID: name}
	}

	ret := &Name{
		Scheme:    parsed.Scheme,
		Authority: parsed.Host,
	}
	split := strings.SplitN(parsed.Path, "/", 3)
	if len(split) < 3 {
		// Path is in the format of "/type/id". There must be at least 3
		// segments after splitting.
		return &Name{ID: name}
	}
	ret.Type = split[1]
	ret.ID = split[2]
	if len(parsed.Query()) != 0 {
		ret.ContextParams = make(map[string]string)
		for k, vs := range parsed.Query() {
			if len(vs) > 0 {
				// We only keep one value of each key. Behavior for multiple values
				// is undefined.
				ret.ContextParams[k] = vs[0]
			}
		}
	}
	// TODO: processing directive (the part comes after "#" in the URL, stored
	// in parsed.RawFragment) is kept but not processed. Add support for that
	// when it's needed.
	ret.processingDirective = parsed.RawFragment
	return ret
}

// String returns a canonicalized string of name. The context parameters are
// sorted by the keys.
func (n *Name) String() string {
	if n.Scheme == "" {
		return n.ID
	}

	// Sort and build query.
	keys := make([]string, 0, len(n.ContextParams))
	for k := range n.ContextParams {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var pairs []string
	for _, k := range keys {
		pairs = append(pairs, strings.Join([]string{k, n.ContextParams[k]}, "="))
	}
	rawQuery := strings.Join(pairs, "&")

	path := n.Type
	if n.ID != "" {
		path = "/" + path + "/" + n.ID
	}

	tempURL := &url.URL{
		Scheme:      n.Scheme,
		Host:        n.Authority,
		Path:        path,
		RawQuery:    rawQuery,
		RawFragment: n.processingDirective,
	}
	return tempURL.String()
}
