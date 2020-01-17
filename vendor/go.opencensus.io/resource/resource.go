// Copyright 2018, OpenCensus Authors
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

// Package resource provides functionality for resource, which capture
// identifying information about the entities for which signals are exported.
package resource

import (
	"context"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Environment variables used by FromEnv to decode a resource.
const (
	EnvVarType   = "OC_RESOURCE_TYPE"
	EnvVarLabels = "OC_RESOURCE_LABELS"
)

// Resource describes an entity about which identifying information and metadata is exposed.
// For example, a type "k8s.io/container" may hold labels describing the pod name and namespace.
type Resource struct {
	Type   string
	Labels map[string]string
}

// EncodeLabels encodes a labels map to a string as provided via the OC_RESOURCE_LABELS environment variable.
func EncodeLabels(labels map[string]string) string {
	sortedKeys := make([]string, 0, len(labels))
	for k := range labels {
		sortedKeys = append(sortedKeys, k)
	}
	sort.Strings(sortedKeys)

	s := ""
	for i, k := range sortedKeys {
		if i > 0 {
			s += ","
		}
		s += k + "=" + strconv.Quote(labels[k])
	}
	return s
}

var labelRegex = regexp.MustCompile(`^\s*([[:ascii:]]{1,256}?)=("[[:ascii:]]{0,256}?")\s*,`)

// DecodeLabels decodes a serialized label map as used in the OC_RESOURCE_LABELS variable.
// A list of labels of the form `<key1>="<value1>",<key2>="<value2>",...` is accepted.
// Domain names and paths are accepted as label keys.
// Most users will want to use FromEnv instead.
func DecodeLabels(s string) (map[string]string, error) {
	m := map[string]string{}
	// Ensure a trailing comma, which allows us to keep the regex simpler
	s = strings.TrimRight(strings.TrimSpace(s), ",") + ","

	for len(s) > 0 {
		match := labelRegex.FindStringSubmatch(s)
		if len(match) == 0 {
			return nil, fmt.Errorf("invalid label formatting, remainder: %s", s)
		}
		v := match[2]
		if v == "" {
			v = match[3]
		} else {
			var err error
			if v, err = strconv.Unquote(v); err != nil {
				return nil, fmt.Errorf("invalid label formatting, remainder: %s, err: %s", s, err)
			}
		}
		m[match[1]] = v

		s = s[len(match[0]):]
	}
	return m, nil
}

// FromEnv is a detector that loads resource information from the OC_RESOURCE_TYPE
// and OC_RESOURCE_labelS environment variables.
func FromEnv(context.Context) (*Resource, error) {
	res := &Resource{
		Type: strings.TrimSpace(os.Getenv(EnvVarType)),
	}
	labels := strings.TrimSpace(os.Getenv(EnvVarLabels))
	if labels == "" {
		return res, nil
	}
	var err error
	if res.Labels, err = DecodeLabels(labels); err != nil {
		return nil, err
	}
	return res, nil
}

var _ Detector = FromEnv

// merge resource information from b into a. In case of a collision, a takes precedence.
func merge(a, b *Resource) *Resource {
	if a == nil {
		return b
	}
	if b == nil {
		return a
	}
	res := &Resource{
		Type:   a.Type,
		Labels: map[string]string{},
	}
	if res.Type == "" {
		res.Type = b.Type
	}
	for k, v := range b.Labels {
		res.Labels[k] = v
	}
	// Labels from resource a overwrite labels from resource b.
	for k, v := range a.Labels {
		res.Labels[k] = v
	}
	return res
}

// Detector attempts to detect resource information.
// If the detector cannot find resource information, the returned resource is nil but no
// error is returned.
// An error is only returned on unexpected failures.
type Detector func(context.Context) (*Resource, error)

// MultiDetector returns a Detector that calls all input detectors in order and
// merges each result with the previous one. In case a type of label key is already set,
// the first set value is takes precedence.
// It returns on the first error that a sub-detector encounters.
func MultiDetector(detectors ...Detector) Detector {
	return func(ctx context.Context) (*Resource, error) {
		return detectAll(ctx, detectors...)
	}
}

// detectall calls all input detectors sequentially an merges each result with the previous one.
// It returns on the first error that a sub-detector encounters.
func detectAll(ctx context.Context, detectors ...Detector) (*Resource, error) {
	var res *Resource
	for _, d := range detectors {
		r, err := d(ctx)
		if err != nil {
			return nil, err
		}
		res = merge(res, r)
	}
	return res, nil
}
