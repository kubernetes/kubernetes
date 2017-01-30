// Copyright 2015 The appc Authors
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

package types

import (
	"encoding/json"
	"fmt"
	"sort"
)

var ValidOSArch = map[string][]string{
	"linux":   {"amd64", "i386", "aarch64", "aarch64_be", "armv6l", "armv7l", "armv7b", "ppc64", "ppc64le", "s390x"},
	"freebsd": {"amd64", "i386", "arm"},
	"darwin":  {"x86_64", "i386"},
}

type Labels []Label

type labels Labels

type Label struct {
	Name  ACIdentifier `json:"name"`
	Value string       `json:"value"`
}

// {appc,go}ArchTuple are internal helper types used to translate arch tuple between go and appc
type appcArchTuple struct {
	appcOs   string
	appcArch string
}
type goArchTuple struct {
	goOs         string
	goArch       string
	goArchFlavor string
}

// IsValidOsArch checks if a OS-architecture combination is valid given a map
// of valid OS-architectures
func IsValidOSArch(labels map[ACIdentifier]string, validOSArch map[string][]string) error {
	if os, ok := labels["os"]; ok {
		if validArchs, ok := validOSArch[os]; !ok {
			// Not a whitelisted OS. TODO: how to warn rather than fail?
			validOses := make([]string, 0, len(validOSArch))
			for validOs := range validOSArch {
				validOses = append(validOses, validOs)
			}
			sort.Strings(validOses)
			return fmt.Errorf(`bad os %#v (must be one of: %v)`, os, validOses)
		} else {
			// Whitelisted OS. We check arch here, as arch makes sense only
			// when os is defined.
			if arch, ok := labels["arch"]; ok {
				found := false
				for _, validArch := range validArchs {
					if arch == validArch {
						found = true
						break
					}
				}
				if !found {
					return fmt.Errorf(`bad arch %#v for %v (must be one of: %v)`, arch, os, validArchs)
				}
			}
		}
	}
	return nil
}

func (l Labels) assertValid() error {
	seen := map[ACIdentifier]string{}
	for _, lbl := range l {
		if lbl.Name == "name" {
			return fmt.Errorf(`invalid label name: "name"`)
		}
		_, ok := seen[lbl.Name]
		if ok {
			return fmt.Errorf(`duplicate labels of name %q`, lbl.Name)
		}
		seen[lbl.Name] = lbl.Value
	}
	return IsValidOSArch(seen, ValidOSArch)
}

func (l Labels) MarshalJSON() ([]byte, error) {
	if err := l.assertValid(); err != nil {
		return nil, err
	}
	return json.Marshal(labels(l))
}

func (l *Labels) UnmarshalJSON(data []byte) error {
	var jl labels
	if err := json.Unmarshal(data, &jl); err != nil {
		return err
	}
	nl := Labels(jl)
	if err := nl.assertValid(); err != nil {
		return err
	}
	*l = nl
	return nil
}

// Get retrieves the value of the label by the given name from Labels, if it exists
func (l Labels) Get(name string) (val string, ok bool) {
	for _, lbl := range l {
		if lbl.Name.String() == name {
			return lbl.Value, true
		}
	}
	return "", false
}

// ToMap creates a map[ACIdentifier]string.
func (l Labels) ToMap() map[ACIdentifier]string {
	labelsMap := make(map[ACIdentifier]string)
	for _, lbl := range l {
		labelsMap[lbl.Name] = lbl.Value
	}
	return labelsMap
}

// LabelsFromMap creates Labels from a map[ACIdentifier]string
func LabelsFromMap(labelsMap map[ACIdentifier]string) (Labels, error) {
	labels := Labels{}
	for n, v := range labelsMap {
		labels = append(labels, Label{Name: n, Value: v})
	}
	if err := labels.assertValid(); err != nil {
		return nil, err
	}
	return labels, nil
}

// ToAppcOSArch translates a Golang arch tuple (OS, architecture, flavor) into
// an appc arch tuple (OS, architecture)
func ToAppcOSArch(goOs string, goArch string, goArchFlavor string) (appcOs string, appcArch string, e error) {
	tabularAppcToGo := map[goArchTuple]appcArchTuple{
		{"linux", "amd64", ""}:   {"linux", "amd64"},
		{"linux", "386", ""}:     {"linux", "i386"},
		{"linux", "arm64", ""}:   {"linux", "aarch64"},
		{"linux", "arm", ""}:     {"linux", "armv6l"},
		{"linux", "arm", "6"}:    {"linux", "armv6l"},
		{"linux", "arm", "7"}:    {"linux", "armv7l"},
		{"linux", "ppc64", ""}:   {"linux", "ppc64"},
		{"linux", "ppc64le", ""}: {"linux", "ppc64le"},
		{"linux", "s390x", ""}:   {"linux", "s390x"},

		{"freebsd", "amd64", ""}: {"freebsd", "amd64"},
		{"freebsd", "386", ""}:   {"freebsd", "i386"},
		{"freebsd", "arm", ""}:   {"freebsd", "arm"},
		{"freebsd", "arm", "5"}:  {"freebsd", "arm"},
		{"freebsd", "arm", "6"}:  {"freebsd", "arm"},
		{"freebsd", "arm", "7"}:  {"freebsd", "arm"},

		{"darwin", "amd64", ""}: {"darwin", "x86_64"},
		{"darwin", "386", ""}:   {"darwin", "i386"},
	}
	archTuple, ok := tabularAppcToGo[goArchTuple{goOs, goArch, goArchFlavor}]
	if !ok {
		return "", "", fmt.Errorf("unknown arch tuple: %q - %q - %q", goOs, goArch, goArchFlavor)
	}
	return archTuple.appcOs, archTuple.appcArch, nil
}

// ToGoOSArch translates an appc arch tuple (OS, architecture) into
// a Golang arch tuple (OS, architecture, flavor)
func ToGoOSArch(appcOs string, appcArch string) (goOs string, goArch string, goArchFlavor string, e error) {
	tabularGoToAppc := map[appcArchTuple]goArchTuple{
		// {"linux", "aarch64_be"}: nil,
		// {"linux", "armv7b"}: nil,
		{"linux", "aarch64"}: {"linux", "arm64", ""},
		{"linux", "amd64"}:   {"linux", "amd64", ""},
		{"linux", "armv6l"}:  {"linux", "arm", "6"},
		{"linux", "armv7l"}:  {"linux", "arm", "7"},
		{"linux", "i386"}:    {"linux", "386", ""},
		{"linux", "ppc64"}:   {"linux", "ppc64", ""},
		{"linux", "ppc64le"}: {"linux", "ppc64le", ""},
		{"linux", "s390x"}:   {"linux", "s390x", ""},

		{"freebsd", "amd64"}: {"freebsd", "amd64", ""},
		{"freebsd", "arm"}:   {"freebsd", "arm", "6"},
		{"freebsd", "386"}:   {"freebsd", "i386", ""},

		{"darwin", "amd64"}: {"darwin", "x86_64", ""},
		{"darwin", "386"}:   {"darwin", "i386", ""},
	}

	archTuple, ok := tabularGoToAppc[appcArchTuple{appcOs, appcArch}]
	if !ok {
		return "", "", "", fmt.Errorf("unknown arch tuple: %q - %q", appcOs, appcArch)
	}
	return archTuple.goOs, archTuple.goArch, archTuple.goArchFlavor, nil
}
