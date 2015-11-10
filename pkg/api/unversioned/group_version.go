/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package unversioned

import (
	"encoding/json"
	"fmt"
	"strings"
)

// TODO: We need to remove the GroupVersion in types.go. We use the name GroupVersion here temporarily.
// GroupVersion contains the "group" and the "version", which uniquely identifies the API.
type GroupVersion struct {
	Group   string
	Version string
}

// String puts "group" and "version" into a single "group/version" string. For the legacy v1
// it returns "v1".
func (gv *GroupVersion) String() string {
	// special case of "v1" for backward compatibility
	if gv.Group == "" && gv.Version == "v1" {
		return gv.Version
	} else {
		return gv.Group + "/" + gv.Version
	}
}

// ParseGroupVersion turns "group/version" string into a GroupVersion struct. It reports error
// if it cannot parse the string.
func ParseGroupVersion(gv string) (GroupVersion, error) {
	s := strings.Split(gv, "/")
	// "v1" is the only special case. Otherwise GroupVersion is expected to contain
	// one "/" dividing the string into two parts.
	switch {
	case len(s) == 1 && gv == "v1":
		return GroupVersion{"", "v1"}, nil
	case len(s) == 2:
		return GroupVersion{s[0], s[1]}, nil
	default:
		return GroupVersion{}, fmt.Errorf("Unexpected GroupVersion string: %v", gv)
	}
}

// MarshalJSON implements the json.Marshaller interface.
func (gv GroupVersion) MarshalJSON() ([]byte, error) {
	s := gv.String()
	if strings.Count(s, "/") > 1 {
		return []byte{}, fmt.Errorf("illegal GroupVersion %v: contains more than one /", s)
	}
	return json.Marshal(s)
}

func (gv *GroupVersion) unmarshal(value []byte) error {
	var s string
	if err := json.Unmarshal(value, &s); err != nil {
		return err
	}
	parsed, err := ParseGroupVersion(s)
	if err != nil {
		return err
	}
	*gv = parsed
	return nil
}

// UnmarshalJSON implements the json.Unmarshaller interface.
func (gv *GroupVersion) UnmarshalJSON(value []byte) error {
	return gv.unmarshal(value)
}

// UnmarshalTEXT implements the Ugorji's encoding.TextUnmarshaler interface.
func (gv *GroupVersion) UnmarshalText(value []byte) error {
	return gv.unmarshal(value)
}
