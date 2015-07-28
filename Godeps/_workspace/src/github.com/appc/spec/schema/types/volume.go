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
	"errors"
	"fmt"
	"net/url"
	"path/filepath"
	"strconv"
	"strings"
)

// Volume encapsulates a volume which should be mounted into the filesystem
// of all apps in a PodManifest
type Volume struct {
	Name ACName `json:"name"`
	Kind string `json:"kind"`

	// currently used only by "host"
	// TODO(jonboulle): factor out?
	Source   string `json:"source,omitempty"`
	ReadOnly *bool  `json:"readOnly,omitempty"`
}

type volume Volume

func (v Volume) assertValid() error {
	if v.Name.Empty() {
		return errors.New("name must be set")
	}

	switch v.Kind {
	case "empty":
		if v.Source != "" {
			return errors.New("source for empty volume must be empty")
		}
		return nil
	case "host":
		if v.Source == "" {
			return errors.New("source for host volume cannot be empty")
		}
		if !filepath.IsAbs(v.Source) {
			return errors.New("source for host volume must be absolute path")
		}
		return nil
	default:
		return errors.New(`unrecognized volume kind: should be one of "empty", "host"`)
	}
}

func (v *Volume) UnmarshalJSON(data []byte) error {
	var vv volume
	if err := json.Unmarshal(data, &vv); err != nil {
		return err
	}
	nv := Volume(vv)
	if err := nv.assertValid(); err != nil {
		return err
	}
	*v = nv
	return nil
}

func (v Volume) MarshalJSON() ([]byte, error) {
	if err := v.assertValid(); err != nil {
		return nil, err
	}
	return json.Marshal(volume(v))
}

func (v Volume) String() string {
	s := fmt.Sprintf("%s,kind=%s,readOnly=%t", v.Name, v.Kind, *v.ReadOnly)
	if v.Source != "" {
		s = s + fmt.Sprintf("source=%s", v.Source)
	}
	return s
}

// VolumeFromString takes a command line volume parameter and returns a volume
//
// Example volume parameters:
// 	database,kind=host,source=/tmp,readOnly=true
func VolumeFromString(vp string) (*Volume, error) {
	var vol Volume

	vp = "name=" + vp
	v, err := url.ParseQuery(strings.Replace(vp, ",", "&", -1))
	if err != nil {
		return nil, err
	}
	for key, val := range v {
		if len(val) > 1 {
			return nil, fmt.Errorf("label %s with multiple values %q", key, val)
		}

		switch key {
		case "name":
			acn, err := NewACName(val[0])
			if err != nil {
				return nil, err
			}
			vol.Name = *acn
		case "kind":
			vol.Kind = val[0]
		case "source":
			vol.Source = val[0]
		case "readOnly":
			ro, err := strconv.ParseBool(val[0])
			if err != nil {
				return nil, err
			}
			vol.ReadOnly = &ro
		default:
			return nil, fmt.Errorf("unknown volume parameter %q", key)
		}
	}
	err = vol.assertValid()
	if err != nil {
		return nil, err
	}

	return &vol, nil
}
