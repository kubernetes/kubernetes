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

	"github.com/appc/spec/schema/common"
)

const (
	emptyVolumeDefaultMode = "0755"
	emptyVolumeDefaultUID  = 0
	emptyVolumeDefaultGID  = 0
)

// Volume encapsulates a volume which should be mounted into the filesystem
// of all apps in a PodManifest
type Volume struct {
	Name ACName `json:"name"`
	Kind string `json:"kind"`

	// currently used only by "host"
	// TODO(jonboulle): factor out?
	Source    string `json:"source,omitempty"`
	ReadOnly  *bool  `json:"readOnly,omitempty"`
	Recursive *bool  `json:"recursive,omitempty"`

	// currently used only by "empty"
	Mode *string `json:"mode,omitempty"`
	UID  *int    `json:"uid,omitempty"`
	GID  *int    `json:"gid,omitempty"`
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
		if v.Mode == nil {
			return errors.New("mode for empty volume must be set")
		}
		if v.UID == nil {
			return errors.New("uid for empty volume must be set")
		}
		if v.GID == nil {
			return errors.New("gid for empty volume must be set")
		}
		return nil
	case "host":
		if v.Source == "" {
			return errors.New("source for host volume cannot be empty")
		}
		if v.Mode != nil {
			return errors.New("mode for host volume cannot be set")
		}
		if v.UID != nil {
			return errors.New("uid for host volume cannot be set")
		}
		if v.GID != nil {
			return errors.New("gid for host volume cannot be set")
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
	maybeSetDefaults(&nv)
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
	s := []string{
		v.Name.String(),
		",kind=",
		v.Kind,
	}
	if v.Source != "" {
		s = append(s, ",source=")
		s = append(s, v.Source)
	}
	if v.ReadOnly != nil {
		s = append(s, ",readOnly=")
		s = append(s, strconv.FormatBool(*v.ReadOnly))
	}
	if v.Recursive != nil {
		s = append(s, ",recursive=")
		s = append(s, strconv.FormatBool(*v.Recursive))
	}
	switch v.Kind {
	case "empty":
		if *v.Mode != emptyVolumeDefaultMode {
			s = append(s, ",mode=")
			s = append(s, *v.Mode)
		}
		if *v.UID != emptyVolumeDefaultUID {
			s = append(s, ",uid=")
			s = append(s, strconv.Itoa(*v.UID))
		}
		if *v.GID != emptyVolumeDefaultGID {
			s = append(s, ",gid=")
			s = append(s, strconv.Itoa(*v.GID))
		}
	}
	return strings.Join(s, "")
}

// VolumeFromString takes a command line volume parameter and returns a volume
//
// Example volume parameters:
// 	database,kind=host,source=/tmp,readOnly=true,recursive=true
func VolumeFromString(vp string) (*Volume, error) {
	vp = "name=" + vp
	vpQuery, err := common.MakeQueryString(vp)
	if err != nil {
		return nil, err
	}

	v, err := url.ParseQuery(vpQuery)
	if err != nil {
		return nil, err
	}
	return VolumeFromParams(v)
}

func VolumeFromParams(params map[string][]string) (*Volume, error) {
	var vol Volume
	for key, val := range params {
		val := val
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
		case "recursive":
			rec, err := strconv.ParseBool(val[0])
			if err != nil {
				return nil, err
			}
			vol.Recursive = &rec
		case "mode":
			vol.Mode = &val[0]
		case "uid":
			u, err := strconv.Atoi(val[0])
			if err != nil {
				return nil, err
			}
			vol.UID = &u
		case "gid":
			g, err := strconv.Atoi(val[0])
			if err != nil {
				return nil, err
			}
			vol.GID = &g
		default:
			return nil, fmt.Errorf("unknown volume parameter %q", key)
		}
	}

	maybeSetDefaults(&vol)

	if err := vol.assertValid(); err != nil {
		return nil, err
	}

	return &vol, nil
}

// maybeSetDefaults sets the correct default values for certain fields on a
// Volume if they are not already been set. These fields are not
// pre-populated on all Volumes as the Volume type is polymorphic.
func maybeSetDefaults(vol *Volume) {
	if vol.Kind == "empty" {
		if vol.Mode == nil {
			m := emptyVolumeDefaultMode
			vol.Mode = &m
		}
		if vol.UID == nil {
			u := emptyVolumeDefaultUID
			vol.UID = &u
		}
		if vol.GID == nil {
			g := emptyVolumeDefaultGID
			vol.GID = &g
		}
	}
}
