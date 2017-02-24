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
	"errors"
	"fmt"
	"net/url"
	"strconv"

	"github.com/appc/spec/schema/common"
)

// MountPoint is the application-side manifestation of a Volume.
type MountPoint struct {
	Name     ACName `json:"name"`
	Path     string `json:"path"`
	ReadOnly bool   `json:"readOnly,omitempty"`
}

func (mount MountPoint) assertValid() error {
	if mount.Name.Empty() {
		return errors.New("name must be set")
	}
	if len(mount.Path) == 0 {
		return errors.New("path must be set")
	}
	return nil
}

// MountPointFromString takes a command line mountpoint parameter and returns a mountpoint
//
// It is useful for actool patch-manifest --mounts
//
// Example mountpoint parameters:
// 	database,path=/tmp,readOnly=true
func MountPointFromString(mp string) (*MountPoint, error) {
	var mount MountPoint

	mp = "name=" + mp
	mpQuery, err := common.MakeQueryString(mp)
	if err != nil {
		return nil, err
	}

	v, err := url.ParseQuery(mpQuery)
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
			mount.Name = *acn
		case "path":
			mount.Path = val[0]
		case "readOnly":
			ro, err := strconv.ParseBool(val[0])
			if err != nil {
				return nil, err
			}
			mount.ReadOnly = ro
		default:
			return nil, fmt.Errorf("unknown mountpoint parameter %q", key)
		}
	}
	err = mount.assertValid()
	if err != nil {
		return nil, err
	}

	return &mount, nil
}
