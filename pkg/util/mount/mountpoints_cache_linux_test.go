// +build linux

/*
Copyright 2018 The Kubernetes Authors.

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

package mount

import (
	"errors"
	"reflect"
	"testing"
)

func TestMountPointsCache(t *testing.T) {
	testcases := []struct {
		name                string
		before              func(c *mountPointsCache)
		list                func() ([]MountPoint, error)
		expectedMountPoints []MountPoint
		expectedErr         error
	}{
		{
			name: "simple-get",
			list: func() ([]MountPoint, error) {
				return []MountPoint{
					{"/dev/0", "/path/to/0", "type0", []string{"flags"}, 0, 0},
				}, nil
			},
			expectedMountPoints: []MountPoint{
				{"/dev/0", "/path/to/0", "type0", []string{"flags"}, 0, 0},
			},
			expectedErr: nil,
		},
		{
			name: "second-get-use-cache",
			before: func(c *mountPointsCache) {
				c.Get(func() ([]MountPoint, error) {
					return []MountPoint{
						{"/dev/0", "/path/to/0", "type0", []string{"flags"}, 0, 0},
					}, nil
				})
			},
			list: func() ([]MountPoint, error) {
				return nil, errors.New("could not get mountpoints")
			},
			expectedMountPoints: []MountPoint{
				{"/dev/0", "/path/to/0", "type0", []string{"flags"}, 0, 0},
			},
			expectedErr: nil,
		},
		{
			name: "mark-stale-invalidates-cache",
			before: func(c *mountPointsCache) {
				c.Get(func() ([]MountPoint, error) {
					return []MountPoint{
						{"/dev/0", "/path/to/0", "type0", []string{"flags"}, 0, 0},
					}, nil
				})
				c.markStale()
			},
			list: func() ([]MountPoint, error) {
				return nil, errors.New("could not get mountpoints")
			},
			expectedMountPoints: nil,
			expectedErr:         errors.New("could not get mountpoints"),
		},
	}
	for _, v := range testcases {
		c := newMountPointsCache()
		if v.before != nil {
			v.before(c)
		}
		mps, err := c.Get(v.list)
		if !reflect.DeepEqual(err, v.expectedErr) {
			t.Errorf("test %s: expected error is %v, got %v", v.name, v.expectedErr, err)
		}
		if !reflect.DeepEqual(mps, v.expectedMountPoints) {
			t.Errorf("test %s: expected mountpoints is %v, got %v", v.name, v.expectedMountPoints, mps)
		}
	}
}
