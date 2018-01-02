// Copyright 2014 The rkt Authors
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

package common

import (
	"reflect"
	"testing"

	"github.com/kr/pretty"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
)

func TestGenerateMounts(t *testing.T) {
	tests := []struct {
		ra         *schema.RuntimeApp
		vols       []types.Volume
		fromDocker bool
		hasErr     bool
		expected   []Mount
	}{
		{ // Test matching ra.mount to volume via name w/o/ mountpoint
			ra: &schema.RuntimeApp{
				Mounts: []schema.Mount{
					{
						Volume: *types.MustACName("foo-mount"),
						Path:   "/app/foo",
					},
				},
				App: &types.App{
					MountPoints: nil,
				},
			},
			vols: []types.Volume{
				{
					Name:     *types.MustACName("foo-mount"),
					Kind:     "host",
					Source:   "/host/foo",
					ReadOnly: &falseVar,
				},
			},
			fromDocker: false,
			hasErr:     false,
			expected: []Mount{
				{
					Mount: schema.Mount{
						Volume: *types.MustACName("foo-mount"),
						Path:   "/app/foo",
					},
					Volume: types.Volume{
						Name:     *types.MustACName("foo-mount"),
						Kind:     "host",
						Source:   "/host/foo",
						ReadOnly: &falseVar,
					},
					DockerImplicit: false,
					ReadOnly:       false,
				},
			},
		},
		{ // Test matching app's mountpoint to a volume w/o a mount
			ra: &schema.RuntimeApp{
				Mounts: nil,
				App: &types.App{
					MountPoints: []types.MountPoint{
						{
							Name:     *types.MustACName("foo-mp"),
							Path:     "/app/foo-mp",
							ReadOnly: false,
						},
					},
				},
			},
			vols: []types.Volume{
				{
					Name:     *types.MustACName("foo-mount"),
					Kind:     "host",
					Source:   "/host/foo",
					ReadOnly: &falseVar,
				},
				{
					Name:     *types.MustACName("foo-mp"),
					Kind:     "host",
					Source:   "/host/bar",
					ReadOnly: &falseVar,
				},
			},
			fromDocker: false,
			hasErr:     false,
			expected: []Mount{
				{
					Mount: schema.Mount{
						Volume: *types.MustACName("foo-mp"),
						Path:   "/app/foo-mp",
					},
					Volume: types.Volume{
						Name:     *types.MustACName("foo-mp"),
						Kind:     "host",
						Source:   "/host/bar",
						ReadOnly: &falseVar,
					},
					DockerImplicit: false,
					ReadOnly:       false,
				},
			},
		},
		{ // Test that app's Mount can override the volume
			ra: &schema.RuntimeApp{
				Mounts: []schema.Mount{
					{
						Volume: *types.MustACName("foo-mount"),
						Path:   "/app/foo",
						AppVolume: &types.Volume{
							Name:     *types.MustACName("foo-mount"),
							Kind:     "host",
							Source:   "/host/overridden",
							ReadOnly: nil,
						},
					},
				},

				App: &types.App{
					MountPoints: nil,
				},
			},
			vols: []types.Volume{
				{
					Name:     *types.MustACName("foo-mount"),
					Kind:     "host",
					Source:   "/host/foo",
					ReadOnly: &falseVar,
				},
				{
					Name:     *types.MustACName("foo-mp"),
					Kind:     "host",
					Source:   "/host/bar",
					ReadOnly: &falseVar,
				},
			},
			fromDocker: false,
			hasErr:     false,
			expected: []Mount{
				{
					Mount: schema.Mount{
						Volume: *types.MustACName("foo-mount"),
						Path:   "/app/foo",
						AppVolume: &types.Volume{
							Name:     *types.MustACName("foo-mount"),
							Kind:     "host",
							Source:   "/host/overridden",
							ReadOnly: nil,
						},
					},
					Volume: types.Volume{
						Name:     *types.MustACName("foo-mount"),
						Kind:     "host",
						Source:   "/host/overridden",
						ReadOnly: nil,
					},
					DockerImplicit: false,
					ReadOnly:       false,
				},
			},
		},
	}

	for i, tt := range tests {
		result, err := GenerateMounts(tt.ra, tt.vols, tt.fromDocker)
		if (err != nil) != tt.hasErr {
			t.Errorf("test %d expected error status %t, didn't get it", i, tt.hasErr)
		}
		if !reflect.DeepEqual(result, tt.expected) {
			t.Errorf("test %d, result != expected, %+v", i, pretty.Diff(tt.expected, result))
		}
	}
}
