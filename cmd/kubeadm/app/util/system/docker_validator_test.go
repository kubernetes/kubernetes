/*
Copyright 2016 The Kubernetes Authors.

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

package system

import (
	"testing"

	"github.com/docker/docker/api/types"
	"github.com/stretchr/testify/assert"
)

func TestValidateDockerInfo(t *testing.T) {
	v := &DockerValidator{
		Reporter: DefaultReporter,
	}
	spec := &DockerSpec{
		Version:     []string{`1\.1[1-3]\..*`, `17\.0[3,6,9]\..*`, `18\.0[6,9]\..*`},
		GraphDriver: []string{"driver_1", "driver_2"},
	}
	for _, test := range []struct {
		name string
		info types.Info
		err  bool
		warn bool
	}{
		{
			name: "unsupported Docker version 1.10.1",
			info: types.Info{Driver: "driver_1", ServerVersion: "1.10.1"},
			err:  true,
			warn: false,
		},
		{
			name: "unsupported driver",
			info: types.Info{Driver: "bad_driver", ServerVersion: "1.11.1"},
			err:  true,
			warn: false,
		},
		{
			name: "valid Docker version 1.11.1",
			info: types.Info{Driver: "driver_1", ServerVersion: "1.11.1"},
			err:  false,
			warn: false,
		},
		{
			name: "valid Docker version 1.12.1",
			info: types.Info{Driver: "driver_2", ServerVersion: "1.12.1"},
			err:  false,
			warn: false,
		},
		{
			name: "valid Docker version 1.13.1",
			info: types.Info{Driver: "driver_2", ServerVersion: "1.13.1"},
			err:  false,
			warn: false,
		},
		{
			name: "valid Docker version 17.03.0-ce",
			info: types.Info{Driver: "driver_2", ServerVersion: "17.03.0-ce"},
			err:  false,
			warn: false,
		},
		{
			name: "valid Docker version 17.06.0-ce",
			info: types.Info{Driver: "driver_2", ServerVersion: "17.06.0-ce"},
			err:  false,
			warn: false,
		},
		{
			name: "valid Docker version 17.09.0-ce",
			info: types.Info{Driver: "driver_2", ServerVersion: "17.09.0-ce"},
			err:  false,
			warn: false,
		},
		{
			name: "valid Docker version 18.06.0-ce",
			info: types.Info{Driver: "driver_2", ServerVersion: "18.06.0-ce"},
			err:  false,
			warn: false,
		},
		{
			name: "valid Docker version 18.09.1-ce",
			info: types.Info{Driver: "driver_2", ServerVersion: "18.09.1-ce"},
			err:  false,
			warn: false,
		},
		{
			name: "Docker version 19.01.0 is not in the list of validated versions",
			info: types.Info{Driver: "driver_2", ServerVersion: "19.01.0"},
			err:  false,
			warn: true,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			warn, err := v.validateDockerInfo(spec, test.info)
			if !test.err {
				assert.Nil(t, err, "Expect error not to occur with docker info %+v", test.info)
			} else {
				assert.NotNil(t, err, "Expect error to occur with docker info %+v", test.info)
			}
			if !test.warn {
				assert.Nil(t, warn, "Expect error not to occur with docker info %+v", test.info)
			} else {
				assert.NotNil(t, warn, "Expect error to occur with docker info %+v", test.info)
			}
		})
	}
}
