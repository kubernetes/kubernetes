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

	"github.com/docker/engine-api/types"
	"github.com/stretchr/testify/assert"
)

func TestValidateDockerInfo(t *testing.T) {
	v := &DockerValidator{
		Reporter: DefaultReporter,
	}
	spec := &DockerSpec{
		Version:     []string{`1\.(9|\d{2,})\..*`},
		GraphDriver: []string{"driver_1", "driver_2"},
	}
	for _, test := range []struct {
		info types.Info
		err  bool
	}{
		{
			info: types.Info{Driver: "driver_1", ServerVersion: "1.10.1"},
			err:  false,
		},
		{
			info: types.Info{Driver: "bad_driver", ServerVersion: "1.9.1"},
			err:  true,
		},
		{
			info: types.Info{Driver: "driver_2", ServerVersion: "1.8.1"},
			err:  true,
		},
	} {
		err := v.validateDockerInfo(spec, test.info)
		if !test.err {
			assert.Nil(t, err, "Expect error not to occur with docker info %+v", test.info)
		} else {
			assert.NotNil(t, err, "Expect error to occur with docker info %+v", test.info)
		}

	}
}
