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

	"github.com/stretchr/testify/assert"
)

func TestValidateCgroupSubsystem(t *testing.T) {
	v := &CgroupsValidator{
		Reporter: DefaultReporter,
	}
	cgroupSpec := []string{"system1", "system2"}
	for desc, test := range map[string]struct {
		cgroupSpec []string
		subsystems []string
		err        bool
	}{
		"missing cgroup subsystem should report error": {
			subsystems: []string{"system1"},
			err:        true,
		},
		"extra cgroup subsystems should not report error": {
			subsystems: []string{"system1", "system2", "system3"},
			err:        false,
		},
		"subsystems the same with spec should not report error": {
			subsystems: []string{"system1", "system2"},
			err:        false,
		},
	} {
		err := v.validateCgroupSubsystems(cgroupSpec, test.subsystems)
		if !test.err {
			assert.Nil(t, err, "%q: Expect error not to occur with cgroup", desc)
		} else {
			assert.NotNil(t, err, "%q: Expect error to occur with docker info", desc)
		}

	}
}
