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

func TestValidateOS(t *testing.T) {
	v := &OSValidator{
		Reporter: DefaultReporter,
	}
	specOS := "Linux"
	for _, test := range []struct {
		os  string
		err bool
	}{
		{
			os:  "Linux",
			err: false,
		},
		{
			os:  "Windows",
			err: true,
		},
		{
			os:  "Darwin",
			err: true,
		},
	} {
		err := v.validateOS(test.os, specOS)
		if !test.err {
			assert.Nil(t, err, "Expect error not to occur with os %q", test.os)
		} else {
			assert.NotNil(t, err, "Expect error to occur with os %q", test.os)
		}
	}
}
