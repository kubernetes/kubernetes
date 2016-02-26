/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package rkt

import (
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewConfig(t *testing.T) {
	tests := []struct {
		path    string
		options string
		config  *Config
		err     bool
	}{
		{
			"rkt-path",
			"rkt-stage1",
			"--debug=false --dir=/var/lib/rkt --insecure-options=image,ondisk --local-config=/etc/rkt --system-config=/usr/lib/rkt --trust-keys-from-https=false --user-config=/tmp/user",
			&Config{
				Path:               "rkt-path",
				Debug:              false,
				Dir:                "/var/lib/rkt",
				InsecureOptions:    "image,ondisk",
				LocalConfig:        "/etc/rkt",
				SystemConfig:       "/usr/lib/rkt",
				TrustKeysFromHttps: false,
				UserConfig:         "/tmp/user",
				Options:            "--debug=false --dir=/var/lib/rkt --insecure-options=image,ondisk --local-config=/etc/rkt --system-config=/usr/lib/rkt --trust-keys-from-https=false --user-config=/tmp/user",
			},
			false,
		},
		{
			"rkt-path",
			"--non-existed-opt=foo",
			nil,
			true,
		},
	}

	for i, tt := range tests {
		testCaseHint := fmt.Sprintf("test case #%d", i)

		result, err := NewConfig(tt.path, tt.options)
		if tt.err {
			assert.Error(t, err, testCaseHint)
		} else {
			assert.NoError(t, err, testCaseHint)
			assert.Equal(t, tt.config, result, testCaseHint)
			assert.Equal(t, strings.Fields(tt.options), result.buildGlobalOptions())
		}
	}
}
