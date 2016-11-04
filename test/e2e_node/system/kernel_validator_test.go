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
	"bytes"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestValidateKernelVersion(t *testing.T) {
	v := &KernelValidator{}
	// Currently, testRegex is align with DefaultSysSpec.KernelVersion, but in the future
	// they may be different.
	// This is fine, because the test mainly tests the kernel version validation logic,
	// not the DefaultSysSpec. The DefaultSysSpec should be tested with node e2e.
	testRegex := []string{`3\.[1-9][0-9].*`, `4\..*`}
	for _, test := range []struct {
		version string
		err     bool
	}{
		// first version regex matches
		{
			version: "3.19.9-99-test",
			err:     false,
		},
		// one of version regexes matches
		{
			version: "4.4.14+",
			err:     false,
		},
		// no version regex matches
		{
			version: "2.0.0",
			err:     true,
		},
		{
			version: "5.0.0",
			err:     true,
		},
		{
			version: "3.9.0",
			err:     true,
		},
	} {
		v.kernelRelease = test.version
		err := v.validateKernelVersion(KernelSpec{Versions: testRegex})
		if !test.err {
			assert.Nil(t, err, "Expect error not to occur with kernel version %q", test.version)
		} else {
			assert.NotNil(t, err, "Expect error to occur with kenrel version %q", test.version)
		}
	}
}

func TestValidateCachedKernelConfig(t *testing.T) {
	v := &KernelValidator{}
	testKernelSpec := KernelSpec{
		Required: []KernelConfig{{Name: "REQUIRED_1"}, {Name: "REQUIRED_2", Aliases: []string{"ALIASE_REQUIRED_2"}}},
		Optional: []KernelConfig{{Name: "OPTIONAL_1"}, {Name: "OPTIONAL_2"}},
		Forbidden: []KernelConfig{
			{Name: "FORBIDDEN_1", Description: "TEST FORBIDDEN"},
			{Name: "FORBIDDEN_2", Aliases: []string{"ALIASE_FORBIDDEN_2"}},
		},
	}
	for c, test := range []struct {
		desc   string
		config map[string]kConfigOption
		err    bool
	}{
		{
			desc: "meet all required configurations should not report error.",
			config: map[string]kConfigOption{
				"REQUIRED_1": builtIn,
				"REQUIRED_2": asModule,
			},
			err: false,
		},
		{
			desc: "one required configuration disabled should report error.",
			config: map[string]kConfigOption{
				"REQUIRED_1": leftOut,
				"REQUIRED_2": builtIn,
			},
			err: true,
		},
		{
			desc: "one required configuration missing should report error.",
			config: map[string]kConfigOption{
				"REQUIRED_1": builtIn,
			},
			err: true,
		},
		{
			desc: "alias of required configuration should not report error.",
			config: map[string]kConfigOption{
				"REQUIRED_1":        builtIn,
				"ALIASE_REQUIRED_2": asModule,
			},
			err: false,
		},
		{
			desc: "optional configuration set or not should not report error.",
			config: map[string]kConfigOption{
				"REQUIRED_1": builtIn,
				"REQUIRED_2": asModule,
				"OPTIONAL_1": builtIn,
			},
			err: false,
		},
		{
			desc: "forbidden configuration disabled should not report error.",
			config: map[string]kConfigOption{
				"REQUIRED_1":  builtIn,
				"REQUIRED_2":  asModule,
				"FORBIDDEN_1": leftOut,
			},
			err: false,
		},
		{
			desc: "forbidden configuration built-in should report error.",
			config: map[string]kConfigOption{
				"REQUIRED_1":  builtIn,
				"REQUIRED_2":  asModule,
				"FORBIDDEN_1": builtIn,
			},
			err: true,
		},
		{
			desc: "forbidden configuration built as module should report error.",
			config: map[string]kConfigOption{
				"REQUIRED_1":  builtIn,
				"REQUIRED_2":  asModule,
				"FORBIDDEN_1": asModule,
			},
			err: true,
		},
		{
			desc: "alias of forbidden configuration should report error.",
			config: map[string]kConfigOption{
				"REQUIRED_1":         builtIn,
				"REQUIRED_2":         asModule,
				"ALIASE_FORBIDDEN_2": asModule,
			},
			err: true,
		},
	} {
		t.Logf("TestCase #%d %s", c, test.desc)
		// Add kernel config prefix.
		for k, v := range test.config {
			delete(test.config, k)
			test.config[kConfigPrefix+k] = v
		}
		err := v.validateCachedKernelConfig(test.config, testKernelSpec)
		if !test.err {
			assert.Nil(t, err, "Expect error not to occur with kernel config %q", test.config)
		} else {
			assert.NotNil(t, err, "Expect error to occur with kenrel config %q", test.config)
		}
	}
}

func TestValidateParseKernelConfig(t *testing.T) {
	config := `CONFIG_1=y
CONFIG_2=m
CONFIG_3=n`
	expected := map[string]kConfigOption{
		"CONFIG_1": builtIn,
		"CONFIG_2": asModule,
		"CONFIG_3": leftOut,
	}
	v := &KernelValidator{}
	got, err := v.parseKernelConfig(bytes.NewReader([]byte(config)))
	assert.Nil(t, err, "Expect error not to occur when parse kernel configuration %q", config)
	assert.Equal(t, expected, got)
}
