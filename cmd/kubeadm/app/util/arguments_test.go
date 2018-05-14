/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"reflect"
	"sort"
	"testing"
)

func TestBuildArgumentListFromMap(t *testing.T) {
	var tests = []struct {
		base      map[string]string
		overrides map[string]string
		expected  []string
	}{
		{ // override an argument from the base
			base: map[string]string{
				"admission-control":     "NamespaceLifecycle",
				"insecure-bind-address": "127.0.0.1",
				"allow-privileged":      "true",
			},
			overrides: map[string]string{
				"admission-control": "NamespaceLifecycle,LimitRanger",
			},
			expected: []string{
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--allow-privileged=true",
				"--insecure-bind-address=127.0.0.1",
			},
		},
		{ // add an argument that is not in base
			base: map[string]string{
				"insecure-bind-address": "127.0.0.1",
				"allow-privileged":      "true",
			},
			overrides: map[string]string{
				"admission-control": "NamespaceLifecycle,LimitRanger",
			},
			expected: []string{
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--allow-privileged=true",
				"--insecure-bind-address=127.0.0.1",
			},
		},
		{ // allow empty strings in base
			base: map[string]string{
				"insecure-bind-address":              "127.0.0.1",
				"allow-privileged":                   "true",
				"something-that-allows-empty-string": "",
			},
			overrides: map[string]string{
				"admission-control": "NamespaceLifecycle,LimitRanger",
			},
			expected: []string{
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--allow-privileged=true",
				"--insecure-bind-address=127.0.0.1",
				"--something-that-allows-empty-string=",
			},
		},
		{ // allow empty strings in overrides
			base: map[string]string{
				"insecure-bind-address":              "127.0.0.1",
				"allow-privileged":                   "true",
				"something-that-allows-empty-string": "foo",
			},
			overrides: map[string]string{
				"admission-control":                  "NamespaceLifecycle,LimitRanger",
				"something-that-allows-empty-string": "",
			},
			expected: []string{
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--something-that-allows-empty-string=",
				"--allow-privileged=true",
				"--insecure-bind-address=127.0.0.1",
			},
		},
	}

	for _, rt := range tests {
		actual := BuildArgumentListFromMap(rt.base, rt.overrides)
		if !reflect.DeepEqual(actual, rt.expected) {
			t.Errorf("failed BuildArgumentListFromMap:\nexpected:\n%v\nsaw:\n%v", rt.expected, actual)
		}
	}
}

func TestParseArgumentListToMap(t *testing.T) {
	var tests = []struct {
		args        []string
		expectedMap map[string]string
	}{
		{
			// normal case
			args: []string{
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--insecure-bind-address=127.0.0.1",
				"--allow-privileged=true",
			},
			expectedMap: map[string]string{
				"admission-control":     "NamespaceLifecycle,LimitRanger",
				"insecure-bind-address": "127.0.0.1",
				"allow-privileged":      "true",
			},
		},
		{
			// test that feature-gates is working
			args: []string{
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--insecure-bind-address=127.0.0.1",
				"--allow-privileged=true",
				"--feature-gates=EnableFoo=true,EnableBar=false",
			},
			expectedMap: map[string]string{
				"admission-control":     "NamespaceLifecycle,LimitRanger",
				"insecure-bind-address": "127.0.0.1",
				"allow-privileged":      "true",
				"feature-gates":         "EnableFoo=true,EnableBar=false",
			},
		},
		{
			// test that a binary can be the first arg
			args: []string{
				"kube-apiserver",
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--insecure-bind-address=127.0.0.1",
				"--allow-privileged=true",
				"--feature-gates=EnableFoo=true,EnableBar=false",
			},
			expectedMap: map[string]string{
				"admission-control":     "NamespaceLifecycle,LimitRanger",
				"insecure-bind-address": "127.0.0.1",
				"allow-privileged":      "true",
				"feature-gates":         "EnableFoo=true,EnableBar=false",
			},
		},
	}

	for _, rt := range tests {
		actualMap := ParseArgumentListToMap(rt.args)
		if !reflect.DeepEqual(actualMap, rt.expectedMap) {
			t.Errorf("failed ParseArgumentListToMap:\nexpected:\n%v\nsaw:\n%v", rt.expectedMap, actualMap)
		}
	}
}

func TestReplaceArgument(t *testing.T) {
	var tests = []struct {
		args         []string
		mutateFunc   func(map[string]string) map[string]string
		expectedArgs []string
	}{
		{
			// normal case
			args: []string{
				"kube-apiserver",
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--insecure-bind-address=127.0.0.1",
				"--allow-privileged=true",
			},
			mutateFunc: func(argMap map[string]string) map[string]string {
				argMap["admission-control"] = "NamespaceLifecycle,LimitRanger,ResourceQuota"
				return argMap
			},
			expectedArgs: []string{
				"kube-apiserver",
				"--admission-control=NamespaceLifecycle,LimitRanger,ResourceQuota",
				"--insecure-bind-address=127.0.0.1",
				"--allow-privileged=true",
			},
		},
		{
			// normal case
			args: []string{
				"kube-apiserver",
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--insecure-bind-address=127.0.0.1",
				"--allow-privileged=true",
			},
			mutateFunc: func(argMap map[string]string) map[string]string {
				argMap["new-arg-here"] = "foo"
				return argMap
			},
			expectedArgs: []string{
				"kube-apiserver",
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--insecure-bind-address=127.0.0.1",
				"--allow-privileged=true",
				"--new-arg-here=foo",
			},
		},
	}

	for _, rt := range tests {
		actualArgs := ReplaceArgument(rt.args, rt.mutateFunc)
		sort.Strings(actualArgs)
		sort.Strings(rt.expectedArgs)
		if !reflect.DeepEqual(actualArgs, rt.expectedArgs) {
			t.Errorf("failed ReplaceArgument:\nexpected:\n%v\nsaw:\n%v", rt.expectedArgs, actualArgs)
		}
	}
}

func TestRoundtrip(t *testing.T) {
	var tests = []struct {
		args []string
	}{
		{
			// normal case
			args: []string{
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--insecure-bind-address=127.0.0.1",
				"--allow-privileged=true",
			},
		},
		{
			// test that feature-gates is working
			args: []string{
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--insecure-bind-address=127.0.0.1",
				"--allow-privileged=true",
				"--feature-gates=EnableFoo=true,EnableBar=false",
			},
		},
	}

	for _, rt := range tests {
		// These two methods should be each other's opposite functions, test that by chaining the methods and see if you get the same result back
		actual := BuildArgumentListFromMap(ParseArgumentListToMap(rt.args), map[string]string{})
		sort.Strings(actual)
		sort.Strings(rt.args)

		if !reflect.DeepEqual(actual, rt.args) {
			t.Errorf("failed TestRoundtrip:\nexpected:\n%v\nsaw:\n%v", rt.args, actual)
		}
	}
}

func TestParseArgument(t *testing.T) {
	var tests = []struct {
		arg         string
		expectedKey string
		expectedVal string
		expectedErr bool
	}{
		{
			// cannot be empty
			arg:         "",
			expectedErr: true,
		},
		{
			// must contain -- and =
			arg:         "a",
			expectedErr: true,
		},
		{
			// must contain -- and =
			arg:         "a-z",
			expectedErr: true,
		},
		{
			// must contain --
			arg:         "a=b",
			expectedErr: true,
		},
		{
			// must contain a key
			arg:         "--=b",
			expectedErr: true,
		},
		{
			// can contain key but no value
			arg:         "--a=",
			expectedKey: "a",
			expectedVal: "",
			expectedErr: false,
		},
		{
			// simple case
			arg:         "--a=b",
			expectedKey: "a",
			expectedVal: "b",
			expectedErr: false,
		},
		{
			// keys/values with '-' should be supported
			arg:         "--very-long-flag-name=some-value",
			expectedKey: "very-long-flag-name",
			expectedVal: "some-value",
			expectedErr: false,
		},
		{
			// numbers should be handled correctly
			arg:         "--some-number=0.2",
			expectedKey: "some-number",
			expectedVal: "0.2",
			expectedErr: false,
		},
		{
			// lists should be handled correctly
			arg:         "--admission-control=foo,bar,baz",
			expectedKey: "admission-control",
			expectedVal: "foo,bar,baz",
			expectedErr: false,
		},
		{
			// more than one '=' should be allowed
			arg:         "--feature-gates=EnableFoo=true,EnableBar=false",
			expectedKey: "feature-gates",
			expectedVal: "EnableFoo=true,EnableBar=false",
			expectedErr: false,
		},
	}

	for _, rt := range tests {
		key, val, actual := parseArgument(rt.arg)
		if (actual != nil) != rt.expectedErr {
			t.Errorf("failed parseArgument:\nexpected error:\n%t\nsaw error:\n%v", rt.expectedErr, actual)
		}
		if (key != rt.expectedKey) || (val != rt.expectedVal) {
			t.Errorf("failed parseArgument:\nexpected key: %s\nsaw key: %s\nexpected value: %s\nsaw value: %s", rt.expectedKey, key, rt.expectedVal, val)
		}
	}
}
