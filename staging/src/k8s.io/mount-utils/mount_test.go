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
	"reflect"
	"strings"
	"testing"
)

func TestMakeBindOpts(t *testing.T) {
	tests := []struct {
		mountOption         []string
		isBind              bool
		expectedBindOpts    []string
		expectedRemountOpts []string
	}{
		{
			[]string{"vers=2", "ro", "_netdev"},
			false,
			[]string{},
			[]string{},
		},
		{

			[]string{"bind", "vers=2", "ro", "_netdev"},
			true,
			[]string{"bind", "_netdev"},
			[]string{"bind", "remount", "vers=2", "ro", "_netdev"},
		},
	}
	for _, test := range tests {
		bind, bindOpts, bindRemountOpts := MakeBindOpts(test.mountOption)
		if bind != test.isBind {
			t.Errorf("Expected bind to be %v but got %v", test.isBind, bind)
		}
		if test.isBind {
			if !reflect.DeepEqual(test.expectedBindOpts, bindOpts) {
				t.Errorf("Expected bind mount options to be %+v got %+v", test.expectedBindOpts, bindOpts)
			}
			if !reflect.DeepEqual(test.expectedRemountOpts, bindRemountOpts) {
				t.Errorf("Expected remount options to be %+v got %+v", test.expectedRemountOpts, bindRemountOpts)
			}
		}

	}
}

func TestMakeBindOptsSensitive(t *testing.T) {
	tests := []struct {
		mountOptions                 []string
		sensitiveMountOptions        []string
		isBind                       bool
		expectedBindOpts             []string
		expectedRemountOpts          []string
		expectedSensitiveRemountOpts []string
	}{
		{
			mountOptions:                 []string{"vers=2", "ro", "_netdev"},
			sensitiveMountOptions:        []string{"user=foo", "pass=bar"},
			isBind:                       false,
			expectedBindOpts:             []string{},
			expectedRemountOpts:          []string{},
			expectedSensitiveRemountOpts: []string{"user=foo", "pass=bar"},
		},
		{

			mountOptions:                 []string{"vers=2", "ro", "_netdev"},
			sensitiveMountOptions:        []string{"user=foo", "pass=bar", "bind"},
			isBind:                       true,
			expectedBindOpts:             []string{"bind", "_netdev"},
			expectedRemountOpts:          []string{"bind", "remount", "vers=2", "ro", "_netdev"},
			expectedSensitiveRemountOpts: []string{"user=foo", "pass=bar"},
		},
		{
			mountOptions:                 []string{"vers=2", "remount", "ro", "_netdev"},
			sensitiveMountOptions:        []string{"user=foo", "pass=bar"},
			isBind:                       false,
			expectedBindOpts:             []string{},
			expectedRemountOpts:          []string{},
			expectedSensitiveRemountOpts: []string{"user=foo", "pass=bar"},
		},
		{
			mountOptions:                 []string{"vers=2", "ro", "_netdev"},
			sensitiveMountOptions:        []string{"user=foo", "pass=bar", "remount"},
			isBind:                       false,
			expectedBindOpts:             []string{},
			expectedRemountOpts:          []string{},
			expectedSensitiveRemountOpts: []string{"user=foo", "pass=bar"},
		},
		{

			mountOptions:                 []string{"vers=2", "bind", "ro", "_netdev"},
			sensitiveMountOptions:        []string{"user=foo", "remount", "pass=bar"},
			isBind:                       true,
			expectedBindOpts:             []string{"bind", "_netdev"},
			expectedRemountOpts:          []string{"bind", "remount", "vers=2", "ro", "_netdev"},
			expectedSensitiveRemountOpts: []string{"user=foo", "pass=bar"},
		},
		{

			mountOptions:                 []string{"vers=2", "bind", "ro", "_netdev"},
			sensitiveMountOptions:        []string{"user=foo", "remount", "pass=bar"},
			isBind:                       true,
			expectedBindOpts:             []string{"bind", "_netdev"},
			expectedRemountOpts:          []string{"bind", "remount", "vers=2", "ro", "_netdev"},
			expectedSensitiveRemountOpts: []string{"user=foo", "pass=bar"},
		},
	}
	for _, test := range tests {
		bind, bindOpts, bindRemountOpts, bindRemountSensitiveOpts := MakeBindOptsSensitive(test.mountOptions, test.sensitiveMountOptions)
		if bind != test.isBind {
			t.Errorf("Expected bind to be %v but got %v", test.isBind, bind)
		}
		if test.isBind {
			if !reflect.DeepEqual(test.expectedBindOpts, bindOpts) {
				t.Errorf("Expected bind mount options to be %+v got %+v", test.expectedBindOpts, bindOpts)
			}
			if !reflect.DeepEqual(test.expectedRemountOpts, bindRemountOpts) {
				t.Errorf("Expected remount options to be %+v got %+v", test.expectedRemountOpts, bindRemountOpts)
			}
			if !reflect.DeepEqual(test.expectedSensitiveRemountOpts, bindRemountSensitiveOpts) {
				t.Errorf("Expected sensitive remount options to be %+v got %+v", test.expectedSensitiveRemountOpts, bindRemountSensitiveOpts)
			}
		}

	}
}

func TestOptionsForLogging(t *testing.T) {
	// Arrange
	testcases := []struct {
		options          []string
		sensitiveOptions []string
	}{
		{
			options:          []string{"o1", "o2"},
			sensitiveOptions: []string{"s1"},
		},
		{
			options:          []string{"o1", "o2"},
			sensitiveOptions: []string{"s1", "s2"},
		},
		{
			sensitiveOptions: []string{"s1", "s2"},
		},
		{
			options: []string{"o1", "o2"},
		},
		{},
	}

	for _, v := range testcases {
		// Act
		maskedStr := sanitizedOptionsForLogging(v.options, v.sensitiveOptions)

		// Assert
		for _, sensitiveOption := range v.sensitiveOptions {
			if strings.Contains(maskedStr, sensitiveOption) {
				t.Errorf("Found sensitive log option %q in %q", sensitiveOption, maskedStr)
			}
		}

		actualCount := strings.Count(maskedStr, sensitiveOptionsRemoved)
		expectedCount := len(v.sensitiveOptions)
		if actualCount != expectedCount {
			t.Errorf("Found %v instances of %q in %q. Expected %v", actualCount, sensitiveOptionsRemoved, maskedStr, expectedCount)
		}
	}
}
