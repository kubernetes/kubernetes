/*
Copyright 2014 The Kubernetes Authors.

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

package config

import (
	"reflect"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/diff"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type stepParserTest struct {
	path                    string
	expectedNavigationSteps navigationSteps
	expectedError           string
}

func TestParseWithDots(t *testing.T) {
	test := stepParserTest{
		path: "clusters.my.dot.delimited.name.server",
		expectedNavigationSteps: navigationSteps{
			steps: []navigationStep{
				{"clusters", reflect.TypeOf(make(map[string]*clientcmdapi.Cluster))},
				{"my.dot.delimited.name", reflect.TypeOf(clientcmdapi.Cluster{})},
				{"server", reflect.TypeOf("")},
			},
		},
	}

	test.run(t)
}

func TestParseWithDotsEndingWithName(t *testing.T) {
	test := stepParserTest{
		path: "contexts.10.12.12.12",
		expectedNavigationSteps: navigationSteps{
			steps: []navigationStep{
				{"contexts", reflect.TypeOf(make(map[string]*clientcmdapi.Context))},
				{"10.12.12.12", reflect.TypeOf(clientcmdapi.Context{})},
			},
		},
	}

	test.run(t)
}

func TestParseWithBadValue(t *testing.T) {
	test := stepParserTest{
		path: "user.bad",
		expectedNavigationSteps: navigationSteps{
			steps: []navigationStep{},
		},
		expectedError: "unable to parse user.bad after [] at api.Config",
	}

	test.run(t)
}

func TestParseWithNoMatchingValue(t *testing.T) {
	test := stepParserTest{
		path: "users.jheiss.exec.command",
		expectedNavigationSteps: navigationSteps{
			steps: []navigationStep{},
		},
		expectedError: "unable to parse one or more field values of users.jheiss.exec",
	}

	test.run(t)
}

func (test stepParserTest) run(t *testing.T) {
	actualSteps, err := newNavigationSteps(test.path)
	if len(test.expectedError) != 0 {
		if err == nil {
			t.Errorf("Did not get %v", test.expectedError)
		} else {
			if !strings.Contains(err.Error(), test.expectedError) {
				t.Errorf("Expected %v, but got %v", test.expectedError, err)
			}
		}
		return
	}

	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if !reflect.DeepEqual(test.expectedNavigationSteps, *actualSteps) {
		t.Errorf("diff: %v", diff.ObjectDiff(test.expectedNavigationSteps, *actualSteps))
		t.Errorf("expected: %#v\n actual:   %#v", test.expectedNavigationSteps, *actualSteps)
	}
}
