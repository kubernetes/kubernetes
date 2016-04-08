/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package validation

import (
	"io/ioutil"
	"math/rand"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/runtime"

	"github.com/ghodss/yaml"
)

func readPod(filename string) ([]byte, error) {
	data, err := ioutil.ReadFile("testdata/" + testapi.Default.GroupVersion().Version + "/" + filename)
	if err != nil {
		return nil, err
	}
	return data, nil
}

func readSwaggerFile() ([]byte, error) {
	// TODO this path is broken
	pathToSwaggerSpec := "../../../api/swagger-spec/" + testapi.Default.GroupVersion().Version + ".json"
	return ioutil.ReadFile(pathToSwaggerSpec)
}

func loadSchemaForTest() (Schema, error) {
	data, err := readSwaggerFile()
	if err != nil {
		return nil, err
	}
	return NewSwaggerSchemaFromBytes(data)
}

func TestLoad(t *testing.T) {
	_, err := loadSchemaForTest()
	if err != nil {
		t.Errorf("Failed to load: %v", err)
	}
}

func TestValidateOk(t *testing.T) {
	schema, err := loadSchemaForTest()
	if err != nil {
		t.Errorf("Failed to load: %v", err)
	}
	tests := []struct {
		obj      runtime.Object
		typeName string
	}{
		{obj: &api.Pod{}},
		{obj: &api.Service{}},
		{obj: &api.ReplicationController{}},
	}

	seed := rand.Int63()
	apiObjectFuzzer := apitesting.FuzzerFor(nil, testapi.Default.InternalGroupVersion(), rand.NewSource(seed))
	for i := 0; i < 5; i++ {
		for _, test := range tests {
			testObj := test.obj
			apiObjectFuzzer.Fuzz(testObj)
			data, err := runtime.Encode(testapi.Default.Codec(), testObj)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			err = schema.ValidateBytes(data)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		}
	}
}

func TestInvalid(t *testing.T) {
	schema, err := loadSchemaForTest()
	if err != nil {
		t.Errorf("Failed to load: %v", err)
	}
	tests := []string{
		"invalidPod1.json", // command is a string, instead of []string.
		"invalidPod2.json", // hostPort if of type string, instead of int.
		"invalidPod3.json", // volumes is not an array of objects.
		"invalidPod.yaml",  // command is a string, instead of []string.
	}
	for _, test := range tests {
		pod, err := readPod(test)
		if err != nil {
			t.Errorf("could not read file: %s, err: %v", test, err)
		}
		err = schema.ValidateBytes(pod)
		if err == nil {
			t.Errorf("unexpected non-error, err: %s for pod: %s", err, pod)
		}
	}
}

func TestValid(t *testing.T) {
	schema, err := loadSchemaForTest()
	if err != nil {
		t.Errorf("Failed to load: %v", err)
	}
	tests := []string{
		"validPod.yaml",
	}
	for _, test := range tests {
		pod, err := readPod(test)
		if err != nil {
			t.Errorf("could not read file: %s, err: %v", test, err)
		}
		err = schema.ValidateBytes(pod)
		if err != nil {
			t.Errorf("unexpected error %s, for pod %s", err, pod)
		}
	}
}

func TestVersionRegex(t *testing.T) {
	testCases := []struct {
		typeName string
		match    bool
	}{
		{
			typeName: "v1.Binding",
			match:    true,
		},
		{
			typeName: "v1beta1.Binding",
			match:    true,
		},
		{
			typeName: "Binding",
			match:    false,
		},
	}
	for _, test := range testCases {
		if versionRegexp.MatchString(test.typeName) && !test.match {
			t.Errorf("unexpected error: expect %s not to match the regular expression", test.typeName)
		}
		if !versionRegexp.MatchString(test.typeName) && test.match {
			t.Errorf("unexpected error: expect %s to match the regular expression", test.typeName)
		}
	}
}

// Tests that validation works fine when spec contains "type": "object" instead of "type": "any"
func TestTypeObject(t *testing.T) {
	data, err := readSwaggerFile()
	if err != nil {
		t.Errorf("failed to read swagger file: %v", err)
	}
	// Replace type: "any" in the spec by type: "object" and verify that the validation still passes.
	newData := strings.Replace(string(data), `"type": "any"`, `"type": "object"`, -1)
	schema, err := NewSwaggerSchemaFromBytes([]byte(newData))
	if err != nil {
		t.Errorf("Failed to load: %v", err)
	}
	tests := []string{
		"validPod.yaml",
	}
	for _, test := range tests {
		podBytes, err := readPod(test)
		if err != nil {
			t.Errorf("could not read file: %s, err: %v", test, err)
		}
		// Verify that pod has at least one label (labels are type "any")
		var pod api.Pod
		err = yaml.Unmarshal(podBytes, &pod)
		if err != nil {
			t.Errorf("error in unmarshalling pod: %v", err)
		}
		if len(pod.Labels) == 0 {
			t.Errorf("invalid test input: the pod should have at least one label")
		}
		err = schema.ValidateBytes(podBytes)
		if err != nil {
			t.Errorf("unexpected error %s, for pod %s", err, string(podBytes))
		}
	}
}
