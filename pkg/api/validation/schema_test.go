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
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/runtime"
)

func readPod(filename string) (string, error) {
	data, err := ioutil.ReadFile("testdata/" + testapi.Default.Version() + "/" + filename)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func loadSchemaForTest() (Schema, error) {
	pathToSwaggerSpec := "../../../api/swagger-spec/" + testapi.Default.Version() + ".json"
	data, err := ioutil.ReadFile(pathToSwaggerSpec)
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
	apiObjectFuzzer := apitesting.FuzzerFor(nil, "", rand.NewSource(seed))
	for i := 0; i < 5; i++ {
		for _, test := range tests {
			testObj := test.obj
			apiObjectFuzzer.Fuzz(testObj)
			data, err := testapi.Default.Codec().Encode(testObj)
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
			t.Errorf("could not read file: %s", pod)
		}
		err = schema.ValidateBytes([]byte(pod))
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
			t.Errorf("could not read file: %s", test)
		}
		err = schema.ValidateBytes([]byte(pod))
		if err != nil {
			t.Errorf("unexpected error %s, for pod %s", err, pod)
		}
	}
}
