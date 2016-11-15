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

package validation

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/runtime"
	k8syaml "k8s.io/kubernetes/pkg/util/yaml"

	"github.com/ghodss/yaml"
)

func readPod(filename string) ([]byte, error) {
	data, err := ioutil.ReadFile("testdata/" + registered.GroupOrDie(api.GroupName).GroupVersion.Version + "/" + filename)
	if err != nil {
		return nil, err
	}
	return data, nil
}

func readSwaggerFile() ([]byte, error) {
	return readSwaggerApiFile(testapi.Default)
}

func readSwaggerApiFile(group testapi.TestGroup) ([]byte, error) {
	// TODO: Figure out a better way of finding these files
	var pathToSwaggerSpec string
	if group.GroupVersion().Group == "" {
		pathToSwaggerSpec = "../../../api/swagger-spec/" + group.GroupVersion().Version + ".json"
	} else {
		pathToSwaggerSpec = "../../../api/swagger-spec/" + group.GroupVersion().Group + "_" + group.GroupVersion().Version + ".json"
	}

	return ioutil.ReadFile(pathToSwaggerSpec)
}

// Mock delegating Schema.  Not a full fake impl.
type Factory struct {
	defaultSchema    Schema
	extensionsSchema Schema
}

var _ Schema = &Factory{}

// TODO: Consider using a mocking library instead or fully fleshing this out into a fake impl and putting it in some
// generally available location
func (f *Factory) ValidateBytes(data []byte) error {
	var obj interface{}
	out, err := k8syaml.ToJSON(data)
	if err != nil {
		return err
	}
	data = out
	if err := json.Unmarshal(data, &obj); err != nil {
		return err
	}
	fields, ok := obj.(map[string]interface{})
	if !ok {
		return fmt.Errorf("error in unmarshaling data %s", string(data))
	}
	// Note: This only supports the 2 api versions we expect from the test it is currently supporting.
	groupVersion := fields["apiVersion"]
	switch groupVersion {
	case "v1":
		return f.defaultSchema.ValidateBytes(data)
	case "extensions/v1beta1":
		return f.extensionsSchema.ValidateBytes(data)
	default:
		return fmt.Errorf("Unsupported API version %s", groupVersion)
	}
}

func loadSchemaForTest() (Schema, error) {
	data, err := readSwaggerFile()
	if err != nil {
		return nil, err
	}
	return NewSwaggerSchemaFromBytes(data, nil)
}

func loadSchemaForTestWithFactory(group testapi.TestGroup, factory Schema) (Schema, error) {
	data, err := readSwaggerApiFile(group)
	if err != nil {
		return nil, err
	}
	return NewSwaggerSchemaFromBytes(data, factory)
}

func NewFactory() (*Factory, error) {
	f := &Factory{}
	defaultSchema, err := loadSchemaForTestWithFactory(testapi.Default, f)
	if err != nil {
		return nil, err
	}
	f.defaultSchema = defaultSchema
	extensionSchema, err := loadSchemaForTestWithFactory(testapi.Extensions, f)
	if err != nil {
		return nil, err
	}
	f.extensionsSchema = extensionSchema
	return f, nil
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
		t.Fatalf("Failed to load: %v", err)
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

func TestValidateDifferentApiVersions(t *testing.T) {
	schema, err := loadSchemaForTest()
	if err != nil {
		t.Fatalf("Failed to load: %v", err)
	}

	pod := &api.Pod{}
	pod.APIVersion = "v1"
	pod.Kind = "Pod"

	deployment := &extensions.Deployment{}
	deployment.APIVersion = "extensions/v1beta1"
	deployment.Kind = "Deployment"

	list := &api.List{}
	list.APIVersion = "v1"
	list.Kind = "List"
	list.Items = []runtime.Object{pod, deployment}
	bytes, err := json.Marshal(list)
	if err != nil {
		t.Error(err)
	}
	err = schema.ValidateBytes(bytes)
	if err == nil {
		t.Error(fmt.Errorf("Expected error when validating different api version and no delegate exists."))
	}
	f, err := NewFactory()
	if err != nil {
		t.Error(fmt.Errorf("Failed to create Schema factory %v.", err))
	}
	err = f.ValidateBytes(bytes)
	if err != nil {
		t.Error(fmt.Errorf("Failed to validate object with multiple ApiGroups: %v.", err))
	}
}

func TestInvalid(t *testing.T) {
	schema, err := loadSchemaForTest()
	if err != nil {
		t.Fatalf("Failed to load: %v", err)
	}
	tests := []string{
		"invalidPod1.json", // command is a string, instead of []string.
		"invalidPod2.json", // hostPort if of type string, instead of int.
		"invalidPod3.json", // volumes is not an array of objects.
		"invalidPod4.yaml", // string list with empty string.
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
		t.Fatalf("Failed to load: %v", err)
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
			t.Errorf("unexpected error: %s, for pod %s", err, pod)
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

// Tests that validation works fine when spec contains "type": "any" instead of "type": "object"
// Ref: https://github.com/kubernetes/kubernetes/issues/24309
func TestTypeAny(t *testing.T) {
	data, err := readSwaggerFile()
	if err != nil {
		t.Errorf("failed to read swagger file: %v", err)
	}
	// Replace type: "any" in the spec by type: "object" and verify that the validation still passes.
	newData := strings.Replace(string(data), `"type": "object"`, `"type": "any"`, -1)
	schema, err := NewSwaggerSchemaFromBytes([]byte(newData), nil)
	if err != nil {
		t.Fatalf("Failed to load: %v", err)
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
			t.Errorf("unexpected error: %s, for pod %s", err, string(podBytes))
		}
	}
}

func TestValidateDuplicateLabelsFailCases(t *testing.T) {
	strs := []string{
		`{
	"metadata": {
		"labels": {
			"foo": "bar",
			"foo": "baz"
		}
	}
}`,
		`{
	"metadata": {
		"annotations": {
			"foo": "bar",
			"foo": "baz"
		}
	}
}`,
		`{
	"metadata": {
		"labels": {
			"foo": "blah"
		},
		"annotations": {
			"foo": "bar",
			"foo": "baz"
		}
	}
}`,
	}
	schema := NoDoubleKeySchema{}
	for _, str := range strs {
		err := schema.ValidateBytes([]byte(str))
		if err == nil {
			t.Errorf("Unexpected non-error %s", str)
		}
	}
}

func TestValidateDuplicateLabelsPassCases(t *testing.T) {
	strs := []string{
		`{
	"metadata": {
		"labels": {
			"foo": "bar"
		},
		"annotations": {
			"foo": "baz"
		}
	}
}`,
		`{
	"metadata": {}
}`,
		`{
	"metadata": {
		"labels": {}
	}
}`,
	}
	schema := NoDoubleKeySchema{}
	for _, str := range strs {
		err := schema.ValidateBytes([]byte(str))
		if err != nil {
			t.Errorf("Unexpected error: %v %s", err, str)
		}
	}
}

type AlwaysInvalidSchema struct{}

func (AlwaysInvalidSchema) ValidateBytes([]byte) error {
	return fmt.Errorf("Always invalid!")
}

func TestConjunctiveSchema(t *testing.T) {
	tests := []struct {
		schemas    []Schema
		shouldPass bool
		name       string
	}{
		{
			schemas:    []Schema{NullSchema{}, NullSchema{}},
			shouldPass: true,
			name:       "all pass",
		},
		{
			schemas:    []Schema{NullSchema{}, AlwaysInvalidSchema{}},
			shouldPass: false,
			name:       "one fail",
		},
		{
			schemas:    []Schema{AlwaysInvalidSchema{}, AlwaysInvalidSchema{}},
			shouldPass: false,
			name:       "all fail",
		},
		{
			schemas:    []Schema{},
			shouldPass: true,
			name:       "empty",
		},
	}

	for _, test := range tests {
		schema := ConjunctiveSchema(test.schemas)
		err := schema.ValidateBytes([]byte{})
		if err != nil && test.shouldPass {
			t.Errorf("Unexpected error: %v in %s", err, test.name)
		}
		if err == nil && !test.shouldPass {
			t.Errorf("Unexpected non-error: %s", test.name)
		}
	}
}
