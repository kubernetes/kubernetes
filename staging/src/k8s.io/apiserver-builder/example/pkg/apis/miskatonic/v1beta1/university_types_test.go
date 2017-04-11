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

package v1beta1

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver-builder/example/pkg/apis"
	v1beta1miskatonic "k8s.io/apiserver-builder/example/pkg/apis/miskatonic/v1beta1"
	"k8s.io/apiserver-builder/example/pkg/client/clientset_generated/clientset"
	v1beta1miskatonicclient "k8s.io/apiserver-builder/example/pkg/client/clientset_generated/clientset/typed/miskatonic/v1beta1"
	"k8s.io/apiserver-builder/example/pkg/openapi"
	"k8s.io/apiserver-builder/pkg/test"
	"k8s.io/client-go/rest"
	"os"
)

var testenv *test.TestEnvironment
var config *rest.Config
var client *v1beta1miskatonicclient.MiskatonicV1beta1Client

// Do Test Suite setup / teardown
func TestMain(m *testing.M) {
	testenv = test.NewTestEnvironment()
	config = testenv.Start(apis.GetAllApiBuilders(), openapi.GetOpenAPIDefinitions)
	client = clientset.NewForConfigOrDie(config).MiskatonicV1beta1Client
	retCode := m.Run()
	testenv.Stop()
	os.Exit(retCode)
}

func TestCreateDeleteUniversities(t *testing.T) {
	intf := client.Universities("test-create-delete-universities")

	univ := &v1beta1miskatonic.University{}
	univ.Name = "miskatonic-university"
	univ.Spec.FacultySize = 7

	// Make sure we can create the resource
	if _, err := intf.Create(univ); err != nil {
		t.Fatalf("Failed to create %T %v", univ, err)
	}

	// Make sure we can list the resource
	result, err := intf.List(metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list Universities %v", err)
	}
	if len(result.Items) != 1 {
		t.Fatalf("Expected to find 1 University, found %d", len(result.Items))
	}
	actual := result.Items[0]
	if actual.Name != univ.Name {
		t.Fatalf("Expected to find University named %s, found %s", univ.Name, actual.Name)
	}
	if actual.Spec.FacultySize != univ.Spec.FacultySize {
		t.Fatalf("Expected to find FacultySize %d, found %d", univ.Spec.FacultySize, actual.Spec.FacultySize)
	}
	if actual.Spec.MaxStudents == nil || *actual.Spec.MaxStudents != *univ.Spec.MaxStudents {
		t.Fatalf("Expected to find MaxStudents %d, found %v", *univ.Spec.MaxStudents, actual.Spec.MaxStudents)
	}

	// Make sure we can delete the resource
	if err = intf.Delete(univ.Name, &metav1.DeleteOptions{}); err != nil {
		t.Fatalf("Failed to delete %T %v", univ, err)
	}
	result, err = intf.List(metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list Universities %v", err)
	}
	if len(result.Items) > 0 {
		t.Fatalf("Expected to find 0 University, found %d", len(result.Items))
	}
}

func TestValidateUniversities(t *testing.T) {
	intf := client.Universities("test-validate-universities")
	univ := &v1beta1miskatonic.University{}
	univ.Name = "miskatonic-university"
	univ.Spec.FacultySize = 7
	maxStudents := 0
	univ.Spec.MaxStudents = &maxStudents

	// To many students - fails validation
	maxStudents = 151
	if _, err := intf.Create(univ); err == nil {
		t.Fatalf("Created University with 151 MaxStudents %v", err)
	}

	// Not enough students - fails validation
	maxStudents = 0
	if _, err := intf.Create(univ); err == nil {
		t.Fatalf("Created University with 0 MaxStudents %v", err)
	}

	// Just right number of students
	maxStudents = 150
	if _, err := intf.Create(univ); err != nil {
		t.Fatalf("Failed to create %T %v", univ, err)
	}

	// Clean up
	if err := intf.Delete(univ.Name, &metav1.DeleteOptions{}); err != nil {
		t.Fatalf("Failed to delete %T %v", univ, err)
	}
}

func TestScaleUniversities(t *testing.T) {
	namespace := "test-scale-universities"
	intf := client.Universities(namespace)

	univ := &v1beta1miskatonic.University{}
	univ.Name = "miskatonic-university"
	univ.Spec.FacultySize = 7
	maxStudents := 150
	univ.Spec.MaxStudents = &maxStudents

	if _, err := intf.Create(univ); err != nil {
		t.Fatalf("Failed to create %T %v", univ, err)
	}

	// Verify the original size
	if newUniv, err := intf.Get(univ.Name, metav1.GetOptions{}); err != nil {
		t.Fatalf("Failed to create University %v", err)
	} else if newUniv.Spec.FacultySize != univ.Spec.FacultySize {
		t.Fatalf("Expected FacultySize %d got %d", univ.Spec.FacultySize, newUniv.Spec.FacultySize)
	}

	// Scale the university
	scale := &v1beta1miskatonic.Scale{
		Faculty: 30,
	}
	scale.Name = univ.Name
	restClient := client.RESTClient()
	err := restClient.Post().Namespace(namespace).
		Name(univ.Name).
		Resource("universities").
		SubResource("scale").
		Body(scale).Do().Error()
	if err != nil {
		t.Fatalf("Failed to create University %v", err)
	}

	// Verify the new size
	if newUniv, err := intf.Get(univ.Name, metav1.GetOptions{}); err != nil {
		t.Fatalf("Failed to create University %v", err)
	} else if newUniv.Spec.FacultySize != 30 {
		t.Fatalf("Expected FacultySize %d got %d", 30, newUniv.Spec.FacultySize)
	}

	// Clean up
	intf.Delete(univ.Name, &metav1.DeleteOptions{})
}
