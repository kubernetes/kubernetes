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

package v1

import (
	"os"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver-builder/example/pkg/apis"
	v1innsmouth "k8s.io/apiserver-builder/example/pkg/apis/innsmouth/v1"
	"k8s.io/apiserver-builder/example/pkg/client/clientset_generated/clientset"
	v1innsmouthclient "k8s.io/apiserver-builder/example/pkg/client/clientset_generated/clientset/typed/innsmouth/v1"
	"k8s.io/apiserver-builder/example/pkg/openapi"
	"k8s.io/apiserver-builder/pkg/test"
	"k8s.io/client-go/rest"
)

var testenv *test.TestEnvironment
var config *rest.Config
var client *v1innsmouthclient.InnsmouthV1Client

// Do Test Suite setup / teardown
func TestMain(m *testing.M) {
	testenv = test.NewTestEnvironment()
	config = testenv.Start(apis.GetAllApiBuilders(), openapi.GetOpenAPIDefinitions)
	client = clientset.NewForConfigOrDie(config).InnsmouthV1Client
	retCode := m.Run()
	testenv.Stop()
	os.Exit(retCode)
}

func TestCreateDeleteUniversities(t *testing.T) {
	intf := client.DeepOnes("test-create-delete-innsmouth")

	deepone := &v1innsmouth.DeepOne{}
	deepone.Name = "fish-eater"
	deepone.Spec.FishRequired = 150

	// Make sure we can create the resource
	if _, err := intf.Create(deepone); err != nil {
		t.Fatalf("Failed to create %T %v", deepone, err)
	}

	// Make sure we can list the resource
	result, err := intf.List(metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list %T %v", deepone, err)
	}
	if len(result.Items) != 1 {
		t.Fatalf("Expected to find 1 DeepOne, found %d", len(result.Items))
	}
	actual := result.Items[0]
	if actual.Name != deepone.Name {
		t.Fatalf("Expected to find DeepOne named %s, found %s", deepone.Name, actual.Name)
	}
	if actual.Spec.FishRequired != deepone.Spec.FishRequired {
		t.Fatalf("Expected to find FishRequired %d, found %d", deepone.Spec.FishRequired, actual.Spec.FishRequired)
	}

	// Make sure we can delete the resource
	if err = intf.Delete(deepone.Name, &metav1.DeleteOptions{}); err != nil {
		t.Fatalf("Failed to delete %T %v", deepone, err)
	}
	result, err = intf.List(metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list %T %v", deepone, err)
	}
	if len(result.Items) > 0 {
		t.Fatalf("Expected to find 0 DeepOnes, found %d", len(result.Items))
	}
}
