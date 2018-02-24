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

package controller

import (
	"reflect"
	"testing"

	apps "k8s.io/api/apps/v1beta2"
	extensions "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
)

var (
	productionLabel         = map[string]string{"type": "production"}
	testLabel               = map[string]string{"type": "testing"}
	productionLabelSelector = labels.Set{"type": "production"}.AsSelector()
	testLabelSelector       = labels.Set{"type": "testing"}.AsSelector()
)

func newDeployment(deploymentName string, label map[string]string, owner metav1.Object) *apps.Deployment {
	d := &apps.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      deploymentName,
			Labels:    label,
			Namespace: metav1.NamespaceDefault,
		},
	}
	if owner != nil {
		d.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(owner, extensions.SchemeGroupVersion.WithKind("Fake"))}
	}
	return d
}

func TestTestClaimDeployments(t *testing.T) {
	controllerKind := schema.GroupVersionKind{}
	type test struct {
		desc        string
		manager     *DeploymentControllerRefManager
		deployments []*apps.Deployment
		filters     []func(*apps.Deployment) bool
		claimed     []*apps.Deployment
		released    []*apps.Deployment
	}

	var tests = []test{
		{
			desc: "Claim deployments with correct label",
			manager: NewDeploymentControllerRefManager(
				fake.NewSimpleClientset(),
				&MyController{},
				productionLabelSelector,
				controllerKind,
				func() error { return nil }),
			deployments: []*apps.Deployment{
				newDeployment("deployment1", productionLabel, nil),
				newDeployment("deployment2", testLabel, nil),
			},
			claimed: []*apps.Deployment{newDeployment("deployment1", productionLabel, nil)},
		},

		func() test {
			controller := MyController{}
			controller2 := MyController{}
			controller.UID = types.UID("123")
			controller2.UID = types.UID("AAAAA")

			return test{
				desc: "Controller can not claim deployments owned by another controller",
				manager: NewDeploymentControllerRefManager(
					fake.NewSimpleClientset(),
					&controller,
					productionLabelSelector,
					controllerKind,
					func() error { return nil }),
				deployments: []*apps.Deployment{
					newDeployment("deployment1", productionLabel, &controller),
					newDeployment("deployment2", productionLabel, &controller2),
				},
				claimed: []*apps.Deployment{newDeployment("deployment1", productionLabel, &controller)},
			}
		}(),
		func() test {
			controller := MyController{}
			controller.UID = types.UID("123")
			return test{
				desc: "Controller releases claimed deployments when selector doesn't match",
				manager: NewDeploymentControllerRefManager(
					fake.NewSimpleClientset(),
					&controller,
					productionLabelSelector,
					controllerKind,
					func() error { return nil }),
				deployments: []*apps.Deployment{
					newDeployment("deployment1", productionLabel, &controller),
					newDeployment("deployment2", testLabel, &controller),
				},
				claimed: []*apps.Deployment{newDeployment("deployment1", productionLabel, &controller)},
			}
		}(),
	}

	for _, test := range tests {
		test.manager.client = fake.NewSimpleClientset(test.deployments[0], test.deployments[1])
		claimed, err := test.manager.ClaimDeployments(test.deployments)
		if err != nil {
			t.Errorf("Test case `%s`, unexpected error: %v", test.desc, err)
		} else if !reflect.DeepEqual(test.claimed, claimed) {
			t.Errorf("Test case `%s`, claimed wrong deployments. Expected %v, got %v",
				test.desc, deploymentsToStringSlice(test.claimed), deploymentsToStringSlice(claimed))
		}
	}

}

func TestAdoptDeployment(t *testing.T) {
	d := newDeployment("test", productionLabel, nil)
	d.UID = types.UID("321")

	c := fake.NewSimpleClientset(d)
	controller := MyController{ObjectMeta: metav1.ObjectMeta{Name: "test"}}
	controller.UID = types.UID("123")

	manager := NewDeploymentControllerRefManager(c, &controller, productionLabelSelector, schema.GroupVersionKind{}, func() error { return nil })
	manager.adoptDeployment(d)

	d.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(&controller, schema.GroupVersionKind{})}

	patch := []byte(`{"metadata":{"ownerReferences":[{"apiVersion":"","kind":"","name":"test","uid":"123","controller":true,"blockOwnerDeletion":true}],"uid":"321"}}`)

	expActions := []core.Action{core.NewPatchAction(schema.GroupVersionResource{Resource: "deployments", Group: "apps", Version: "v1beta2"}, d.Namespace, d.Name, patch)}
	if !reflect.DeepEqual(expActions, c.Actions()) {
		t.Error("Expected and actual API actions are diferent")
		t.Fatal(diff.ObjectGoPrintDiff(expActions, c.Actions()))
	}

}

func TestReleaseDeployment(t *testing.T) {
	controller := MyController{}
	controller.UID = types.UID("123")
	d := newDeployment("test", productionLabel, &controller)
	d.UID = types.UID("321")

	c := fake.NewSimpleClientset(d)

	manager := NewDeploymentControllerRefManager(c, &controller, productionLabelSelector, schema.GroupVersionKind{}, func() error { return nil })
	manager.releaseDeployment(d)

	d.OwnerReferences = []metav1.OwnerReference{}

	patch := []byte(`{"metadata":{"ownerReferences":[{"$patch":"delete","uid":"123"}],"uid":"321"}}`)

	expActions := []core.Action{core.NewPatchAction(schema.GroupVersionResource{Resource: "deployments", Group: "apps", Version: "v1beta2"}, d.Namespace, d.Name, patch)}
	if !reflect.DeepEqual(expActions, c.Actions()) {
		t.Error("Expected and actual API actions are diferent")
		t.Fatal(diff.ObjectGoPrintDiff(expActions, c.Actions()))
	}
}

func deploymentsToStringSlice(deployments []*apps.Deployment) []string {
	var names []string
	for _, d := range deployments {
		names = append(names, d.Name)
	}
	return names
}
