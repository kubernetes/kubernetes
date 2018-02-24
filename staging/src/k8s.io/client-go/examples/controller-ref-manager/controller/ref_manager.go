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
	"fmt"

	apps "k8s.io/api/apps/v1beta2"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/client-go/kubernetes"
	controllertools "k8s.io/client-go/tools/controller"
)

// MyController is simple controller mock
type MyController struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
}

// DeploymentControllerRefManager manage deployments owner references
type DeploymentControllerRefManager struct {
	controllertools.BaseControllerRefManager
	controllerKind schema.GroupVersionKind
	client         kubernetes.Interface
}

// NewDeploymentControllerRefManager returns a DeploymentControllerRefManager that
// exposes methods to manage the controllerRef of deployments.
//
// The CanAdopt() function can be used to perform check prior to the first adoption.
// If CanAdopt() returns a non-nil error, all adoptions will fail.
//
// NOTE: Once CanAdopt() is called, it will not be called again by the same
//       DeploymentControllerRefManager instance.
func NewDeploymentControllerRefManager(
	client kubernetes.Interface,
	controller metav1.Object,
	selector labels.Selector,
	controllerKind schema.GroupVersionKind,
	canAdopt func() error,
) *DeploymentControllerRefManager {
	return &DeploymentControllerRefManager{
		BaseControllerRefManager: controllertools.BaseControllerRefManager{
			Controller:   controller,
			Selector:     selector,
			CanAdoptFunc: canAdopt,
		},
		controllerKind: controllerKind,
		client:         client,
	}
}

// ClaimDeployments tries to take ownership of a list of Deployments.
//
// It will reconcile the following:
//   * Adopt orphans if the selector matches.
//   * Release owned objects if the selector no longer matches.
//
// A non-nil error is returned if some form of reconciliation was attempted and
// failed.
//
// If the error is nil, either the reconciliation succeeded, or no
// reconciliation was necessary. The list of Deployments that you now own is
// returned.
func (m *DeploymentControllerRefManager) ClaimDeployments(
	deployments []*apps.Deployment) ([]*apps.Deployment, error) {

	var claimed []*apps.Deployment
	var errlist []error

	match := func(obj metav1.Object) bool {
		return m.Selector.Matches(labels.Set(obj.GetLabels()))
	}
	adopt := func(obj metav1.Object) error {
		return m.adoptDeployment(obj.(*apps.Deployment))
	}
	release := func(obj metav1.Object) error {
		return m.releaseDeployment(obj.(*apps.Deployment))
	}

	for _, deployment := range deployments {
		ok, err := m.ClaimObject(deployment, match, adopt, release)
		if err != nil {
			errlist = append(errlist, err)
			continue
		}
		if ok {
			claimed = append(claimed, deployment)
		}
	}

	return claimed, utilerrors.NewAggregate(errlist)

}

func (m *DeploymentControllerRefManager) adoptDeployment(d *apps.Deployment) error {
	if err := m.CanAdopt(); err != nil {
		return fmt.Errorf("can't adopt deployment %v/%v (%v): %v", d.Namespace, d.Name, d.UID, err)
	}

	addControllerPatch := fmt.Sprintf(
		`{"metadata":{"ownerReferences":[{"apiVersion":"%s","kind":"%s","name":"%s","uid":"%s","controller":true,"blockOwnerDeletion":true}],"uid":"%s"}}`,
		m.controllerKind.GroupVersion(), m.controllerKind.Kind,
		m.Controller.GetName(), m.Controller.GetUID(), d.UID)

	_, err := m.client.AppsV1beta2().Deployments(d.Namespace).Patch(d.Name, types.StrategicMergePatchType, []byte(addControllerPatch))
	return err
}

func (m *DeploymentControllerRefManager) releaseDeployment(d *apps.Deployment) error {
	deleteOwnerRefPatch := fmt.Sprintf(`{"metadata":{"ownerReferences":[{"$patch":"delete","uid":"%s"}],"uid":"%s"}}`, m.Controller.GetUID(), d.UID)
	_, err := m.client.AppsV1beta2().Deployments(d.Namespace).Patch(d.Name, types.StrategicMergePatchType, []byte(deleteOwnerRefPatch))
	if err != nil {
		if errors.IsNotFound(err) {
			// If the deployment no longer exists, ignore it.
			return nil
		}
		if errors.IsInvalid(err) {
			// Invalid error will be returned in two cases: 1. the deployment
			// has no owner reference, 2. the uid of the deployment doesn't
			// match, which means the deployment is deleted and then recreated.
			// In both cases, the error can be ignored.
			return nil
		}
	}
	return err
}
