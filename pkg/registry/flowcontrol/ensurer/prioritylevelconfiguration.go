/*
Copyright 2021 The Kubernetes Authors.

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

package ensurer

import (
	"context"
	"fmt"

	flowcontrolv1beta3 "k8s.io/api/flowcontrol/v1beta3"
	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	flowcontrolclient "k8s.io/client-go/kubernetes/typed/flowcontrol/v1beta3"
	flowcontrollisters "k8s.io/client-go/listers/flowcontrol/v1beta3"
	flowcontrolapisv1beta3 "k8s.io/kubernetes/pkg/apis/flowcontrol/v1beta3"
)

// WrapBootstrapPriorityLevelConfigurations creates a generic representation of the given bootstrap objects bound with their operations.
// Every object in `boots` is immutable.
func WrapBootstrapPriorityLevelConfigurations(client flowcontrolclient.PriorityLevelConfigurationInterface, lister flowcontrollisters.PriorityLevelConfigurationLister, boots []*flowcontrolv1beta3.PriorityLevelConfiguration) BootstrapObjects {
	return &bootstrapPriorityLevelConfigurations{
		priorityLevelConfigurationClient: priorityLevelConfigurationClient{
			client: client,
			lister: lister},
		boots: boots,
	}
}

type priorityLevelConfigurationClient struct {
	client flowcontrolclient.PriorityLevelConfigurationInterface
	lister flowcontrollisters.PriorityLevelConfigurationLister
}

type bootstrapPriorityLevelConfigurations struct {
	priorityLevelConfigurationClient

	// Every member is a pointer to immutable content
	boots []*flowcontrolv1beta3.PriorityLevelConfiguration
}

func (*priorityLevelConfigurationClient) typeName() string {
	return "PriorityLevelConfiguration"
}

func (boots *bootstrapPriorityLevelConfigurations) len() int {
	return len(boots.boots)
}

func (boots *bootstrapPriorityLevelConfigurations) get(i int) bootstrapObject {
	return &bootstrapPriorityLevelConfiguration{
		priorityLevelConfigurationClient: &boots.priorityLevelConfigurationClient,
		bootstrap:                        boots.boots[i],
	}
}

func (boots *bootstrapPriorityLevelConfigurations) getExistingObjects() ([]deletable, error) {
	objs, err := boots.lister.List(labels.Everything())
	if err != nil {
		return nil, fmt.Errorf("failed to list PriorityLevelConfiguration objects - %w", err)
	}
	dels := make([]deletable, len(objs))
	for i, obj := range objs {
		dels[i] = &deletablePriorityLevelConfiguration{
			PriorityLevelConfiguration: obj,
			client:                     boots.client,
		}
	}
	return dels, nil
}

type bootstrapPriorityLevelConfiguration struct {
	*priorityLevelConfigurationClient

	// points to immutable content
	bootstrap *flowcontrolv1beta3.PriorityLevelConfiguration
}

func (boot *bootstrapPriorityLevelConfiguration) getName() string {
	return boot.bootstrap.Name
}

func (boot *bootstrapPriorityLevelConfiguration) create(ctx context.Context) error {
	// Copy the object here because the Encoder in the client code may modify the object; see
	// https://github.com/kubernetes/kubernetes/pull/117107
	// and WithVersionEncoder in apimachinery/pkg/runtime/helper.go.
	_, err := boot.client.Create(ctx, boot.bootstrap.DeepCopy(), metav1.CreateOptions{FieldManager: fieldManager})
	return err
}

func (boot *bootstrapPriorityLevelConfiguration) getCurrent() (wantAndHave, error) {
	current, err := boot.lister.Get(boot.bootstrap.Name)
	if err != nil {
		return nil, err
	}
	return &wantAndHavePriorityLevelConfiguration{
		client: boot.client,
		want:   boot.bootstrap,
		have:   current,
	}, nil
}

type wantAndHavePriorityLevelConfiguration struct {
	client flowcontrolclient.PriorityLevelConfigurationInterface
	want   *flowcontrolv1beta3.PriorityLevelConfiguration
	have   *flowcontrolv1beta3.PriorityLevelConfiguration
}

func (wah *wantAndHavePriorityLevelConfiguration) getWant() configurationObject {
	return wah.want
}

func (wah *wantAndHavePriorityLevelConfiguration) getHave() configurationObject {
	return wah.have
}

func (wah *wantAndHavePriorityLevelConfiguration) copyHave(specFromWant bool) updatable {
	copy := wah.have.DeepCopy()
	if specFromWant {
		copy.Spec = *wah.want.Spec.DeepCopy()
	}
	return &updatablePriorityLevelConfiguration{
		PriorityLevelConfiguration: copy,
		client:                     wah.client,
	}
}

func (wah *wantAndHavePriorityLevelConfiguration) specsDiffer() bool {
	return priorityLevelSpecChanged(wah.want, wah.have)
}

func priorityLevelSpecChanged(expected, actual *flowcontrolv1beta3.PriorityLevelConfiguration) bool {
	copiedExpectedPriorityLevel := expected.DeepCopy()
	flowcontrolapisv1beta3.SetObjectDefaults_PriorityLevelConfiguration(copiedExpectedPriorityLevel)
	return !equality.Semantic.DeepEqual(copiedExpectedPriorityLevel.Spec, actual.Spec)
}

type updatablePriorityLevelConfiguration struct {
	*flowcontrolv1beta3.PriorityLevelConfiguration
	client flowcontrolclient.PriorityLevelConfigurationInterface
}

func (u *updatablePriorityLevelConfiguration) update(ctx context.Context) error {
	_, err := u.client.Update(ctx, u.PriorityLevelConfiguration, metav1.UpdateOptions{FieldManager: fieldManager})
	return err
}

type deletablePriorityLevelConfiguration struct {
	*flowcontrolv1beta3.PriorityLevelConfiguration
	client flowcontrolclient.PriorityLevelConfigurationInterface
}

func (dbl *deletablePriorityLevelConfiguration) delete(ctx context.Context /* resourceVersion string */) error {
	// return dbl.client.Delete(context.TODO(), dbl.Name, metav1.DeleteOptions{Preconditions: &metav1.Preconditions{ResourceVersion: &resourceVersion}})
	return dbl.client.Delete(ctx, dbl.Name, metav1.DeleteOptions{})
}
