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

	flowcontrolv1beta2 "k8s.io/api/flowcontrol/v1beta2"
	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	flowcontrolclient "k8s.io/client-go/kubernetes/typed/flowcontrol/v1beta2"
	flowcontrollisters "k8s.io/client-go/listers/flowcontrol/v1beta2"
	flowcontrolapisv1beta2 "k8s.io/kubernetes/pkg/apis/flowcontrol/v1beta2"
)

// WrapBootstrapFlowSchemas creates a generic representation of the given bootstrap objects bound with their operations
func WrapBootstrapFlowSchemas(client flowcontrolclient.FlowSchemaInterface, lister flowcontrollisters.FlowSchemaLister, boots []*flowcontrolv1beta2.FlowSchema) BootstrapObjects {
	return &bootstrapFlowSchemas{
		client: client,
		lister: lister,
		boots:  boots,
	}
}

type bootstrapFlowSchemas struct {
	client flowcontrolclient.FlowSchemaInterface
	lister flowcontrollisters.FlowSchemaLister
	boots  []*flowcontrolv1beta2.FlowSchema
}

func (*bootstrapFlowSchemas) typeName() string {
	return "FlowSchema"
}

func (boots *bootstrapFlowSchemas) len() int {
	return len(boots.boots)
}

func (boots *bootstrapFlowSchemas) get(i int) bootstrapObject {
	return &bootstrapFlowSchema{
		bootstrapFlowSchemas: boots,
		bootstrap:            boots.boots[i],
	}
}

func (boots *bootstrapFlowSchemas) getExistingObjects() ([]deletable, error) {
	objs, err := boots.lister.List(labels.Everything())
	if err != nil {
		return nil, fmt.Errorf("failed to list FlowSchema objects - %w", err)
	}
	dels := make([]deletable, len(objs))
	for i, obj := range objs {
		dels[i] = &deletableFlowSchema{
			FlowSchema: obj,
			client:     boots.client,
		}
	}
	return dels, nil
}

type bootstrapFlowSchema struct {
	*bootstrapFlowSchemas
	bootstrap *flowcontrolv1beta2.FlowSchema
}

func (boot *bootstrapFlowSchema) getName() string {
	return boot.bootstrap.Name
}

func (boot *bootstrapFlowSchema) create() error {
	_, err := boot.bootstrapFlowSchemas.client.Create(context.TODO(), boot.bootstrap, metav1.CreateOptions{FieldManager: fieldManager})
	return err
}

func (boot *bootstrapFlowSchema) getCurrent() (wantAndHave, error) {
	current, err := boot.bootstrapFlowSchemas.lister.Get(boot.bootstrap.Name)
	if err != nil {
		return nil, err
	}
	return &wantAndHaveFlowSchema{
		client: boot.bootstrapFlowSchemas.client,
		want:   boot.bootstrap,
		have:   current,
	}, nil
}

type wantAndHaveFlowSchema struct {
	client flowcontrolclient.FlowSchemaInterface
	want   *flowcontrolv1beta2.FlowSchema
	have   *flowcontrolv1beta2.FlowSchema
}

func (wah *wantAndHaveFlowSchema) getWant() configurationObject {
	return wah.want
}

func (wah *wantAndHaveFlowSchema) getHave() configurationObject {
	return wah.have
}

func (wah *wantAndHaveFlowSchema) copyHave(specFromWant bool) updatable {
	copy := wah.have.DeepCopy()
	if specFromWant {
		copy.Spec = *wah.want.Spec.DeepCopy()
	}
	return &updatableFlowSchema{
		FlowSchema: copy,
		client:     wah.client,
	}
}

func (wah *wantAndHaveFlowSchema) specsDiffer() bool {
	return flowSchemaSpecChanged(wah.want, wah.have)
}

func flowSchemaSpecChanged(expected, actual *flowcontrolv1beta2.FlowSchema) bool {
	copiedExpectedFlowSchema := expected.DeepCopy()
	flowcontrolapisv1beta2.SetObjectDefaults_FlowSchema(copiedExpectedFlowSchema)
	return !equality.Semantic.DeepEqual(copiedExpectedFlowSchema.Spec, actual.Spec)
}

type updatableFlowSchema struct {
	*flowcontrolv1beta2.FlowSchema
	client flowcontrolclient.FlowSchemaInterface
}

func (u *updatableFlowSchema) update() error {
	_, err := u.client.Update(context.TODO(), u.FlowSchema, metav1.UpdateOptions{FieldManager: fieldManager})
	return err
}

type deletableFlowSchema struct {
	*flowcontrolv1beta2.FlowSchema
	client flowcontrolclient.FlowSchemaInterface
}

func (dbl *deletableFlowSchema) delete(resourceVersion string) error {
	return dbl.client.Delete(context.TODO(), dbl.Name, metav1.DeleteOptions{Preconditions: &metav1.Preconditions{ResourceVersion: &resourceVersion}})
}
