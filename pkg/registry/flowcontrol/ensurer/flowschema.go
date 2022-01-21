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
	"errors"
	"fmt"

	flowcontrolv1beta2 "k8s.io/api/flowcontrol/v1beta2"
	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	flowcontrolclient "k8s.io/client-go/kubernetes/typed/flowcontrol/v1beta2"
	flowcontrollisters "k8s.io/client-go/listers/flowcontrol/v1beta2"
	flowcontrolapisv1beta2 "k8s.io/kubernetes/pkg/apis/flowcontrol/v1beta2"
)

var (
	errObjectNotFlowSchema = errors.New("object is not a FlowSchema type")
)

// NewFlowSchemaWrapper makes a ConfigurationWrapper for FlowSchema objects
func NewFlowSchemaWrapper(client flowcontrolclient.FlowSchemaInterface, lister flowcontrollisters.FlowSchemaLister) ConfigurationWrapper {
	return &flowSchemaWrapper{
		client: client,
		lister: lister,
	}
}

// ObjectifyFlowSchemas copies the given list to a generic form
func ObjectifyFlowSchemas(objs []*flowcontrolv1beta2.FlowSchema) configurationObjectSlice {
	slice := make(configurationObjectSlice, 0, len(objs))
	for _, obj := range objs {
		slice = append(slice, obj)
	}
	return slice
}

// flowSchemaWrapper abstracts all FlowSchema specific logic, with this
// we can manage all boiler plate code in one place.
type flowSchemaWrapper struct {
	client flowcontrolclient.FlowSchemaInterface
	lister flowcontrollisters.FlowSchemaLister
}

var _ ConfigurationWrapper = &flowSchemaWrapper{}

func (fs *flowSchemaWrapper) TypeName() string {
	return "FlowSchema"
}

func (fs *flowSchemaWrapper) Create(object runtime.Object) (runtime.Object, error) {
	fsObject, ok := object.(*flowcontrolv1beta2.FlowSchema)
	if !ok {
		return nil, errObjectNotFlowSchema
	}

	return fs.client.Create(context.TODO(), fsObject, metav1.CreateOptions{FieldManager: fieldManager})
}

func (fs *flowSchemaWrapper) Update(object runtime.Object) (runtime.Object, error) {
	fsObject, ok := object.(*flowcontrolv1beta2.FlowSchema)
	if !ok {
		return nil, errObjectNotFlowSchema
	}

	return fs.client.Update(context.TODO(), fsObject, metav1.UpdateOptions{FieldManager: fieldManager})
}

func (fs *flowSchemaWrapper) Get(name string) (configurationObject, error) {
	return fs.lister.Get(name)
}

func (fs *flowSchemaWrapper) Delete(name, resourceVersion string) error {
	return fs.client.Delete(context.TODO(), name, metav1.DeleteOptions{Preconditions: &metav1.Preconditions{ResourceVersion: &resourceVersion}})
}

func (fs *flowSchemaWrapper) CopySpec(bootstrap, current runtime.Object) error {
	bootstrapFS, ok := bootstrap.(*flowcontrolv1beta2.FlowSchema)
	if !ok {
		return errObjectNotFlowSchema
	}
	currentFS, ok := current.(*flowcontrolv1beta2.FlowSchema)
	if !ok {
		return errObjectNotFlowSchema
	}

	specCopy := bootstrapFS.Spec.DeepCopy()
	currentFS.Spec = *specCopy
	return nil
}

func (fs *flowSchemaWrapper) HasSpecChanged(bootstrap, current runtime.Object) (bool, error) {
	bootstrapFS, ok := bootstrap.(*flowcontrolv1beta2.FlowSchema)
	if !ok {
		return false, errObjectNotFlowSchema
	}
	currentFS, ok := current.(*flowcontrolv1beta2.FlowSchema)
	if !ok {
		return false, errObjectNotFlowSchema
	}

	return flowSchemaSpecChanged(bootstrapFS, currentFS), nil
}

func flowSchemaSpecChanged(expected, actual *flowcontrolv1beta2.FlowSchema) bool {
	copiedExpectedFlowSchema := expected.DeepCopy()
	flowcontrolapisv1beta2.SetObjectDefaults_FlowSchema(copiedExpectedFlowSchema)
	return !equality.Semantic.DeepEqual(copiedExpectedFlowSchema.Spec, actual.Spec)
}

func (wr flowSchemaWrapper) GetExistingObjects() (configurationObjectSlice, error) {
	objs, err := wr.lister.List(labels.Everything())
	if err != nil {
		return nil, fmt.Errorf("failed to list FlowSchema objects - %w", err)
	}
	return ObjectifyFlowSchemas(objs), nil
}
