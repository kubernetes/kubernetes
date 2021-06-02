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

	flowcontrolv1beta1 "k8s.io/api/flowcontrol/v1beta1"
	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	flowcontrolclient "k8s.io/client-go/kubernetes/typed/flowcontrol/v1beta1"
	flowcontrolapisv1beta1 "k8s.io/kubernetes/pkg/apis/flowcontrol/v1beta1"
)

var (
	errObjectNotFlowSchema = errors.New("object is not a FlowSchema type")
)

// FlowSchemaEnsurer ensures the specified bootstrap configuration objects
type FlowSchemaEnsurer interface {
	Ensure([]*flowcontrolv1beta1.FlowSchema) error
}

// FlowSchemaRemover removes the specified bootstrap configuration objects
type FlowSchemaRemover interface {
	Remove([]string) error
}

// NewSuggestedFlowSchemaEnsurer returns a FlowSchemaEnsurer instance that
// can be used to ensure a set of suggested FlowSchema configuration objects.
func NewSuggestedFlowSchemaEnsurer(client flowcontrolclient.FlowSchemaInterface) FlowSchemaEnsurer {
	wrapper := &flowSchemaWrapper{
		client: client,
	}
	return &fsEnsurer{
		strategy: newSuggestedEnsureStrategy(wrapper),
		wrapper:  wrapper,
	}
}

// NewMandatoryFlowSchemaEnsurer returns a FlowSchemaEnsurer instance that
// can be used to ensure a set of mandatory FlowSchema configuration objects.
func NewMandatoryFlowSchemaEnsurer(client flowcontrolclient.FlowSchemaInterface) FlowSchemaEnsurer {
	wrapper := &flowSchemaWrapper{
		client: client,
	}
	return &fsEnsurer{
		strategy: newMandatoryEnsureStrategy(wrapper),
		wrapper:  wrapper,
	}
}

// NewFlowSchemaRemover returns a FlowSchemaRemover instance that
// can be used to remove a set of FlowSchema configuration objects.
func NewFlowSchemaRemover(client flowcontrolclient.FlowSchemaInterface) FlowSchemaRemover {
	return &fsEnsurer{
		wrapper: &flowSchemaWrapper{
			client: client,
		},
	}
}

// GetFlowSchemaRemoveCandidate returns a list of FlowSchema object
// names that are candidates for deletion from the cluster.
// bootstrap: a set of hard coded FlowSchema configuration objects
// kube-apiserver maintains in-memory.
func GetFlowSchemaRemoveCandidate(client flowcontrolclient.FlowSchemaInterface, bootstrap []*flowcontrolv1beta1.FlowSchema) ([]string, error) {
	// TODO(101667): Use a lister here to avoid periodic LIST calls
	fsList, err := client.List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to list FlowSchema - %w", err)
	}

	bootstrapNames := sets.String{}
	for i := range bootstrap {
		bootstrapNames.Insert(bootstrap[i].GetName())
	}

	currentObjects := make([]metav1.Object, len(fsList.Items))
	for i := range fsList.Items {
		currentObjects[i] = &fsList.Items[i]
	}

	return getRemoveCandidate(bootstrapNames, currentObjects), nil
}

type fsEnsurer struct {
	strategy ensureStrategy
	wrapper  configurationWrapper
}

func (e *fsEnsurer) Ensure(flowSchemas []*flowcontrolv1beta1.FlowSchema) error {
	for _, flowSchema := range flowSchemas {
		if err := ensureConfiguration(e.wrapper, e.strategy, flowSchema); err != nil {
			return err
		}
	}

	return nil
}

func (e *fsEnsurer) Remove(flowSchemas []string) error {
	for _, flowSchema := range flowSchemas {
		if err := removeConfiguration(e.wrapper, flowSchema); err != nil {
			return err
		}
	}

	return nil
}

// flowSchemaWrapper abstracts all FlowSchema specific logic, with this
// we can manage all boiler plate code in one place.
type flowSchemaWrapper struct {
	client flowcontrolclient.FlowSchemaInterface
}

func (fs *flowSchemaWrapper) TypeName() string {
	return "FlowSchema"
}

func (fs *flowSchemaWrapper) Create(object runtime.Object) (runtime.Object, error) {
	fsObject, ok := object.(*flowcontrolv1beta1.FlowSchema)
	if !ok {
		return nil, errObjectNotFlowSchema
	}

	return fs.client.Create(context.TODO(), fsObject, metav1.CreateOptions{FieldManager: fieldManager})
}

func (fs *flowSchemaWrapper) Update(object runtime.Object) (runtime.Object, error) {
	fsObject, ok := object.(*flowcontrolv1beta1.FlowSchema)
	if !ok {
		return nil, errObjectNotFlowSchema
	}

	return fs.client.Update(context.TODO(), fsObject, metav1.UpdateOptions{FieldManager: fieldManager})
}

func (fs *flowSchemaWrapper) Get(name string) (configurationObject, error) {
	return fs.client.Get(context.TODO(), name, metav1.GetOptions{})
}

func (fs *flowSchemaWrapper) Delete(name string) error {
	return fs.client.Delete(context.TODO(), name, metav1.DeleteOptions{})
}

func (fs *flowSchemaWrapper) CopySpec(bootstrap, current runtime.Object) error {
	bootstrapFS, ok := bootstrap.(*flowcontrolv1beta1.FlowSchema)
	if !ok {
		return errObjectNotFlowSchema
	}
	currentFS, ok := current.(*flowcontrolv1beta1.FlowSchema)
	if !ok {
		return errObjectNotFlowSchema
	}

	specCopy := bootstrapFS.Spec.DeepCopy()
	currentFS.Spec = *specCopy
	return nil
}

func (fs *flowSchemaWrapper) HasSpecChanged(bootstrap, current runtime.Object) (bool, error) {
	bootstrapFS, ok := bootstrap.(*flowcontrolv1beta1.FlowSchema)
	if !ok {
		return false, errObjectNotFlowSchema
	}
	currentFS, ok := current.(*flowcontrolv1beta1.FlowSchema)
	if !ok {
		return false, errObjectNotFlowSchema
	}

	return flowSchemaSpecChanged(bootstrapFS, currentFS), nil
}

func flowSchemaSpecChanged(expected, actual *flowcontrolv1beta1.FlowSchema) bool {
	copiedExpectedFlowSchema := expected.DeepCopy()
	flowcontrolapisv1beta1.SetObjectDefaults_FlowSchema(copiedExpectedFlowSchema)
	return !equality.Semantic.DeepEqual(copiedExpectedFlowSchema.Spec, actual.Spec)
}
