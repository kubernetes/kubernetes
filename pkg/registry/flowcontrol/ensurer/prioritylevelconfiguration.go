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
	errObjectNotPriorityLevel = errors.New("object is not a PriorityLevelConfiguration type")
)

// NewPriorityLevelConfigurationWrapper makes a ConfigurationWrapper for PriorityLevelConfiguration objects
func NewPriorityLevelConfigurationWrapper(client flowcontrolclient.PriorityLevelConfigurationInterface, lister flowcontrollisters.PriorityLevelConfigurationLister) ConfigurationWrapper {
	return &priorityLevelConfigurationWrapper{
		client: client,
		lister: lister,
	}
}

// ObjectifyPriorityLevelConfigurations copies the given list to a generic form
func ObjectifyPriorityLevelConfigurations(objs []*flowcontrolv1beta2.PriorityLevelConfiguration) configurationObjectSlice {
	slice := make(configurationObjectSlice, 0, len(objs))
	for _, obj := range objs {
		slice = append(slice, obj)
	}
	return slice
}

// priorityLevelConfigurationWrapper abstracts all PriorityLevelConfiguration specific logic,
// with this we can manage all boiler plate code in one place.
type priorityLevelConfigurationWrapper struct {
	client flowcontrolclient.PriorityLevelConfigurationInterface
	lister flowcontrollisters.PriorityLevelConfigurationLister
}

var _ ConfigurationWrapper = &priorityLevelConfigurationWrapper{}

func (fs *priorityLevelConfigurationWrapper) TypeName() string {
	return "PriorityLevelConfiguration"
}

func (fs *priorityLevelConfigurationWrapper) Create(object runtime.Object) (runtime.Object, error) {
	plObject, ok := object.(*flowcontrolv1beta2.PriorityLevelConfiguration)
	if !ok {
		return nil, errObjectNotPriorityLevel
	}

	return fs.client.Create(context.TODO(), plObject, metav1.CreateOptions{FieldManager: fieldManager})
}

func (fs *priorityLevelConfigurationWrapper) Update(object runtime.Object) (runtime.Object, error) {
	fsObject, ok := object.(*flowcontrolv1beta2.PriorityLevelConfiguration)
	if !ok {
		return nil, errObjectNotPriorityLevel
	}

	return fs.client.Update(context.TODO(), fsObject, metav1.UpdateOptions{FieldManager: fieldManager})
}

func (fs *priorityLevelConfigurationWrapper) Get(name string) (configurationObject, error) {
	return fs.lister.Get(name)
}

func (fs *priorityLevelConfigurationWrapper) Delete(name, resourceVersion string) error {
	return fs.client.Delete(context.TODO(), name, metav1.DeleteOptions{Preconditions: &metav1.Preconditions{ResourceVersion: &resourceVersion}})
}

func (fs *priorityLevelConfigurationWrapper) CopySpec(bootstrap, current runtime.Object) error {
	bootstrapFS, ok := bootstrap.(*flowcontrolv1beta2.PriorityLevelConfiguration)
	if !ok {
		return errObjectNotPriorityLevel
	}
	currentFS, ok := current.(*flowcontrolv1beta2.PriorityLevelConfiguration)
	if !ok {
		return errObjectNotPriorityLevel
	}

	specCopy := bootstrapFS.Spec.DeepCopy()
	currentFS.Spec = *specCopy
	return nil
}

func (fs *priorityLevelConfigurationWrapper) HasSpecChanged(bootstrap, current runtime.Object) (bool, error) {
	bootstrapFS, ok := bootstrap.(*flowcontrolv1beta2.PriorityLevelConfiguration)
	if !ok {
		return false, errObjectNotPriorityLevel
	}
	currentFS, ok := current.(*flowcontrolv1beta2.PriorityLevelConfiguration)
	if !ok {
		return false, errObjectNotPriorityLevel
	}

	return priorityLevelSpecChanged(bootstrapFS, currentFS), nil
}

func priorityLevelSpecChanged(expected, actual *flowcontrolv1beta2.PriorityLevelConfiguration) bool {
	copiedExpectedPriorityLevel := expected.DeepCopy()
	flowcontrolapisv1beta2.SetObjectDefaults_PriorityLevelConfiguration(copiedExpectedPriorityLevel)
	return !equality.Semantic.DeepEqual(copiedExpectedPriorityLevel.Spec, actual.Spec)
}

func (wr priorityLevelConfigurationWrapper) GetExistingObjects() (configurationObjectSlice, error) {
	objs, err := wr.lister.List(labels.Everything())
	if err != nil {
		return nil, fmt.Errorf("failed to list PriorityLevelConfiguration objects - %w", err)
	}
	return ObjectifyPriorityLevelConfigurations(objs), nil
}
