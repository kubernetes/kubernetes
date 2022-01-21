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
	"errors"
	"fmt"
	"strconv"

	flowcontrolv1beta2 "k8s.io/api/flowcontrol/v1beta2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"

	"github.com/google/go-cmp/cmp"
)

const (
	fieldManager = "api-priority-and-fairness-config-producer-v1"
)

// EnsureStrategy provides a strategy for ensuring apf bootstrap ConfigurationWrapper.
// We have two types of ConfigurationWrapper objects:
// - mandatory: the mandatory ConfigurationWrapper objects are about ensuring that the P&F
//   system itself won't crash; we have to be sure there's 'catch-all' place for
//   everything to go. Any changes made by the cluster operators to these
//   ConfigurationWrapper objects will be stomped by the apiserver.
//
// - suggested: additional ConfigurationWrapper objects for initial behavior.
//   the cluster operators have an option to edit or delete these ConfigurationWrapper objects.
type EnsureStrategy interface {
	// Name of the strategy, for now we have two: 'mandatory' and 'suggested'.
	// This comes handy in logging.
	Name() string

	// ShouldUpdate accepts the current and the bootstrap configuration and determines
	// whether an update is necessary.
	// current is the existing in-cluster configuration object.
	// bootstrap is the configuration the kube-apiserver maintains in-memory.
	//
	// ok: true if auto update is required, otherwise false
	// object: the new object represents the new configuration to be stored in-cluster.
	// err: err is set when the function runs into an error and can not
	// determine if auto update is needed.
	ShouldUpdate(sCopier specCopier, current, bootstrap configurationObject) (object runtime.Object, ok bool, err error)
}

// this internal interface provides abstraction for dealing with the `Spec`
// of both 'FlowSchema' and 'PriorityLevelConfiguration' objects.
// Since the ensure logic for both types is common, we use a few internal interfaces
// to abstract out the differences of these two types.
type specCopier interface {
	// HasSpecChanged returns true if the spec of both the bootstrap and
	// the current configuration object is same, otherwise false.
	HasSpecChanged(bootstrap, current runtime.Object) (bool, error)

	// CopySpec makes a deep copy the spec of the bootstrap object
	// and copies it to that of the current object.
	// CopySpec assumes that the current object is safe to mutate, so it
	// rests with the caller to make a deep copy of the current.
	CopySpec(bootstrap, current runtime.Object) error
}

// this internal interface provides abstraction for CRUD operation
// related to both 'FlowSchema' and 'PriorityLevelConfiguration' objects.
// Since the ensure logic for both types is common, we use a few internal interfaces
// to abstract out the differences of these two types.
type configurationClient interface {
	Create(object runtime.Object) (runtime.Object, error)
	Update(object runtime.Object) (runtime.Object, error)
	Get(name string) (configurationObject, error)
	Delete(name, resourceVersion string) error
}

// ConfigurationWrapper collects together generic versions of all the operations
// that are needed on configuration objects.
type ConfigurationWrapper interface {
	// TypeName returns the type of the configuration that this interface deals with.
	// We use it to log the type name of the configuration object being ensured.
	// It is either 'PriorityLevelConfiguration' or 'FlowSchema'
	TypeName() string

	configurationClient
	specCopier
	GetExistingObjects() (configurationObjectSlice, error)
}

// A convenient wrapper interface that is used by the ensure logic.
type configurationObject interface {
	metav1.Object
	runtime.Object
}

type configurationObjectSlice []configurationObject

func (cobjs configurationObjectSlice) Names() sets.String {
	names := sets.String{}
	for _, obj := range cobjs {
		names.Insert(obj.GetName())
	}
	return names
}

// NewSuggestedEnsureStrategy returns an EnsureStrategy for suggested config objects
func NewSuggestedEnsureStrategy() EnsureStrategy {
	return &strategy{
		alwaysAutoUpdateSpec: false,
		name:                 "suggested",
	}
}

// NewMandatoryEnsureStrategy returns an EnsureStrategy for mandatory config objects
func NewMandatoryEnsureStrategy() EnsureStrategy {
	return &strategy{
		alwaysAutoUpdateSpec: true,
		name:                 "mandatory",
	}
}

// auto-update strategy for the configuration objects
type strategy struct {
	alwaysAutoUpdateSpec bool
	name                 string
}

func (s *strategy) Name() string {
	return s.name
}

func (s *strategy) ShouldUpdate(sCopier specCopier, current, bootstrap configurationObject) (runtime.Object, bool, error) {
	if current == nil || bootstrap == nil {
		return nil, false, nil
	}

	autoUpdateSpec := s.alwaysAutoUpdateSpec
	if !autoUpdateSpec {
		autoUpdateSpec = shouldUpdateSpec(current)
	}
	updateAnnotation := shouldUpdateAnnotation(current, autoUpdateSpec)

	var specChanged bool
	if autoUpdateSpec {
		changed, err := sCopier.HasSpecChanged(bootstrap, current)
		if err != nil {
			return nil, false, fmt.Errorf("failed to compare spec - %w", err)
		}
		specChanged = changed
	}

	if !(updateAnnotation || specChanged) {
		// the annotation key is up to date and the spec has not changed, no update is necessary
		return nil, false, nil
	}

	// if we are here, either we need to update the annotation key or the spec.
	copy, ok := current.DeepCopyObject().(configurationObject)
	if !ok {
		// we should never be here
		return nil, false, errors.New("incompatible object type")
	}

	if updateAnnotation {
		setAutoUpdateAnnotation(copy, autoUpdateSpec)
	}
	if specChanged {
		sCopier.CopySpec(bootstrap, copy)
	}

	return copy, true, nil
}

// shouldUpdateSpec inspects the auto-update annotation key and generation field to determine
// whether the ConfigurationWrapper object should be auto-updated.
func shouldUpdateSpec(accessor metav1.Object) bool {
	value, _ := accessor.GetAnnotations()[flowcontrolv1beta2.AutoUpdateAnnotationKey]
	if autoUpdate, err := strconv.ParseBool(value); err == nil {
		return autoUpdate
	}

	// We are here because of either a or b:
	// a. the annotation key is missing.
	// b. the annotation key is present but the value does not represent a boolean.
	// In either case, if the operator hasn't changed the spec, we can safely auto update.
	// Please note that we can't protect the changes made by the operator in the following scenario:
	// - The operator deletes and recreates the same object with a variant spec (generation resets to 1).
	if accessor.GetGeneration() == 1 {
		return true
	}
	return false
}

// shouldUpdateAnnotation determines whether the current value of the auto-update annotation
// key matches the desired value.
func shouldUpdateAnnotation(accessor metav1.Object, desired bool) bool {
	if value, ok := accessor.GetAnnotations()[flowcontrolv1beta2.AutoUpdateAnnotationKey]; ok {
		if current, err := strconv.ParseBool(value); err == nil && current == desired {
			return false
		}
	}

	return true
}

// setAutoUpdateAnnotation sets the auto-update annotation key to the specified value.
func setAutoUpdateAnnotation(accessor metav1.Object, autoUpdate bool) {
	if accessor.GetAnnotations() == nil {
		accessor.SetAnnotations(map[string]string{})
	}

	accessor.GetAnnotations()[flowcontrolv1beta2.AutoUpdateAnnotationKey] = strconv.FormatBool(autoUpdate)
}

// EnsureConfigurations applies the given maintenance strategy to the given objects.
// At the first error, if any, it stops and returns that error.
func EnsureConfigurations(wrapper ConfigurationWrapper, strategy EnsureStrategy, bootstrap configurationObjectSlice) error {
	for _, obj := range bootstrap {
		err := EnsureConfiguration(wrapper, strategy, obj)
		if err != nil {
			return err
		}
	}
	return nil
}

// EnsureConfiguration applies the given maintenance strategy to the given object.
func EnsureConfiguration(wrapper ConfigurationWrapper, strategy EnsureStrategy, bootstrap configurationObject) error {
	name := bootstrap.GetName()
	configurationType := strategy.Name()

	var current configurationObject
	var err error
	for {
		current, err = wrapper.Get(bootstrap.GetName())
		if err == nil {
			break
		}
		if !apierrors.IsNotFound(err) {
			return fmt.Errorf("failed to retrieve %s type=%s name=%q error=%w", wrapper.TypeName(), configurationType, name, err)
		}

		// we always re-create a missing configuration object
		if _, err = wrapper.Create(bootstrap); err == nil {
			klog.V(2).InfoS(fmt.Sprintf("Successfully created %s", wrapper.TypeName()), "type", configurationType, "name", name)
			return nil
		}

		if !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("cannot create %s type=%s name=%q error=%w", wrapper.TypeName(), configurationType, name, err)
		}
		klog.V(5).InfoS(fmt.Sprintf("Something created the %s concurrently", wrapper.TypeName()), "type", configurationType, "name", name)
	}

	klog.V(5).InfoS(fmt.Sprintf("The %s already exists, checking whether it is up to date", wrapper.TypeName()), "type", configurationType, "name", name)
	newObject, update, err := strategy.ShouldUpdate(wrapper, current, bootstrap)
	if err != nil {
		return fmt.Errorf("failed to determine whether auto-update is required for %s type=%s name=%q error=%w", wrapper.TypeName(), configurationType, name, err)
	}
	if !update {
		if klogV := klog.V(5); klogV.Enabled() {
			klogV.InfoS("No update required", "wrapper", wrapper.TypeName(), "type", configurationType, "name", name,
				"diff", cmp.Diff(current, bootstrap))
		}
		return nil
	}

	if _, err = wrapper.Update(newObject); err == nil {
		klog.V(2).Infof("Updated the %s type=%s name=%q diff: %s", wrapper.TypeName(), configurationType, name, cmp.Diff(current, newObject))
		return nil
	}

	if apierrors.IsConflict(err) {
		klog.V(2).InfoS(fmt.Sprintf("Something updated the %s concurrently, I will check its spec later", wrapper.TypeName()), "type", configurationType, "name", name)
		return nil
	}

	return fmt.Errorf("failed to update the %s, will retry later type=%s name=%q error=%w", wrapper.TypeName(), configurationType, name, err)
}

// RemoveUnwantedObjects attempts to delete the configuration objects
// that exist, are annotated `apf.kubernetes.io/autoupdate-spec=true`, and do not
// have a name in the given set.  A refusal due to concurrent update is logged
// and not considered an error; the object will be reconsidered later.
func RemoveUnwantedObjects(wrapper ConfigurationWrapper, bootstrap sets.String) error {
	current, err := wrapper.GetExistingObjects()
	if err != nil {
		klog.ErrorS(err, "Failed to list existing APF configuration objects", "type", wrapper.TypeName())
	}
	for _, object := range current {
		name := object.GetName()
		if bootstrap.Has(name) {
			continue
		}
		var value string
		var ok, autoUpdate bool
		var err error
		if value, ok = object.GetAnnotations()[flowcontrolv1beta2.AutoUpdateAnnotationKey]; !ok {
			// the configuration object does not have the annotation key,
			// it's probably a user defined configuration object,
			// so we can skip it.
			klog.V(5).InfoS("Skipping deletion of APF object with no "+flowcontrolv1beta2.AutoUpdateAnnotationKey+" annotation", "name", name)
			continue
		}
		autoUpdate, err = strconv.ParseBool(value)
		if err != nil {
			// Log this because it is not an expected situation.
			klog.V(4).InfoS("Skipping deletion of APF object with malformed "+flowcontrolv1beta2.AutoUpdateAnnotationKey+" annotation", "name", name, "annotationValue", value, "parseError", err)
			continue
		}
		if !autoUpdate {
			klog.V(5).InfoS("Skipping deletion of APF object with "+flowcontrolv1beta2.AutoUpdateAnnotationKey+"=false annotation", "name", name)
			continue
		}
		expectedResourceVersion := object.GetResourceVersion()
		err = wrapper.Delete(name, expectedResourceVersion)
		if err == nil {
			klog.V(2).InfoS(fmt.Sprintf("Successfully deleted the unwanted %s", wrapper.TypeName()), "name", name)
			continue
		}
		if apierrors.IsConflict(err) {
			klog.V(4).InfoS("Skipped deletion of potentially unwanted APF object due to concurrent update", "name", name, "expectedResourceVersion", expectedResourceVersion)
		} else if apierrors.IsNotFound(err) {
			klog.V(5).InfoS("Unwanted APF object was concurrently deleted", "name", name)
		} else {
			return fmt.Errorf("failed to delete unwatned APF object %q - %w", name, err)
		}
	}
	return nil
}
