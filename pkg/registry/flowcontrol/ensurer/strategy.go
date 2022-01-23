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
	"fmt"
	"strconv"

	"github.com/google/go-cmp/cmp"
	flowcontrolv1beta2 "k8s.io/api/flowcontrol/v1beta2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
)

const (
	fieldManager = "api-priority-and-fairness-config-producer-v1"
)

// EnsureStrategy provides a maintenance strategy for APF configuration objects.
// We have two types of strategy, corresponding to the two types of config objetcs:
// - mandatory: the mandatory objects are about ensuring that the P&F
//   system itself won't crash; we have to be sure there's 'catch-all' place for
//   everything to go. Any changes made by the cluster operators to these
//   objects will be stomped by the apiserver.
//
// - suggested: additional config objects for default behavior.
//   the cluster operators have an option to modify these objects.
type EnsureStrategy interface {
	// Name of the strategy, for now we have two: 'mandatory' and 'suggested'.
	// This comes handy in logging.
	Name() string

	// ShouldUpdate accepts a pair of the current and the bootstrap configuration and determines
	// whether an update is necessary.
	// current is the existing in-cluster configuration object.
	// bootstrap is the configuration the kube-apiserver maintains in-memory.
	//
	// revised: the new object represents the new configuration to be stored in-cluster.
	// ok: true if auto update is required, otherwise false
	// err: err is set when the function runs into an error and can not
	//      determine if auto update is needed.
	ShouldUpdate(wantAndHave) (revised updatable, ok bool, err error)
}

// BootstrapObjects is a generic interface to a list of bootstrap objects bound up with the relevant operations on them.
// The binding makes it unnecessary to have any type casts.
// A bootstrap object is a mandatory or suggested config object,
// with the spec that the code is built to provide.
type BootstrapObjects interface {
	typeName() string                         // the Kind of the objects
	len() int                                 // number of objects
	get(int) bootstrapObject                  // extract one object, origin 0
	getExistingObjects() ([]deletable, error) // returns all the APF config objects that exist at the moment
}

// deletable is an existing config object and it supports the delete operation
type deletable interface {
	configurationObject
	delete(resourceVersion string) error // delete the object if and only if it has the given resourceVersion
}

// bootstrapObject is a single bootstrap object.
// Its spec is what the code provides.
type bootstrapObject interface {
	typeName() string                 // the Kind of the object
	getName() string                  // the object's name
	create() error                    // request the server to create the object
	getCurrent() (wantAndHave, error) // pair up with the object as it currently exists
}

// wantAndHave is a pair of versions of an APF config object.
// The "want" has the spec that the code provides.
// The "have" is what came from the server.
type wantAndHave interface {
	getWant() configurationObject
	getHave() configurationObject

	specsDiffer() bool

	// copyHave returns a copy of the "have" version,
	// optionally with spec replaced by the spec from "want".
	copyHave(specFromWant bool) updatable
}

// updatable is an APF config object that can be written back to the apiserver
type updatable interface {
	configurationObject
	update() error
}

// A convenient wrapper interface that is used by the ensure logic.
type configurationObject interface {
	metav1.Object
	runtime.Object
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

func (s *strategy) ShouldUpdate(wah wantAndHave) (updatable, bool, error) {
	current := wah.getHave()

	if current == nil {
		return nil, false, nil
	}

	autoUpdateSpec := s.alwaysAutoUpdateSpec
	if !autoUpdateSpec {
		autoUpdateSpec = shouldUpdateSpec(current)
	}
	updateAnnotation := shouldUpdateAnnotation(current, autoUpdateSpec)

	specChanged := autoUpdateSpec && wah.specsDiffer()

	if !(updateAnnotation || specChanged) {
		// the annotation key is up to date and the spec has not changed, no update is necessary
		return nil, false, nil
	}

	revised := wah.copyHave(specChanged)
	if updateAnnotation {
		setAutoUpdateAnnotation(revised, autoUpdateSpec)
	}

	return revised, true, nil
}

// shouldUpdateSpec inspects the auto-update annotation key and generation field to determine
// whether the config object should be auto-updated.
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
func EnsureConfigurations(boots BootstrapObjects, strategy EnsureStrategy) error {
	len := boots.len()
	for i := 0; i < len; i++ {
		bo := boots.get(i)
		err := EnsureConfiguration(bo, strategy)
		if err != nil {
			return err
		}
	}
	return nil
}

// EnsureConfiguration applies the given maintenance strategy to the given object.
func EnsureConfiguration(bootstrap bootstrapObject, strategy EnsureStrategy) error {
	name := bootstrap.getName()
	configurationType := strategy.Name()

	var wah wantAndHave
	var err error
	for {
		wah, err = bootstrap.getCurrent()
		if err == nil {
			break
		}
		if !apierrors.IsNotFound(err) {
			return fmt.Errorf("failed to retrieve %s type=%s name=%q error=%w", bootstrap.typeName(), configurationType, name, err)
		}

		// we always re-create a missing configuration object
		if err = bootstrap.create(); err == nil {
			klog.V(2).InfoS(fmt.Sprintf("Successfully created %s", bootstrap.typeName()), "type", configurationType, "name", name)
			return nil
		}

		if !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("cannot create %s type=%s name=%q error=%w", bootstrap.typeName(), configurationType, name, err)
		}
		klog.V(5).InfoS(fmt.Sprintf("Something created the %s concurrently", bootstrap.typeName()), "type", configurationType, "name", name)
	}

	klog.V(5).InfoS(fmt.Sprintf("The %s already exists, checking whether it is up to date", bootstrap.typeName()), "type", configurationType, "name", name)
	newObject, update, err := strategy.ShouldUpdate(wah)
	if err != nil {
		return fmt.Errorf("failed to determine whether auto-update is required for %s type=%s name=%q error=%w", bootstrap.typeName(), configurationType, name, err)
	}
	if !update {
		if klogV := klog.V(5); klogV.Enabled() {
			klogV.InfoS("No update required", "wrapper", bootstrap.typeName(), "type", configurationType, "name", name,
				"diff", cmp.Diff(wah.getHave(), wah.getWant()))
		}
		return nil
	}

	if err = newObject.update(); err == nil {
		klog.V(2).Infof("Updated the %s type=%s name=%q diff: %s", bootstrap.typeName(), configurationType, name, cmp.Diff(wah.getHave(), wah.getWant()))
		return nil
	}

	if apierrors.IsConflict(err) {
		klog.V(2).InfoS(fmt.Sprintf("Something updated the %s concurrently, I will check its spec later", bootstrap.typeName()), "type", configurationType, "name", name)
		return nil
	}

	return fmt.Errorf("failed to update the %s, will retry later type=%s name=%q error=%w", bootstrap.typeName(), configurationType, name, err)
}

// RemoveUnwantedObjects attempts to delete the configuration objects
// that exist, are annotated `apf.kubernetes.io/autoupdate-spec=true`, and do not
// have a name in the given set.  A refusal due to concurrent update is logged
// and not considered an error; the object will be reconsidered later.
func RemoveUnwantedObjects(boots BootstrapObjects) error {
	current, err := boots.getExistingObjects()
	if err != nil {
		klog.ErrorS(err, "Failed to list existing APF configuration objects", "type", boots.typeName())
	}
	wantedNames := namesOfBootstrapObjects(boots)
	for _, object := range current {
		name := object.GetName()
		if wantedNames.Has(name) {
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
		err = object.delete(expectedResourceVersion)
		if err == nil {
			klog.V(2).InfoS(fmt.Sprintf("Successfully deleted the unwanted %s", boots.typeName()), "name", name)
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

func namesOfBootstrapObjects(bos BootstrapObjects) sets.String {
	names := sets.NewString()
	len := bos.len()
	for i := 0; i < len; i++ {
		bo := bos.get(i)
		names.Insert(bo.getName())
	}
	return names
}
