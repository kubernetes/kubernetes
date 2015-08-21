/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package lock

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/expapi/validation"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/fielderrors"
)

// lockStrategy implements verification logic for Locks.
type lockStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Will not generate a name at all.
type noNameGenerator struct{}

func (noNameGenerator) GenerateName(base string) string {
	return base
}

var NoNameGenerator api.NameGenerator = noNameGenerator{}

// Strategy is the default logic that applies when creating and updating Lock objects.
var Strategy = lockStrategy{api.Scheme, NoNameGenerator}

// NamespaceScoped returns true because all Locks need to be within a namespace.
func (lockStrategy) NamespaceScoped() bool {
	return true
}

// AllowCreateOnUpdate returns false because a Lock must exist to be updated.
func (lockStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (lockStrategy) PrepareForCreate(obj runtime.Object) {
	lock := obj.(*expapi.Lock)
	lock.Status.AcquiredTime = util.NewTime(time.Now())
	lock.Status.LastRenewalTime = lock.Status.AcquiredTime
}

func (lockStrategy) PrepareForUpdate(obj, old runtime.Object) {
	lock := obj.(*expapi.Lock)
	oldLock := old.(*expapi.Lock)
	lock.Status.LastRenewalTime = util.NewTime(time.Now())
	lock.Status.AcquiredTime = oldLock.Status.AcquiredTime
}

// Validate validates a new lock.
func (lockStrategy) Validate(ctx api.Context, obj runtime.Object) fielderrors.ValidationErrorList {
	lock := obj.(*expapi.Lock)
	return validation.ValidateLock(lock, nil)
}

// ValidateUpdate validates an update to a lock.
func (lockStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) fielderrors.ValidationErrorList {
	lock := obj.(*expapi.Lock)
	oldLock := old.(*expapi.Lock)
	errs := validation.ValidateLock(lock, oldLock)
	return errs
}

func (lockStrategy) AllowUnconditionalUpdate() bool {
	return false
}

// LockToSelectableFields returns a label set that represents the object.
func LockToSelectableFields(lock *expapi.Lock) fields.Set {
	return fields.Set{
		"metadata.name":       lock.Name,
		"spec.heldBy":         lock.Spec.HeldBy,
		"spec.leaseSeconds":   string(lock.Spec.LeaseSeconds),
		"status.acquiredTime": lock.Status.AcquiredTime.String(),
		"status.renewTime":    lock.Status.LastRenewalTime.String(),
	}
}

// MatchLock is the filter used by the generic etcd backend to route
// watch events from etcd to clients of the apiserver only interested in specific
// labels/fields.
func MatchLock(label labels.Selector, field fields.Selector) generic.Matcher {
	return &generic.SelectionPredicate{
		Label: label,
		Field: field,
		GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
			rc, ok := obj.(*expapi.Lock)
			if !ok {
				return nil, nil, fmt.Errorf("Given object is not a lock.")
			}
			return labels.Set(rc.ObjectMeta.Labels), LockToSelectableFields(rc), nil
		},
	}
}
