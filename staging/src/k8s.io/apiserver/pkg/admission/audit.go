/*
Copyright 2018 The Kubernetes Authors.

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

package admission

import (
	"fmt"

	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit"
)

// auditHandler logs annotations set by other admission handlers
type auditHandler struct {
	Interface
	ae *auditinternal.Event
}

var _ Interface = &auditHandler{}
var _ MutationInterface = &auditHandler{}
var _ ValidationInterface = &auditHandler{}

// WithAudit is a decorator for a admission phase. It saves annotations
// of attribute into the audit event. Attributes passed to the Admit and
// Validate function must be instance of privateAnnotationsGetter or
// AnnotationsGetter, otherwise an error is returned.
func WithAudit(i Interface, ae *auditinternal.Event) Interface {
	if i == nil {
		return i
	}
	return &auditHandler{i, ae}
}

func (handler auditHandler) Admit(a Attributes, o ObjectInterfaces) error {
	if !handler.Interface.Handles(a.GetOperation()) {
		return nil
	}
	if err := ensureAnnotationGetter(a); err != nil {
		return err
	}
	var err error
	if mutator, ok := handler.Interface.(MutationInterface); ok {
		err = mutator.Admit(a, o)
		handler.logAnnotations(a)
	}
	return err
}

func (handler auditHandler) Validate(a Attributes, o ObjectInterfaces) error {
	if !handler.Interface.Handles(a.GetOperation()) {
		return nil
	}
	if err := ensureAnnotationGetter(a); err != nil {
		return err
	}
	var err error
	if validator, ok := handler.Interface.(ValidationInterface); ok {
		err = validator.Validate(a, o)
		handler.logAnnotations(a)
	}
	return err
}

func ensureAnnotationGetter(a Attributes) error {
	_, okPrivate := a.(privateAnnotationsGetter)
	_, okPublic := a.(AnnotationsGetter)
	if okPrivate || okPublic {
		return nil
	}
	return fmt.Errorf("attributes must be an instance of privateAnnotationsGetter or AnnotationsGetter")
}

func (handler auditHandler) logAnnotations(a Attributes) {
	switch a := a.(type) {
	case privateAnnotationsGetter:
		audit.LogAnnotations(handler.ae, a.getAnnotations())
	case AnnotationsGetter:
		audit.LogAnnotations(handler.ae, a.GetAnnotations())
	default:
		// this will never happen, because we have already checked it in ensureAnnotationGetter
	}
}
