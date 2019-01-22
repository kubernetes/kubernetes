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
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeHandler implements Interface
type fakeHandler struct {
	// return value of Admit()
	admit error
	// annotations add to attributesRecord during Admit() phase
	admitAnnotations map[string]string
	// return value of Validate()
	validate error
	// annotations add to attributesRecord during Validate() phase
	validateAnnotations map[string]string
	// return value of Handles()
	handles bool
}

var _ Interface = &fakeHandler{}
var _ MutationInterface = &fakeHandler{}
var _ ValidationInterface = &fakeHandler{}

func (h fakeHandler) Admit(a Attributes) error {
	for k, v := range h.admitAnnotations {
		a.AddAnnotation(k, v)
	}
	return h.admit
}

func (h fakeHandler) Validate(a Attributes) error {
	for k, v := range h.validateAnnotations {
		a.AddAnnotation(k, v)
	}
	return h.validate
}

func (h fakeHandler) Handles(o Operation) bool {
	return h.handles
}

func attributes() Attributes {
	return NewAttributesRecord(nil, nil, schema.GroupVersionKind{}, "", "", schema.GroupVersionResource{}, "", "", false, nil)
}

func TestWithAudit(t *testing.T) {
	var testCases = map[string]struct {
		admit               error
		admitAnnotations    map[string]string
		validate            error
		validateAnnotations map[string]string
		handles             bool
	}{
		"not handle": {
			nil,
			nil,
			nil,
			nil,
			false,
		},
		"allow": {
			nil,
			nil,
			nil,
			nil,
			true,
		},
		"allow with annotations": {
			nil,
			map[string]string{
				"plugin.example.com/foo": "bar",
			},
			nil,
			nil,
			true,
		},
		"allow with annotations overwrite": {
			nil,
			map[string]string{
				"plugin.example.com/foo": "bar",
			},
			nil,
			map[string]string{
				"plugin.example.com/foo": "bar",
			},
			true,
		},
		"forbidden error": {
			NewForbidden(attributes(), fmt.Errorf("quota exceeded")),
			nil,
			NewForbidden(attributes(), fmt.Errorf("quota exceeded")),
			nil,
			true,
		},
		"forbidden error with annotations": {
			NewForbidden(attributes(), fmt.Errorf("quota exceeded")),
			nil,
			NewForbidden(attributes(), fmt.Errorf("quota exceeded")),
			map[string]string{
				"plugin.example.com/foo": "bar",
			},
			true,
		},
		"forbidden error with annotations overwrite": {
			NewForbidden(attributes(), fmt.Errorf("quota exceeded")),
			map[string]string{
				"plugin.example.com/foo": "bar",
			},
			NewForbidden(attributes(), fmt.Errorf("quota exceeded")),
			map[string]string{
				"plugin.example.com/foo": "bar",
			},
			true,
		},
	}
	for tcName, tc := range testCases {
		var handler Interface = fakeHandler{tc.admit, tc.admitAnnotations, tc.validate, tc.validateAnnotations, tc.handles}
		ae := &auditinternal.Event{Level: auditinternal.LevelMetadata}
		auditHandler := WithAudit(handler, ae)
		a := attributes()

		assert.Equal(t, handler.Handles(Create), auditHandler.Handles(Create), tcName+": WithAudit decorator should not effect the return value")

		mutator, ok := handler.(MutationInterface)
		require.True(t, ok)
		auditMutator, ok := auditHandler.(MutationInterface)
		require.True(t, ok)
		assert.Equal(t, mutator.Admit(a), auditMutator.Admit(a), tcName+": WithAudit decorator should not effect the return value")

		validator, ok := handler.(ValidationInterface)
		require.True(t, ok)
		auditValidator, ok := auditHandler.(ValidationInterface)
		require.True(t, ok)
		assert.Equal(t, validator.Validate(a), auditValidator.Validate(a), tcName+": WithAudit decorator should not effect the return value")

		annotations := make(map[string]string, len(tc.admitAnnotations)+len(tc.validateAnnotations))
		for k, v := range tc.admitAnnotations {
			annotations[k] = v
		}
		for k, v := range tc.validateAnnotations {
			annotations[k] = v
		}
		if len(annotations) == 0 {
			assert.Nil(t, ae.Annotations, tcName+": unexptected annotations set in audit event")
		} else {
			assert.Equal(t, annotations, ae.Annotations, tcName+": unexptected annotations set in audit event")
		}
	}
}
