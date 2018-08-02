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

package validation

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/node"
)

func TestValidateValidRuntimeClass(t *testing.T) {
	tests := map[string]*node.RuntimeClass{
		"empty": &node.RuntimeClass{
			ObjectMeta: metav1.ObjectMeta{Name: "native"},
		},
		"non-empty": &node.RuntimeClass{
			ObjectMeta: metav1.ObjectMeta{Name: "gvisor"},
			Spec:       node.RuntimeClassSpec{RuntimeHandler: "gvisor"},
		},
	}

	for name, rc := range tests {
		t.Run(name, func(t *testing.T) {
			assert.Empty(t, ValidateRuntimeClass(rc))
		})
	}
}

func TestValidateInvalidRuntimeClass(t *testing.T) {
	tests := map[string]*node.RuntimeClass{
		"namespace": &node.RuntimeClass{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "native",
				Namespace: "default",
			},
		},
		"invalid name": &node.RuntimeClass{
			ObjectMeta: metav1.ObjectMeta{Name: "sandbox/gvisor"},
		},
		"long name": &node.RuntimeClass{
			ObjectMeta: metav1.ObjectMeta{Name: strings.Repeat("a", 256)},
		},
		"invalid runtimeHandler": &node.RuntimeClass{
			ObjectMeta: metav1.ObjectMeta{Name: "gvisor"},
			Spec:       node.RuntimeClassSpec{RuntimeHandler: "sandbox/gvisor"},
		},
		"long runtimeHandler": &node.RuntimeClass{
			ObjectMeta: metav1.ObjectMeta{Name: "gvisor"},
			Spec:       node.RuntimeClassSpec{RuntimeHandler: strings.Repeat("a", 256)},
		},
	}

	for name, rc := range tests {
		t.Run(name, func(t *testing.T) {
			assert.Len(t, ValidateRuntimeClass(rc), 1)
		})
	}
}

func TestValidateRuntimeClassUpdate(t *testing.T) {
	original := node.RuntimeClass{
		ObjectMeta: metav1.ObjectMeta{Name: "gvisor"},
		Spec:       node.RuntimeClassSpec{RuntimeHandler: "gvisor"},
	}
	assert.Empty(t, ValidateRuntimeClass(&original))

	valid := original
	valid.ObjectMeta.ResourceVersion = "2"
	valid.ObjectMeta.Labels = map[string]string{"sandboxed": "true"}
	assert.Empty(t, ValidateRuntimeClassUpdate(&valid, &original), "label changes should be allowed")

	invalid := original
	invalid.Spec.RuntimeHandler = "runc"
	assert.Len(t, ValidateRuntimeClassUpdate(&invalid, &original), 2, "expected resourceVersion & runtimeHandler errors")
}
