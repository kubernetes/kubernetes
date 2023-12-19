/*
Copyright 2019 The Kubernetes Authors.

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

package benchmark

import (
	"context"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// createOp defines an op where some object gets created from a template.
// Everything specific for that object (create call, op code, names) gets
// provided through a type.
type createOp[T interface{}, P createOpType[T]] struct {
	// Must match createOpType.Opcode().
	Opcode operationCode
	// Namespace the object should be created in. Must be empty for cluster-scoped objects.
	Namespace string
	// Path to spec file describing the object to create.
	TemplatePath string
}

func (cro *createOp[T, P]) isValid(allowParameterization bool) error {
	var p P
	if cro.Opcode != p.Opcode() {
		return fmt.Errorf("invalid opcode %q; expected %q", cro.Opcode, p.Opcode())
	}
	if p.Namespaced() && cro.Namespace == "" {
		return fmt.Errorf("Namespace must be set")
	}
	if !p.Namespaced() && cro.Namespace != "" {
		return fmt.Errorf("Namespace must not be set")
	}
	if cro.TemplatePath == "" {
		return fmt.Errorf("TemplatePath must be set")
	}
	return nil
}

func (cro *createOp[T, P]) collectsMetrics() bool {
	return false
}

func (cro *createOp[T, P]) patchParams(w *workload) (realOp, error) {
	return cro, cro.isValid(false)
}

func (cro *createOp[T, P]) requiredNamespaces() []string {
	if cro.Namespace == "" {
		return nil
	}
	return []string{cro.Namespace}
}

func (cro *createOp[T, P]) run(tCtx ktesting.TContext) {
	var obj *T
	var p P
	if err := getSpecFromFile(&cro.TemplatePath, &obj); err != nil {
		tCtx.Fatalf("parsing %s %q: %v", p.Name(), cro.TemplatePath, err)
	}
	if _, err := p.CreateCall(tCtx.Client(), cro.Namespace)(tCtx, obj, metav1.CreateOptions{}); err != nil {
		tCtx.Fatalf("create %s: %v", p.Name(), err)
	}
}

// createOpType provides type-specific values for the generic createOp.
type createOpType[T interface{}] interface {
	Opcode() operationCode
	Name() string
	Namespaced() bool
	CreateCall(client clientset.Interface, namespace string) func(context.Context, *T, metav1.CreateOptions) (*T, error)
}
