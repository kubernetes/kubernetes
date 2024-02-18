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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/kubernetes/test/utils/ktesting/kobject"
)

// createAny defines an op where some object gets created from a YAML file.
// The nameset can be specified.
type createAny struct {
	// Must match createAnyOpcode.
	Opcode operationCode
	// Namespace the object should be created in. Must be empty for cluster-scoped objects.
	Namespace string
	// Path to spec file describing the object to create.
	TemplatePath string
}

var _ runnableOp = &createAny{}

func (c *createAny) isValid(allowParameterization bool) error {
	if c.Opcode != createAnyOpcode {
		return fmt.Errorf("invalid opcode %q; expected %q", c.Opcode, createAnyOpcode)
	}
	if c.TemplatePath == "" {
		return fmt.Errorf("TemplatePath must be set")
	}
	// The namespace can only be checked during later because we don't know yet
	// whether the object is namespaced or cluster-scoped.
	return nil
}

func (c *createAny) collectsMetrics() bool {
	return false
}

func (c *createAny) patchParams(w *workload) (realOp, error) {
	return c, c.isValid(false)
}

func (c *createAny) requiredNamespaces() []string {
	if c.Namespace == "" {
		return nil
	}
	return []string{c.Namespace}
}

func (c *createAny) run(tCtx ktesting.TContext) {
	var obj *unstructured.Unstructured
	if err := getSpecFromFile(&c.TemplatePath, &obj); err != nil {
		tCtx.Fatalf("%s: parsing failed: %v", c.TemplatePath, err)
	}

	tCtx = kobject.WithNamespace(tCtx, c.Namespace)

	// Retry, some errors (like CRD just created and type not ready for use yet) are temporary.
	ctx, cancel := context.WithTimeout(tCtx, 20*time.Second)
	defer cancel()
	for {
		_, err := kobject.Create(tCtx, obj, metav1.CreateOptions{})
		if err == nil {
			return
		}
		select {
		case <-ctx.Done():
			tCtx.Fatalf("%s: timed out (%q) while creating %q, last error was: %v", c.TemplatePath, context.Cause(ctx), klog.KObj(obj), err)
		case <-time.After(time.Second):
		}
	}
}
