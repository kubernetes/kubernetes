/*
Copyright 2024 The Kubernetes Authors.

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
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

// updateAny defines an op where some object gets updated from a YAML file.
// The nameset can be specified.
type updateAny struct {
	// Must match updateAny.
	Opcode operationCode
	// Namespace the object should be updated in. Must be empty for cluster-scoped objects.
	Namespace string
	// Path to spec file describing the object to update.
	// This will be processed with text/template.
	// .Index will be in the range [0, Count-1] when updating
	// more than one object. .Count is the total number of objects.
	TemplatePath string
	// Count determines how many objects get updated. Defaults to 1 if unset.
	Count *int
	// Template parameter for Count.
	CountParam string
	// Number of objects to be updated per second.
	// If set to 0, all objects are updated at once.
	// Optional
	UpdatePerSecond int
	// Internal field of the struct used for caching the mapping.
	cachedMapping *meta.RESTMapping
	// List of subresources to update.
	// If empty, update operation is performed on the actual resource.
	// Optional
	Subresources []string
}

var _ runnableOp = &updateAny{}

func (c *updateAny) isValid(allowParameterization bool) error {
	if c.TemplatePath == "" {
		return fmt.Errorf("TemplatePath must be set")
	}
	if c.UpdatePerSecond < 0 {
		return fmt.Errorf("invalid UpdatePerSecond=%d; should be non-negative", c.UpdatePerSecond)
	}
	// The namespace can only be checked during later because we don't know yet
	// whether the object is namespaced or cluster-scoped.
	return nil
}

func (c *updateAny) collectsMetrics() bool {
	return false
}

func (c updateAny) patchParams(w *workload) (realOp, error) {
	if c.CountParam != "" {
		count, err := w.Params.get(c.CountParam[1:])
		if err != nil {
			return nil, err
		}
		c.Count = ptr.To(count)
	}
	c.cachedMapping = nil
	return &c, c.isValid(false)
}

func (c *updateAny) requiredNamespaces() []string {
	if c.Namespace == "" {
		return nil
	}
	return []string{c.Namespace}
}

func (c *updateAny) run(tCtx ktesting.TContext) {
	count := 1
	if c.Count != nil {
		count = *c.Count
	}

	if c.UpdatePerSecond == 0 {
		for index := 0; index < count; index++ {
			err := c.update(tCtx, map[string]any{"Index": index, "Count": count})
			if err != nil {
				tCtx.Fatalf("Failed to update object: %w", err)
			}
		}
		return
	}

	ticker := time.NewTicker(time.Second / time.Duration(c.UpdatePerSecond))
	defer ticker.Stop()
	for index := 0; index < count; index++ {
		select {
		case <-ticker.C:
			err := c.update(tCtx, map[string]any{"Index": index, "Count": count})
			if err != nil {
				tCtx.Fatalf("Failed to update object: %w", err)
			}
		case <-tCtx.Done():
			return
		}
	}
}

func (c *updateAny) update(tCtx ktesting.TContext, env map[string]any) error {
	var obj *unstructured.Unstructured
	if err := getSpecFromTextTemplateFile(c.TemplatePath, env, &obj); err != nil {
		return fmt.Errorf("%s: parsing failed: %w", c.TemplatePath, err)
	}

	if c.cachedMapping == nil {
		mapping, err := restMappingFromUnstructuredObj(tCtx, obj)
		if err != nil {
			return err
		}
		c.cachedMapping = mapping
	}
	resourceClient := tCtx.Dynamic().Resource(c.cachedMapping.Resource)

	options := metav1.UpdateOptions{
		// If the YAML input is invalid, then we want the
		// apiserver to tell us via an error. This can
		// happen because decoding into an unstructured object
		// doesn't validate.
		FieldValidation: "Strict",
	}
	if c.Namespace != "" {
		if c.cachedMapping.Scope.Name() != meta.RESTScopeNameNamespace {
			return fmt.Errorf("namespace %q set for %q, but %q has scope %q", c.Namespace, c.TemplatePath, c.cachedMapping.GroupVersionKind, c.cachedMapping.Scope.Name())
		}
		_, err := resourceClient.Namespace(c.Namespace).Update(tCtx, obj, options, c.Subresources...)
		if err != nil {
			return fmt.Errorf("failed to update object in namespace %q: %w", c.Namespace, err)
		}
		_, err = resourceClient.Namespace(c.Namespace).UpdateStatus(tCtx, obj, options)
		if err != nil {
			return fmt.Errorf("failed to update object status in namespace %q: %w", c.Namespace, err)
		}
		return nil
	}
	if c.cachedMapping.Scope.Name() != meta.RESTScopeNameRoot {
		return fmt.Errorf("namespace not set for %q, but %q has scope %q", c.TemplatePath, c.cachedMapping.GroupVersionKind, c.cachedMapping.Scope.Name())
	}
	_, err := resourceClient.Update(tCtx, obj, options, c.Subresources...)
	if err != nil {
		return fmt.Errorf("failed to update object: %w", err)
	}
	_, err = resourceClient.UpdateStatus(tCtx, obj, options)
	if err != nil {
		return fmt.Errorf("failed to update object status: %w", err)
	}
	return nil
}
