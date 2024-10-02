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
	"bytes"
	"context"
	"fmt"
	"html/template"
	"os"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/restmapper"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/yaml"
)

// createAny defines an op where some object gets created from a YAML file.
// The nameset can be specified.
type createAny struct {
	// Must match createAnyOpcode.
	Opcode operationCode
	// Namespace the object should be created in. Must be empty for cluster-scoped objects.
	Namespace string
	// Path to spec file describing the object to create.
	// This will be processed with text/template.
	// .Index will be in the range [0, Count-1] when creating
	// more than one object. .Count is the total number of objects.
	TemplatePath string
	// Count determines how many objects get created. Defaults to 1 if unset.
	Count      *int
	CountParam string
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

func (c createAny) patchParams(w *workload) (realOp, error) {
	if c.CountParam != "" {
		count, err := w.Params.get(c.CountParam[1:])
		if err != nil {
			return nil, err
		}
		c.Count = ptr.To(count)
	}
	return &c, c.isValid(false)
}

func (c *createAny) requiredNamespaces() []string {
	if c.Namespace == "" {
		return nil
	}
	return []string{c.Namespace}
}

func (c *createAny) run(tCtx ktesting.TContext) {
	count := 1
	if c.Count != nil {
		count = *c.Count
	}
	for index := 0; index < count; index++ {
		c.create(tCtx, map[string]any{"Index": index, "Count": count})
	}
}

func (c *createAny) create(tCtx ktesting.TContext, env map[string]any) {
	var obj *unstructured.Unstructured
	if err := getSpecFromTextTemplateFile(c.TemplatePath, env, &obj); err != nil {
		tCtx.Fatalf("%s: parsing failed: %v", c.TemplatePath, err)
	}

	// Not caching the discovery result isn't very efficient, but good enough when
	// createAny isn't done often.
	discoveryCache := memory.NewMemCacheClient(tCtx.Client().Discovery())
	restMapper := restmapper.NewDeferredDiscoveryRESTMapper(discoveryCache)
	gv, err := schema.ParseGroupVersion(obj.GetAPIVersion())
	if err != nil {
		tCtx.Fatalf("%s: extract group+version from object %q: %v", c.TemplatePath, klog.KObj(obj), err)
	}
	gk := schema.GroupKind{Group: gv.Group, Kind: obj.GetKind()}

	create := func() error {
		mapping, err := restMapper.RESTMapping(gk, gv.Version)
		if err != nil {
			// Cached mapping might be stale, refresh on next try.
			restMapper.Reset()
			return fmt.Errorf("map %q to resource: %v", gk, err)
		}
		resourceClient := tCtx.Dynamic().Resource(mapping.Resource)
		options := metav1.CreateOptions{
			// If the YAML input is invalid, then we want the
			// apiserver to tell us via an error. This can
			// happen because decoding into an unstructured object
			// doesn't validate.
			FieldValidation: "Strict",
		}
		if c.Namespace != "" {
			if mapping.Scope.Name() != meta.RESTScopeNameNamespace {
				return fmt.Errorf("namespace %q set for %q, but %q has scope %q", c.Namespace, c.TemplatePath, gk, mapping.Scope.Name())
			}
			_, err = resourceClient.Namespace(c.Namespace).Create(tCtx, obj, options)
		} else {
			if mapping.Scope.Name() != meta.RESTScopeNameRoot {
				return fmt.Errorf("namespace not set for %q, but %q has scope %q", c.TemplatePath, gk, mapping.Scope.Name())
			}
			_, err = resourceClient.Create(tCtx, obj, options)
		}
		return err
	}
	// Retry, some errors (like CRD just created and type not ready for use yet) are temporary.
	ctx, cancel := context.WithTimeout(tCtx, 20*time.Second)
	defer cancel()
	for {
		err := create()
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

func getSpecFromTextTemplateFile(path string, env map[string]any, spec interface{}) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	fm := template.FuncMap{"div": func(a, b int) int {
		return a / b
	}}
	tmpl, err := template.New("object").Funcs(fm).Parse(string(content))
	if err != nil {
		return err
	}
	var buffer bytes.Buffer
	if err := tmpl.Execute(&buffer, env); err != nil {
		return err
	}

	return yaml.UnmarshalStrict(buffer.Bytes(), spec)
}
