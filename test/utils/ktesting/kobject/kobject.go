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

package kobject

import (
	"errors"
	"fmt"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// This package works by converting to/from unstructured.Unstructured and then
// calling the dynamic client with a group/version/resource determined via
// the REST mapper.
//
// We could avoid double conversion by replicating the code of the dynamic client
// and decoding the response directly into the right type. Better for performance,
// but more code to maintain, so not worth it?

func Get[T runtime.Object](tCtx ktesting.TContext, what Object, options metav1.GetOptions) (T, error) {
	tCtx.Helper()

	var object T
	gvk, err := getGVK(tCtx, object)
	if err != nil {
		return object, err
	}
	mapping, err := tCtx.RESTMapper().RESTMapping(gvk.GroupKind(), gvk.Version)
	if err != nil {
		// Try again once after resetting the mapping. The information
		// might have been stale.
		tCtx.RESTMapper().Reset()
		mapping, err = tCtx.RESTMapper().RESTMapping(gvk.GroupKind(), gvk.Version)
	}
	if err != nil {
		return object, fmt.Errorf("not resource found for %T: %w", object, err)
	}

	var client dynamic.ResourceInterface
	if mapping.Scope.Name() == meta.RESTScopeNameNamespace {
		client = tCtx.Dynamic().Resource(mapping.Resource).Namespace(what.GetNamespace())
	} else {
		client = tCtx.Dynamic().Resource(mapping.Resource)
	}
	obj, err := client.Get(tCtx, what.GetName(), options)
	if err != nil {
		return object, fmt.Errorf("get failed: %w", err)
	}
	if err := scheme.Convert(obj, object, nil); err != nil {
		return object, fmt.Errorf("conversion from %s to %T failed: %w", gvk, object, err)
	}

	return object, nil
}

// Create creates an object. If the corresponding resource is namespaced, then
// the namespace in the object is used or, if that is unset, the default
// namespace in the context (see [WithNamespace]).
//
// Supported object types are [unstructured.Unstructured] and any of the API
// types that the typed clients supported by ktesting support (like Pod,
// CustomResourceDefinition, etc.)
//
// The object will get removed during test cleanup automatically. This can be
// disabled via [WithCleanup].
//
// A message gets logged about creating and (if that is enabled) deleting the
// object at V(0). Use [ktesting.WithLogger] and a logger with higher verbosity
// reduce log output.
//
// API calls get retried automatically if the apiserver's response indicates
// that the error was transient.
func Create[T runtime.Object](tCtx ktesting.TContext, object T, options metav1.CreateOptions) (T, error) {
	tCtx.Helper()

	var result T
	gvk := object.GetObjectKind().GroupVersionKind()

	// gvk might be unset, for example when T is *v1.Pod.
	// In that case we have to look it up.
	//
	// For unstructured.Unstructured it has to be set because
	// there is no way to guess it.
	var emptyGVK schema.GroupVersionKind
	if gvk == emptyGVK {
		objGVK, err := getGVK(tCtx, object)
		if err != nil {
			return result, err
		}
		gvk = objGVK
	}

	mapping, err := tCtx.RESTMapper().RESTMapping(gvk.GroupKind(), gvk.Version)
	if err != nil {
		// Try again once after resetting the mapping. The information
		// might have been stale.
		tCtx.RESTMapper().Reset()
		mapping, err = tCtx.RESTMapper().RESTMapping(gvk.GroupKind(), gvk.Version)
	}
	if err != nil {
		return result, fmt.Errorf("not resource found for %T: %w", result, err)
	}

	accessor, err := meta.Accessor(object)
	if err != nil {
		return result, fmt.Errorf("cannot access meta data of %T: %w", object, err)
	}
	named := NamespacedName{
		Name: accessor.GetName(),
	}
	var client dynamic.ResourceInterface
	if mapping.Scope.Name() == meta.RESTScopeNameNamespace {
		namespace := accessor.GetNamespace()
		if namespace == "" {
			namespace = tCtx.Namespace()
		}
		if namespace == "" {
			return result, fmt.Errorf("%T is namespaced, but namespace in object and context are unset", object)
		}
		named.Namespace = namespace
		client = tCtx.Dynamic().Resource(mapping.Resource).Namespace(namespace)
	} else {
		client = tCtx.Dynamic().Resource(mapping.Resource)
	}

	tCtx.Logger().Info("Creating object", "gvk", gvk, "object", klog.KObj(named))
	tCtx.Logf("Creating %q %s", gvk, named)

	// This is where generics in Go stop being useful:
	// we need to handle API types and unstructured.Unstructured differently,
	// but can only do so via runtime reflection.
	out, err := create(tCtx, client, gvk, object, options)
	if err != nil {
		return result, err
	}

	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		tCtx.Logger().Info("Cleaning up object", "gvk", gvk, "object", klog.KObj(named))
		err := client.Delete(tCtx, named.Name, metav1.DeleteOptions{})
		if err == nil || apierrors.IsNotFound(err) {
			return
		}
		tCtx.Errorf("Deleting %s %s failed: %v", gvk, named, err)
	})

	return out.(T), nil
}

func create(tCtx ktesting.TContext, client dynamic.ResourceInterface, gvk schema.GroupVersionKind, object any, options metav1.CreateOptions) (any, error) {
	tCtx.Helper()

	if object, ok := object.(*unstructured.Unstructured); ok {
		return client.Create(tCtx, object, options)
	}

	var out unstructured.Unstructured
	if err := scheme.Convert(object, &out, nil); err != nil {
		return nil, fmt.Errorf("conversion from %T to unstructured %s failed: %w", object, gvk, err)
	}
	in, err := client.Create(tCtx, &out, options)
	if err != nil {
		return nil, err
	}
	result, err := scheme.New(gvk)
	if err != nil {
		return nil, fmt.Errorf("cannot create new object of type %T from %s: %w", object, gvk, err)
	}
	if err := scheme.Convert(in, result, nil); err != nil {
		return result, fmt.Errorf("conversion from unstructured %s to %T failed: %w", gvk, result, err)
	}

	return result, nil
}

func getGVK(tCtx ktesting.TContext, object runtime.Object) (gvk schema.GroupVersionKind, finalErr error) {
	defer func() {
		if finalErr != nil {
			finalErr = fmt.Errorf("look up group/version/kind for %T: %w", object, finalErr)
		}
	}()

	gvks, unversioned, err := scheme.ObjectKinds(object)
	if err != nil {
		return gvk, err
	}
	if unversioned {
		return gvk, errors.New("not versioned")
	}
	if len(gvks) == 0 {
		return gvk, errors.New("no type information")
	}
	if len(gvks) > 1 {
		return gvk, fmt.Errorf("type is ambiguous (%v)", gvks)
	}
	return gvks[0], nil
}
