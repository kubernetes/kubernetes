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

package patch

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/managedfields"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/openapi"
	"k8s.io/kube-openapi/pkg/spec3"
)

type TypeConverterManager interface {
	// GetTypeConverter returns a type converter for the given GVK
	GetTypeConverter(gvk schema.GroupVersionKind) managedfields.TypeConverter
	Run(ctx context.Context)
}

func NewTypeConverterManager(
	staticTypeConverter managedfields.TypeConverter,
	openapiClient openapi.Client,
) TypeConverterManager {
	return &typeConverterManager{
		staticTypeConverter: staticTypeConverter,
		openapiClient:       openapiClient,
		typeConverterMap:    make(map[schema.GroupVersion]typeConverterCacheEntry),
		lastFetchedPaths:    make(map[schema.GroupVersion]openapi.GroupVersion),
	}
}

type typeConverterCacheEntry struct {
	typeConverter managedfields.TypeConverter
	entry         openapi.GroupVersion
}

// typeConverterManager helps us make sure we have an up to date schema and
// type converter for our openapi models. It should be connfigured to use a
// static type converter for natively typed schemas, and fetches the schema
// for CRDs/other over the network on demand (trying to reduce network calls where necessary)
type typeConverterManager struct {
	// schemaCache is used to cache the schema for a given GVK
	staticTypeConverter managedfields.TypeConverter

	// discoveryClient is used to fetch the schema for a given GVK
	openapiClient openapi.Client

	lock sync.RWMutex

	typeConverterMap map[schema.GroupVersion]typeConverterCacheEntry
	lastFetchedPaths map[schema.GroupVersion]openapi.GroupVersion
}

func (t *typeConverterManager) Run(ctx context.Context) {
	// Loop every 5s refershing the OpenAPI schema list to know which
	// schemas have been invalidated. This should use e-tags under the hood
	_ = wait.PollUntilContextCancel(ctx, 5*time.Second, true, func(_ context.Context) (done bool, err error) {
		paths, err := t.openapiClient.Paths()
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("failed to fetch openapi paths: %w", err))
			return false, nil
		}

		// The /openapi/v3 endpoint contains a list of paths whose ServerRelativeURL
		// value changes every time the schema is updated. So we poll /openapi/v3
		// to get the "version number" for each schema, and invalidate our cache
		// if the version number has changed since we pulled it.
		parsedPaths := make(map[schema.GroupVersion]openapi.GroupVersion, len(paths))
		for path, entry := range paths {
			if !strings.HasPrefix(path, "apis/") && !strings.HasPrefix(path, "api/") {
				continue
			}
			path = strings.TrimPrefix(path, "apis/")
			path = strings.TrimPrefix(path, "api/")

			gv, err := schema.ParseGroupVersion(path)
			if err != nil {
				utilruntime.HandleError(fmt.Errorf("failed to parse group version %q: %w", path, err))
				return false, nil
			}

			parsedPaths[gv] = entry
		}

		t.lock.Lock()
		defer t.lock.Unlock()
		t.lastFetchedPaths = parsedPaths
		return false, nil
	})
}

func (t *typeConverterManager) GetTypeConverter(gvk schema.GroupVersionKind) managedfields.TypeConverter {
	// Check to see if the static type converter handles this GVK
	if t.staticTypeConverter != nil {
		//!TODO: Add ability to check existence to type converter
		// working around for now but seeing if getting a typed version of an
		// empty object returns error
		stub := &unstructured.Unstructured{}
		stub.SetGroupVersionKind(gvk)

		if _, err := t.staticTypeConverter.ObjectToTyped(stub); err == nil {
			return t.staticTypeConverter
		}
	}

	gv := gvk.GroupVersion()

	existing, entry, err := func() (managedfields.TypeConverter, openapi.GroupVersion, error) {
		t.lock.RLock()
		defer t.lock.RUnlock()

		// If schema is not supported by static type converter, ask discovery
		// for the schema
		entry, ok := t.lastFetchedPaths[gv]
		if !ok {
			// If we can't get the schema, we can't do anything
			return nil, nil, fmt.Errorf("no schema for %v", gvk)
		}

		// If the entry schema has not changed, used the same type converter
		if existing, ok := t.typeConverterMap[gv]; ok && existing.entry.ServerRelativeURL() == entry.ServerRelativeURL() {
			// If we have a type converter for this GVK, return it
			return existing.typeConverter, existing.entry, nil
		}

		return nil, entry, nil
	}()
	if err != nil {
		utilruntime.HandleError(err)
		return nil
	} else if existing != nil {
		return existing
	}

	schBytes, err := entry.Schema(runtime.ContentTypeJSON)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("failed to get schema for %v: %w", gvk, err))
		return nil
	}

	var sch spec3.OpenAPI
	if err := json.Unmarshal(schBytes, &sch); err != nil {
		utilruntime.HandleError(fmt.Errorf("failed to unmarshal schema for %v: %w", gvk, err))
		return nil
	}

	// The schema has changed, or there is no entry for it, generate
	// a new type converter for this GV
	tc, err := managedfields.NewTypeConverter(sch.Components.Schemas, false)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("failed to create type converter for %v: %w", gvk, err))
		return nil
	}

	t.lock.Lock()
	defer t.lock.Unlock()

	t.typeConverterMap[gv] = typeConverterCacheEntry{
		typeConverter: tc,
		entry:         entry,
	}

	return tc
}
