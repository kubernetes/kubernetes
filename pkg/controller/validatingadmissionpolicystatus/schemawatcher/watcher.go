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

package schemawatcher

import (
	"fmt"
	"sync"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/openapi"
)

var ErrNoHash = fmt.Errorf("fail to obtain hash")

type gvkToHashMap map[schema.GroupVersionKind][32]byte

// Instance keeps track of all types and the hash of their schemas.
type Instance struct {
	client openapi.Client

	groupVersionToHash     map[schema.GroupVersion]string
	groupVersionKindToHash map[schema.GroupVersion]gvkToHashMap
	mu                     sync.Mutex
}

// New creates a new schema watcher with given OpenAPI v3 client.
func New(client openapi.Client) *Instance {
	return &Instance{
		client:                 client,
		groupVersionToHash:     make(map[schema.GroupVersion]string),
		groupVersionKindToHash: make(map[schema.GroupVersion]gvkToHashMap),
	}
}

// ChangedGVKs refreshes and compares the stored schema hashes, and returns
// all GroupVersionKind that have a schema that changes compared to last refresh.
func (w *Instance) ChangedGVKs() ([]schema.GroupVersionKind, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	return w.detectChanges()
}

func (w *Instance) detectChanges() (changedGVKs []schema.GroupVersionKind, err error) {
	paths, err := w.client.Paths()
	if err != nil {
		return nil, err
	}

	newGroupVersionToHash, openapiGroupVersions := parseDiscoveryRoot(paths)
	changed, removed := diffGroupVersionToHash(w.groupVersionToHash, newGroupVersionToHash)

	for gv := range changed {
		if _, ok := w.groupVersionKindToHash[gv]; !ok {
			w.groupVersionKindToHash[gv] = make(gvkToHashMap)
		}
		doc, err := openapiGroupVersions[gv].Schema(runtime.ContentTypeJSON)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("fail to fetch schema: %w", err))
			continue
		}
		changed, err := w.detectChangeInGroupVersion(gv, doc)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("fail to detect changes in GV %q schema: %w", gv, err))
			continue
		}
		changedGVKs = append(changedGVKs, changed...)
	}

	for gv := range removed {
		changed := sets.KeySet(w.groupVersionKindToHash[gv]).UnsortedList()
		delete(w.groupVersionKindToHash, gv)
		changedGVKs = append(changedGVKs, changed...)
	}

	w.groupVersionToHash = newGroupVersionToHash

	return changedGVKs, nil
}

func (w *Instance) detectChangeInGroupVersion(groupVersion schema.GroupVersion, doc []byte) (changedGVKs []schema.GroupVersionKind, err error) {
	newGVKToHashMap, err := parseOpenAPIGroupVersion(groupVersion, doc)
	if err != nil {
		return nil, err
	}
	oldGVKToHashMap := w.groupVersionKindToHash[groupVersion]
	// changed = removed + hash changed
	changed := sets.KeySet(oldGVKToHashMap).Difference(sets.KeySet(newGVKToHashMap))
	for gv, hash := range newGVKToHashMap {
		oldHash := oldGVKToHashMap[gv]
		if hash != oldHash {
			changed.Insert(gv)
		}
	}
	w.groupVersionKindToHash[groupVersion] = newGVKToHashMap

	return changed.UnsortedList(), nil
}

func parseDiscoveryRoot(paths map[string]openapi.GroupVersion) (map[schema.GroupVersion]string, map[schema.GroupVersion]openapi.GroupVersion) {
	hashes := make(map[schema.GroupVersion]string)
	openapiGroupVersions := make(map[schema.GroupVersion]openapi.GroupVersion)
	for path, groupVersion := range paths {
		gv, hash, err := parseOpenAPIPathItem(path, groupVersion)
		if err != nil {
			// malformed OpenAPI v3 response, considered internal error.
			// ignore the broken GV and continue
			utilruntime.HandleError(err)
			continue
		}
		hashes[gv] = hash
		openapiGroupVersions[gv] = groupVersion
	}
	return hashes, openapiGroupVersions
}

func parseOpenAPIGroupVersion(gv schema.GroupVersion, doc []byte) (gvkToHashMap, error) {
	schemasWithHash := parseOpenAPIv3Doc(doc)
	result := make(gvkToHashMap)
	for _, schemaWithHash := range schemasWithHash {
		s := schemaWithHash.Schema
		hash := schemaWithHash.Hash
		var gvks []schema.GroupVersionKind
		err := s.Extensions.GetObject(gvkExtensionName, &gvks)
		if err != nil {
			return nil, err
		}
		for _, gvk := range gvks {
			if gvk.Group != gv.Group || gvk.Version != gv.Version {
				// ignore dependencies, which does not match the current GV
				continue
			}
			result[gvk] = hash
		}
	}
	return result, nil
}

func diffGroupVersionToHash(oldMap, newMap map[schema.GroupVersion]string) (changed, removed sets.Set[schema.GroupVersion]) {
	oldKeySet, newKeySet := sets.KeySet(oldMap), sets.KeySet(newMap)
	removed = oldKeySet.Difference(newKeySet)
	changed = sets.New[schema.GroupVersion]()
	for gv, hash := range newMap {
		oldHash := oldMap[gv]
		if hash != oldHash {
			changed.Insert(gv)
		}
	}
	return
}
