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

package pruning

import (
	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
)

// Prune removes object fields in obj which are not specified in s. It skips TypeMeta and ObjectMeta fields
// if XEmbeddedResource is set to true, or for the root if isResourceRoot=true.
func Prune(obj interface{}, s *structuralschema.Structural, isResourceRoot bool) {
	if isResourceRoot {
		if s == nil {
			s = &structuralschema.Structural{}
		}
		if !s.XEmbeddedResource {
			clone := *s
			clone.XEmbeddedResource = true
			s = &clone
		}
	}
	prune(obj, s)
}

var metaFields = map[string]bool{
	"apiVersion": true,
	"kind":       true,
	"metadata":   true,
}

func prune(x interface{}, s *structuralschema.Structural) {
	if s != nil && s.XPreserveUnknownFields {
		skipPrune(x, s)
		return
	}

	switch x := x.(type) {
	case map[string]interface{}:
		if s == nil {
			for k := range x {
				delete(x, k)
			}
			return
		}
		for k, v := range x {
			if s.XEmbeddedResource && metaFields[k] {
				continue
			}
			prop, ok := s.Properties[k]
			if ok {
				prune(v, &prop)
			} else if s.AdditionalProperties != nil {
				prune(v, s.AdditionalProperties.Structural)
			} else {
				delete(x, k)
			}
		}
	case []interface{}:
		if s == nil {
			for _, v := range x {
				prune(v, nil)
			}
			return
		}
		for _, v := range x {
			prune(v, s.Items)
		}
	default:
		// scalars, do nothing
	}
}

func skipPrune(x interface{}, s *structuralschema.Structural) {
	if s == nil {
		return
	}

	switch x := x.(type) {
	case map[string]interface{}:
		for k, v := range x {
			if s.XEmbeddedResource && metaFields[k] {
				continue
			}
			if prop, ok := s.Properties[k]; ok {
				prune(v, &prop)
			} else if s.AdditionalProperties != nil {
				prune(v, s.AdditionalProperties.Structural)
			}
		}
	case []interface{}:
		for _, v := range x {
			skipPrune(v, s.Items)
		}
	default:
		// scalars, do nothing
	}
}
