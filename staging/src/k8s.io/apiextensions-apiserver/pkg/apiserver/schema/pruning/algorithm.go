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
	"sort"

	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
)

// PruneWithOptions removes object fields in obj which are not specified in s. It skips TypeMeta
// and ObjectMeta fields if XEmbeddedResource is set to true, or for the root if isResourceRoot=true,
// i.e. it does not prune unknown metadata fields.
// It returns the set of fields that it prunes if opts.TrackUnknownFieldPaths is true
func PruneWithOptions(obj interface{}, s *structuralschema.Structural, isResourceRoot bool, opts structuralschema.UnknownFieldPathOptions) []string {
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
	prune(obj, s, &opts)
	sort.Strings(opts.UnknownFieldPaths)
	return opts.UnknownFieldPaths
}

// Prune is equivalent to
// PruneWithOptions(obj, s, isResourceRoot, structuralschema.UnknownFieldPathOptions{})
func Prune(obj interface{}, s *structuralschema.Structural, isResourceRoot bool) {
	PruneWithOptions(obj, s, isResourceRoot, structuralschema.UnknownFieldPathOptions{})
}

var metaFields = map[string]bool{
	"apiVersion": true,
	"kind":       true,
	"metadata":   true,
}

func prune(x interface{}, s *structuralschema.Structural, opts *structuralschema.UnknownFieldPathOptions) {
	if s != nil && s.XPreserveUnknownFields {
		skipPrune(x, s, opts)
		return
	}

	origPathLen := len(opts.ParentPath)
	defer func() {
		opts.ParentPath = opts.ParentPath[:origPathLen]
	}()
	switch x := x.(type) {
	case map[string]interface{}:
		if s == nil {
			for k := range x {
				opts.RecordUnknownField(k)
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
				opts.AppendKey(k)
				prune(v, &prop, opts)
				opts.ParentPath = opts.ParentPath[:origPathLen]
			} else if s.AdditionalProperties != nil {
				opts.AppendKey(k)
				prune(v, s.AdditionalProperties.Structural, opts)
				opts.ParentPath = opts.ParentPath[:origPathLen]
			} else {
				if !metaFields[k] || len(opts.ParentPath) > 0 {
					opts.RecordUnknownField(k)
				}
				delete(x, k)
			}
		}
	case []interface{}:
		if s == nil {
			for i, v := range x {
				opts.AppendIndex(i)
				prune(v, nil, opts)
				opts.ParentPath = opts.ParentPath[:origPathLen]
			}
			return
		}
		for i, v := range x {
			opts.AppendIndex(i)
			prune(v, s.Items, opts)
			opts.ParentPath = opts.ParentPath[:origPathLen]
		}
	default:
		// scalars, do nothing
	}
}

func skipPrune(x interface{}, s *structuralschema.Structural, opts *structuralschema.UnknownFieldPathOptions) {
	if s == nil {
		return
	}
	origPathLen := len(opts.ParentPath)
	defer func() {
		opts.ParentPath = opts.ParentPath[:origPathLen]
	}()

	switch x := x.(type) {
	case map[string]interface{}:
		for k, v := range x {
			if s.XEmbeddedResource && metaFields[k] {
				continue
			}
			if prop, ok := s.Properties[k]; ok {
				opts.AppendKey(k)
				prune(v, &prop, opts)
				opts.ParentPath = opts.ParentPath[:origPathLen]
			} else if s.AdditionalProperties != nil {
				opts.AppendKey(k)
				prune(v, s.AdditionalProperties.Structural, opts)
				opts.ParentPath = opts.ParentPath[:origPathLen]
			}
		}
	case []interface{}:
		for i, v := range x {
			opts.AppendIndex(i)
			skipPrune(v, s.Items, opts)
			opts.ParentPath = opts.ParentPath[:origPathLen]
		}
	default:
		// scalars, do nothing
	}
}
