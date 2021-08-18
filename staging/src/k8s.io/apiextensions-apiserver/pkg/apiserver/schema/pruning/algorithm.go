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
	"strconv"
	"strings"

	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
)

// PruneOptions sets options for pruning
// unknown fields
type PruneOptions struct {
	// parentPath collects the path that the pruning
	// takes as it traverses the object.
	// It is used to report the full path to any unknown
	// fields that the pruning encounters.
	// It is only populated if ReturnPruned is true.
	parentPath []string

	// prunedPaths collects pruned field paths resulting from
	// calls to recordPrunedKey.
	// It is only populated if ReturnPruned is true.
	prunedPaths []string

	// ReturnPruned defines whether we want to track the
	// fields that are pruned
	ReturnPruned bool
}

// PruneWithOptions removes object fields in obj which are not specified in s. It skips TypeMeta
// and ObjectMeta fields if XEmbeddedResource is set to true, or for the root if isResourceRoot=true,
// i.e. it does not prune unknown metadata fields.
// It returns the set of fields that it prunes if opts.ReturnPruned is true
func PruneWithOptions(obj interface{}, s *structuralschema.Structural, isResourceRoot bool, opts PruneOptions) []string {
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
	sort.Strings(opts.prunedPaths)
	return opts.prunedPaths
}

// Prune is equivalent to
// PruneWithOptions(obj, s, isResourceRoot, PruneOptions{})
func Prune(obj interface{}, s *structuralschema.Structural, isResourceRoot bool) {
	PruneWithOptions(obj, s, isResourceRoot, PruneOptions{})
}

var metaFields = map[string]bool{
	"apiVersion": true,
	"kind":       true,
	"metadata":   true,
}

func (p *PruneOptions) recordPrunedKey(key string) {
	if !p.ReturnPruned {
		return
	}
	l := len(p.parentPath)
	p.appendKey(key)
	p.prunedPaths = append(p.prunedPaths, strings.Join(p.parentPath, ""))
	p.parentPath = p.parentPath[:l]
}

func (p *PruneOptions) appendKey(key string) {
	if !p.ReturnPruned {
		return
	}
	if len(p.parentPath) > 0 {
		p.parentPath = append(p.parentPath, ".")
	}
	p.parentPath = append(p.parentPath, key)
}

func (p *PruneOptions) appendIndex(index int) {
	if !p.ReturnPruned {
		return
	}
	p.parentPath = append(p.parentPath, "[", strconv.Itoa(index), "]")
}

func prune(x interface{}, s *structuralschema.Structural, opts *PruneOptions) {
	if s != nil && s.XPreserveUnknownFields {
		skipPrune(x, s, opts)
		return
	}

	origPathLen := len(opts.parentPath)
	defer func() {
		opts.parentPath = opts.parentPath[:origPathLen]
	}()
	switch x := x.(type) {
	case map[string]interface{}:
		if s == nil {
			for k := range x {
				opts.recordPrunedKey(k)
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
				opts.appendKey(k)
				prune(v, &prop, opts)
				opts.parentPath = opts.parentPath[:origPathLen]
			} else if s.AdditionalProperties != nil {
				opts.appendKey(k)
				prune(v, s.AdditionalProperties.Structural, opts)
				opts.parentPath = opts.parentPath[:origPathLen]
			} else {
				if !metaFields[k] || len(opts.parentPath) > 0 {
					opts.recordPrunedKey(k)
				}
				delete(x, k)
			}
		}
	case []interface{}:
		if s == nil {
			for i, v := range x {
				opts.appendIndex(i)
				prune(v, nil, opts)
				opts.parentPath = opts.parentPath[:origPathLen]
			}
			return
		}
		for i, v := range x {
			opts.appendIndex(i)
			prune(v, s.Items, opts)
			opts.parentPath = opts.parentPath[:origPathLen]
		}
	default:
		// scalars, do nothing
	}
}

func skipPrune(x interface{}, s *structuralschema.Structural, opts *PruneOptions) {
	if s == nil {
		return
	}
	origPathLen := len(opts.parentPath)
	defer func() {
		opts.parentPath = opts.parentPath[:origPathLen]
	}()

	switch x := x.(type) {
	case map[string]interface{}:
		for k, v := range x {
			if s.XEmbeddedResource && metaFields[k] {
				continue
			}
			if prop, ok := s.Properties[k]; ok {
				opts.appendKey(k)
				prune(v, &prop, opts)
				opts.parentPath = opts.parentPath[:origPathLen]
			} else if s.AdditionalProperties != nil {
				opts.appendKey(k)
				prune(v, s.AdditionalProperties.Structural, opts)
				opts.parentPath = opts.parentPath[:origPathLen]
			}
		}
	case []interface{}:
		for i, v := range x {
			opts.appendIndex(i)
			prune(v, s.Items, opts)
			opts.parentPath = opts.parentPath[:origPathLen]
		}
	default:
		// scalars, do nothing
	}
}
