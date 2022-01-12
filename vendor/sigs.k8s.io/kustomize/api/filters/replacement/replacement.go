// Copyright 2021 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package replacement

import (
	"fmt"
	"strings"

	"sigs.k8s.io/kustomize/api/internal/utils"
	"sigs.k8s.io/kustomize/api/resource"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/resid"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

type Filter struct {
	Replacements []types.Replacement `json:"replacements,omitempty" yaml:"replacements,omitempty"`
}

// Filter replaces values of targets with values from sources
func (f Filter) Filter(nodes []*yaml.RNode) ([]*yaml.RNode, error) {
	for _, r := range f.Replacements {
		if r.Source == nil || r.Targets == nil {
			return nil, fmt.Errorf("replacements must specify a source and at least one target")
		}
		value, err := getReplacement(nodes, &r)
		if err != nil {
			return nil, err
		}
		nodes, err = applyReplacement(nodes, value, r.Targets)
		if err != nil {
			return nil, err
		}
	}
	return nodes, nil
}

func applyReplacement(nodes []*yaml.RNode, value *yaml.RNode, targets []*types.TargetSelector) ([]*yaml.RNode, error) {
	for _, t := range targets {
		if t.Select == nil {
			return nil, fmt.Errorf("target must specify resources to select")
		}
		if len(t.FieldPaths) == 0 {
			t.FieldPaths = []string{types.DefaultReplacementFieldPath}
		}
		for _, n := range nodes {
			ids, err := utils.MakeResIds(n)
			if err != nil {
				return nil, err
			}

			// filter targets by label and annotation selectors
			selectByAnnoAndLabel, err := selectByAnnoAndLabel(n, t)
			if err != nil {
				return nil, err
			}
			if !selectByAnnoAndLabel {
				continue
			}

			// filter targets by matching resource IDs
			for _, id := range ids {
				if id.IsSelectedBy(t.Select.ResId) && !rejectId(t.Reject, &id) {
					err := applyToNode(n, value, t)
					if err != nil {
						return nil, err
					}
					break
				}
			}
		}
	}
	return nodes, nil
}

func selectByAnnoAndLabel(n *yaml.RNode, t *types.TargetSelector) (bool, error) {
	if matchesSelect, err := matchesAnnoAndLabelSelector(n, t.Select); !matchesSelect || err != nil {
		return false, err
	}
	for _, reject := range t.Reject {
		if reject.AnnotationSelector == "" && reject.LabelSelector == "" {
			continue
		}
		if m, err := matchesAnnoAndLabelSelector(n, reject); m || err != nil {
			return false, err
		}
	}
	return true, nil
}

func matchesAnnoAndLabelSelector(n *yaml.RNode, selector *types.Selector) (bool, error) {
	r := resource.Resource{RNode: *n}
	annoMatch, err := r.MatchesAnnotationSelector(selector.AnnotationSelector)
	if err != nil {
		return false, err
	}
	labelMatch, err := r.MatchesLabelSelector(selector.LabelSelector)
	if err != nil {
		return false, err
	}
	return annoMatch && labelMatch, nil
}

func rejectId(rejects []*types.Selector, id *resid.ResId) bool {
	for _, r := range rejects {
		if !r.ResId.IsEmpty() && id.IsSelectedBy(r.ResId) {
			return true
		}
	}
	return false
}

func applyToNode(node *yaml.RNode, value *yaml.RNode, target *types.TargetSelector) error {
	for _, fp := range target.FieldPaths {
		fieldPath := utils.SmarterPathSplitter(fp, ".")
		var t *yaml.RNode
		var err error
		if target.Options != nil && target.Options.Create {
			t, err = node.Pipe(yaml.LookupCreate(value.YNode().Kind, fieldPath...))
		} else {
			t, err = node.Pipe(yaml.Lookup(fieldPath...))
		}
		if err != nil {
			return err
		}
		if t != nil {
			if err = setTargetValue(target.Options, t, value); err != nil {
				return err
			}
		}
	}
	return nil
}

func setTargetValue(options *types.FieldOptions, t *yaml.RNode, value *yaml.RNode) error {
	value = value.Copy()
	if options != nil && options.Delimiter != "" {
		if t.YNode().Kind != yaml.ScalarNode {
			return fmt.Errorf("delimiter option can only be used with scalar nodes")
		}
		tv := strings.Split(t.YNode().Value, options.Delimiter)
		v := yaml.GetValue(value)
		// TODO: Add a way to remove an element
		switch {
		case options.Index < 0: // prefix
			tv = append([]string{v}, tv...)
		case options.Index >= len(tv): // suffix
			tv = append(tv, v)
		default: // replace an element
			tv[options.Index] = v
		}
		value.YNode().Value = strings.Join(tv, options.Delimiter)
	}
	t.SetYNode(value.YNode())
	return nil
}

func getReplacement(nodes []*yaml.RNode, r *types.Replacement) (*yaml.RNode, error) {
	source, err := selectSourceNode(nodes, r.Source)
	if err != nil {
		return nil, err
	}

	if r.Source.FieldPath == "" {
		r.Source.FieldPath = types.DefaultReplacementFieldPath
	}
	fieldPath := utils.SmarterPathSplitter(r.Source.FieldPath, ".")

	rn, err := source.Pipe(yaml.Lookup(fieldPath...))
	if err != nil {
		return nil, err
	}
	if rn.IsNilOrEmpty() {
		return nil, fmt.Errorf("fieldPath `%s` is missing for replacement source %s", r.Source.FieldPath, r.Source)
	}

	return getRefinedValue(r.Source.Options, rn)
}

func getRefinedValue(options *types.FieldOptions, rn *yaml.RNode) (*yaml.RNode, error) {
	if options == nil || options.Delimiter == "" {
		return rn, nil
	}
	if rn.YNode().Kind != yaml.ScalarNode {
		return nil, fmt.Errorf("delimiter option can only be used with scalar nodes")
	}
	value := strings.Split(yaml.GetValue(rn), options.Delimiter)
	if options.Index >= len(value) || options.Index < 0 {
		return nil, fmt.Errorf("options.index %d is out of bounds for value %s", options.Index, yaml.GetValue(rn))
	}
	n := rn.Copy()
	n.YNode().Value = value[options.Index]
	return n, nil
}

// selectSourceNode finds the node that matches the selector, returning
// an error if multiple or none are found
func selectSourceNode(nodes []*yaml.RNode, selector *types.SourceSelector) (*yaml.RNode, error) {
	var matches []*yaml.RNode
	for _, n := range nodes {
		ids, err := utils.MakeResIds(n)
		if err != nil {
			return nil, err
		}
		for _, id := range ids {
			if id.IsSelectedBy(selector.ResId) {
				if len(matches) > 0 {
					return nil, fmt.Errorf(
						"multiple matches for selector %s", selector)
				}
				matches = append(matches, n)
				break
			}
		}
	}
	if len(matches) == 0 {
		return nil, fmt.Errorf("nothing selected by %s", selector)
	}
	return matches[0], nil
}
