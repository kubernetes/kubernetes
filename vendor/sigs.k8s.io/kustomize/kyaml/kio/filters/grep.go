// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package filters

import (
	"regexp"
	"strings"

	"sigs.k8s.io/kustomize/kyaml/kio"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

type GrepType int

const (
	Regexp GrepType = 1 << iota
	GreaterThanEq
	GreaterThan
	LessThan
	LessThanEq
)

// GrepFilter filters RNodes with a matching field
type GrepFilter struct {
	Path        []string `yaml:"path,omitempty"`
	Value       string   `yaml:"value,omitempty"`
	MatchType   GrepType `yaml:"matchType,omitempty"`
	InvertMatch bool     `yaml:"invertMatch,omitempty"`
	Compare     func(a, b string) (int, error)
}

var _ kio.Filter = GrepFilter{}

func (f GrepFilter) Filter(input []*yaml.RNode) ([]*yaml.RNode, error) {
	// compile the regular expression 1 time if we are matching using regex
	var reg *regexp.Regexp
	var err error
	if f.MatchType == Regexp || f.MatchType == 0 {
		reg, err = regexp.Compile(f.Value)
		if err != nil {
			return nil, err
		}
	}

	var output kio.ResourceNodeSlice
	for i := range input {
		node := input[i]
		val, err := node.Pipe(&yaml.PathMatcher{Path: f.Path})
		if err != nil {
			return nil, err
		}
		if val == nil || len(val.Content()) == 0 {
			if f.InvertMatch {
				output = append(output, input[i])
			}
			continue
		}
		found := false
		err = val.VisitElements(func(elem *yaml.RNode) error {
			// get the value
			var str string
			if f.MatchType == Regexp {
				style := elem.YNode().Style
				defer func() { elem.YNode().Style = style }()
				elem.YNode().Style = yaml.FlowStyle
				str, err = elem.String()
				if err != nil {
					return err
				}
				str = strings.TrimSpace(strings.ReplaceAll(str, `"`, ""))
			} else {
				// if not regexp, then it needs to parse into a quantity and comments will
				// break that
				str = elem.YNode().Value
				if str == "" {
					return nil
				}
			}

			if f.MatchType == Regexp || f.MatchType == 0 {
				if reg.MatchString(str) {
					found = true
				}
				return nil
			}

			comp, err := f.Compare(str, f.Value)
			if err != nil {
				return err
			}

			if f.MatchType == GreaterThan && comp > 0 {
				found = true
			}
			if f.MatchType == GreaterThanEq && comp >= 0 {
				found = true
			}
			if f.MatchType == LessThan && comp < 0 {
				found = true
			}
			if f.MatchType == LessThanEq && comp <= 0 {
				found = true
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
		if found == f.InvertMatch {
			continue
		}

		output = append(output, input[i])
	}
	return output, nil
}
