// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package filters

import (
	"fmt"
	"sort"
	"strings"

	"sigs.k8s.io/kustomize/kyaml/kio"
	"sigs.k8s.io/kustomize/kyaml/kio/kioutil"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// Filters are the list of known filters for unmarshalling a filter into a concrete
// implementation.
var Filters = map[string]func() kio.Filter{
	"FileSetter":    func() kio.Filter { return &FileSetter{} },
	"FormatFilter":  func() kio.Filter { return &FormatFilter{} },
	"GrepFilter":    func() kio.Filter { return GrepFilter{} },
	"MatchModifier": func() kio.Filter { return &MatchModifyFilter{} },
	"Modifier":      func() kio.Filter { return &Modifier{} },
}

// filter wraps a kio.filter so that it can be unmarshalled from yaml.
type KFilter struct {
	kio.Filter
}

func (t KFilter) MarshalYAML() (interface{}, error) {
	return t.Filter, nil
}

func (t *KFilter) UnmarshalYAML(unmarshal func(interface{}) error) error {
	i := map[string]interface{}{}
	if err := unmarshal(i); err != nil {
		return err
	}
	meta := &yaml.ResourceMeta{}
	if err := unmarshal(meta); err != nil {
		return err
	}
	filter, found := Filters[meta.Kind]
	if !found {
		var knownFilters []string
		for k := range Filters {
			knownFilters = append(knownFilters, k)
		}
		sort.Strings(knownFilters)
		return fmt.Errorf("unsupported filter Kind %v:  may be one of: [%s]",
			meta, strings.Join(knownFilters, ","))
	}
	t.Filter = filter()

	return unmarshal(t.Filter)
}

// Modifier modifies the input Resources by invoking the provided pipeline.
// Modifier will return any Resources for which the pipeline does not return an error.
type Modifier struct {
	Kind string `yaml:"kind,omitempty"`

	Filters yaml.YFilters `yaml:"pipeline,omitempty"`
}

var _ kio.Filter = &Modifier{}

func (f Modifier) Filter(input []*yaml.RNode) ([]*yaml.RNode, error) {
	for i := range input {
		if _, err := input[i].Pipe(f.Filters.Filters()...); err != nil {
			return nil, err
		}
	}
	return input, nil
}

type MatchModifyFilter struct {
	Kind string `yaml:"kind,omitempty"`

	MatchFilters []yaml.YFilters `yaml:"match,omitempty"`

	ModifyFilters yaml.YFilters `yaml:"modify,omitempty"`
}

var _ kio.Filter = &MatchModifyFilter{}

func (f MatchModifyFilter) Filter(input []*yaml.RNode) ([]*yaml.RNode, error) {
	var matches = input
	var err error
	for _, filter := range f.MatchFilters {
		matches, err = MatchFilter{Filters: filter}.Filter(matches)
		if err != nil {
			return nil, err
		}
	}
	_, err = Modifier{Filters: f.ModifyFilters}.Filter(matches)
	if err != nil {
		return nil, err
	}
	return input, nil
}

type MatchFilter struct {
	Kind string `yaml:"kind,omitempty"`

	Filters yaml.YFilters `yaml:"pipeline,omitempty"`
}

var _ kio.Filter = &MatchFilter{}

func (f MatchFilter) Filter(input []*yaml.RNode) ([]*yaml.RNode, error) {
	var output []*yaml.RNode
	for i := range input {
		if v, err := input[i].Pipe(f.Filters.Filters()...); err != nil {
			return nil, err
		} else if v == nil {
			continue
		}
		output = append(output, input[i])
	}
	return output, nil
}

type FilenameFmtVerb string

const (
	// KindFmt substitutes kind
	KindFmt FilenameFmtVerb = "%k"

	// NameFmt substitutes metadata.name
	NameFmt FilenameFmtVerb = "%n"

	// NamespaceFmt substitutes metdata.namespace
	NamespaceFmt FilenameFmtVerb = "%s"
)

// FileSetter sets the file name and mode annotations on Resources.
type FileSetter struct {
	Kind string `yaml:"kind,omitempty"`

	// FilenamePattern is the pattern to use for generating filenames.  FilenameFmtVerb
	// FielnameFmtVerbs may be specified to substitute Resource metadata into the filename.
	FilenamePattern string `yaml:"filenamePattern,omitempty"`

	// Mode is the filemode to write.
	Mode string `yaml:"mode,omitempty"`

	// Override will override the existing filename if it is set on the pattern.
	// Otherwise the existing filename is kept.
	Override bool `yaml:"override,omitempty"`
}

var _ kio.Filter = &FileSetter{}

const DefaultFilenamePattern = "%n_%k.yaml"

func (f *FileSetter) Filter(input []*yaml.RNode) ([]*yaml.RNode, error) {
	if f.Mode == "" {
		f.Mode = fmt.Sprintf("%d", 0600)
	}
	if f.FilenamePattern == "" {
		f.FilenamePattern = DefaultFilenamePattern
	}

	resources := map[string][]*yaml.RNode{}
	for i := range input {
		if err := kioutil.CopyLegacyAnnotations(input[i]); err != nil {
			return nil, err
		}

		m, err := input[i].GetMeta()
		if err != nil {
			return nil, err
		}
		file := f.FilenamePattern
		file = strings.ReplaceAll(file, string(KindFmt), strings.ToLower(m.Kind))
		file = strings.ReplaceAll(file, string(NameFmt), strings.ToLower(m.Name))
		file = strings.ReplaceAll(file, string(NamespaceFmt), strings.ToLower(m.Namespace))

		if _, found := m.Annotations[kioutil.PathAnnotation]; !found || f.Override {
			if _, err := input[i].Pipe(yaml.SetAnnotation(kioutil.PathAnnotation, file)); err != nil {
				return nil, err
			}
			if _, err := input[i].Pipe(yaml.SetAnnotation(kioutil.LegacyPathAnnotation, file)); err != nil {
				return nil, err
			}
		}
		resources[file] = append(resources[file], input[i])
	}

	var output []*yaml.RNode
	for i := range resources {
		if err := kioutil.SortNodes(resources[i]); err != nil {
			return nil, err
		}
		for j := range resources[i] {
			if _, err := resources[i][j].Pipe(
				yaml.SetAnnotation(kioutil.IndexAnnotation, fmt.Sprintf("%d", j))); err != nil {
				return nil, err
			}
			if _, err := resources[i][j].Pipe(
				yaml.SetAnnotation(kioutil.LegacyIndexAnnotation, fmt.Sprintf("%d", j))); err != nil {
				return nil, err
			}
			output = append(output, resources[i][j])
		}
	}
	return output, nil
}
