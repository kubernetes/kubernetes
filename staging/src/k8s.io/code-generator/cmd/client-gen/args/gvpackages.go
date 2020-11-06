/*
Copyright 2017 The Kubernetes Authors.

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

package args

import (
	"bytes"
	"encoding/csv"
	"flag"
	"path"
	"sort"
	"strings"

	"k8s.io/code-generator/cmd/client-gen/generators/util"
	"k8s.io/code-generator/cmd/client-gen/types"
)

type inputBasePathValue struct {
	builder *groupVersionsBuilder
}

var _ flag.Value = &inputBasePathValue{}

func NewInputBasePathValue(builder *groupVersionsBuilder, def string) *inputBasePathValue {
	v := &inputBasePathValue{
		builder: builder,
	}
	v.Set(def)
	return v
}

func (s *inputBasePathValue) Set(val string) error {
	s.builder.importBasePath = val
	return s.builder.update()
}

func (s *inputBasePathValue) Type() string {
	return "string"
}

func (s *inputBasePathValue) String() string {
	return s.builder.importBasePath
}

type gvPackagesValue struct {
	builder *groupVersionsBuilder
	groups  []string
	changed bool
}

func NewGVPackagesValue(builder *groupVersionsBuilder, def []string) *gvPackagesValue {
	gvp := new(gvPackagesValue)
	gvp.builder = builder
	if def != nil {
		if err := gvp.set(def); err != nil {
			panic(err)
		}
	}
	return gvp
}

var _ flag.Value = &gvPackagesValue{}

func (s *gvPackagesValue) set(vs []string) error {
	if s.changed {
		s.groups = append(s.groups, vs...)
	} else {
		s.groups = append([]string(nil), vs...)
	}

	s.builder.groups = s.groups
	return s.builder.update()
}

func (s *gvPackagesValue) Set(val string) error {
	vs, err := readAsCSV(val)
	if err != nil {
		return err
	}
	if err := s.set(vs); err != nil {
		return err
	}
	s.changed = true
	return nil
}

func (s *gvPackagesValue) Type() string {
	return "stringSlice"
}

func (s *gvPackagesValue) String() string {
	str, _ := writeAsCSV(s.groups)
	return "[" + str + "]"
}

type groupVersionsBuilder struct {
	value          *[]types.GroupVersions
	groups         []string
	importBasePath string
}

func NewGroupVersionsBuilder(groups *[]types.GroupVersions) *groupVersionsBuilder {
	return &groupVersionsBuilder{
		value: groups,
	}
}

func (p *groupVersionsBuilder) update() error {
	var seenGroups = make(map[types.Group]*types.GroupVersions)
	for _, v := range p.groups {
		pth, gvString := util.ParsePathGroupVersion(v)
		gv, err := types.ToGroupVersion(gvString)
		if err != nil {
			return err
		}

		versionPkg := types.PackageVersion{Package: path.Join(p.importBasePath, pth, gv.Group.NonEmpty(), gv.Version.String()), Version: gv.Version}
		if group, ok := seenGroups[gv.Group]; ok {
			seenGroups[gv.Group].Versions = append(group.Versions, versionPkg)
		} else {
			seenGroups[gv.Group] = &types.GroupVersions{
				PackageName: gv.Group.NonEmpty(),
				Group:       gv.Group,
				Versions:    []types.PackageVersion{versionPkg},
			}
		}
	}

	var groupNames []string
	for groupName := range seenGroups {
		groupNames = append(groupNames, groupName.String())
	}
	sort.Strings(groupNames)
	*p.value = []types.GroupVersions{}
	for _, groupName := range groupNames {
		*p.value = append(*p.value, *seenGroups[types.Group(groupName)])
	}

	return nil
}

func readAsCSV(val string) ([]string, error) {
	if val == "" {
		return []string{}, nil
	}
	stringReader := strings.NewReader(val)
	csvReader := csv.NewReader(stringReader)
	return csvReader.Read()
}

func writeAsCSV(vals []string) (string, error) {
	b := &bytes.Buffer{}
	w := csv.NewWriter(b)
	err := w.Write(vals)
	if err != nil {
		return "", err
	}
	w.Flush()
	return strings.TrimSuffix(b.String(), "\n"), nil
}
