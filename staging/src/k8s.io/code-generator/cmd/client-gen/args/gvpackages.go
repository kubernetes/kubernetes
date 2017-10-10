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
	"strings"

	"path/filepath"
	"sort"

	"k8s.io/code-generator/cmd/client-gen/types"
)

type gvPackagesValue struct {
	gvToPath *map[types.GroupVersion]string
	groups   *[]types.GroupVersions
	changed  bool
}

func NewGVPackagesValue(gvToPath *map[types.GroupVersion]string, groups *[]types.GroupVersions, def []string) *gvPackagesValue {
	gvp := new(gvPackagesValue)
	gvp.gvToPath = gvToPath
	gvp.groups = groups
	if def != nil {
		if err := gvp.set(def); err != nil {
			panic(err)
		}
	}
	return gvp
}

var _ flag.Value = &gvPackagesValue{}

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

func (s *gvPackagesValue) set(vs []string) error {
	if !s.changed {
		*s.gvToPath = map[types.GroupVersion]string{}
		*s.groups = []types.GroupVersions{}
	}

	var seenGroups = make(map[types.Group]*types.GroupVersions)
	for _, g := range *s.groups {
		seenGroups[g.Group] = &g
	}

	for _, v := range vs {
		pth, gvString := parsePathGroupVersion(v)
		gv, err := types.ToGroupVersion(gvString)
		if err != nil {
			return err
		}

		if group, ok := seenGroups[gv.Group]; ok {
			seenGroups[gv.Group].Versions = append(group.Versions, gv.Version)
		} else {
			seenGroups[gv.Group] = &types.GroupVersions{
				PackageName: gv.Group.NonEmpty(),
				Group:       gv.Group,
				Versions:    []types.Version{gv.Version},
			}
		}

		(*s.gvToPath)[gv] = groupVersionPath(pth, gv.Group.String(), gv.Version.String())
	}

	var groupNames []string
	for groupName := range seenGroups {
		groupNames = append(groupNames, groupName.String())
	}
	sort.Strings(groupNames)
	*s.groups = []types.GroupVersions{}
	for _, groupName := range groupNames {
		*s.groups = append(*s.groups, *seenGroups[types.Group(groupName)])
	}

	return nil
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
	strs := make([]string, 0, len(*s.gvToPath))
	for gv, pth := range *s.gvToPath {
		strs = append(strs, path.Join(pth, gv.Group.String(), gv.Version.String()))
	}
	str, _ := writeAsCSV(strs)
	return "[" + str + "]"
}

func parsePathGroupVersion(pgvString string) (gvPath string, gvString string) {
	subs := strings.Split(pgvString, "/")
	length := len(subs)
	switch length {
	case 0, 1, 2:
		return "", pgvString
	default:
		return strings.Join(subs[:length-2], "/"), strings.Join(subs[length-2:], "/")
	}
}

func groupVersionPath(gvPath string, group string, version string) (path string) {
	// special case for the core group
	if group == "api" {
		path = filepath.Join("core", version)
	} else {
		path = filepath.Join(gvPath, group, version)
	}
	return
}
