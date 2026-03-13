/*
Copyright 2021 The Kubernetes Authors.

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
	"fmt"
	"strings"

	"k8s.io/gengo/v2/types"
)

type externalApplyConfigurationValue struct {
	externals *map[types.Name]string
}

func NewExternalApplyConfigurationValue(externals *map[types.Name]string, def []string) *externalApplyConfigurationValue {
	val := new(externalApplyConfigurationValue)
	val.externals = externals
	if def != nil {
		if err := val.set(def); err != nil {
			panic(err)
		}
	}
	return val
}

var _ flag.Value = &externalApplyConfigurationValue{}

func (s *externalApplyConfigurationValue) set(vs []string) error {
	for _, input := range vs {
		typ, pkg, err := parseExternalMapping(input)
		if err != nil {
			return err
		}
		if _, ok := (*s.externals)[typ]; ok {
			return fmt.Errorf("duplicate type found in --external-applyconfigurations: %v", typ)
		}
		(*s.externals)[typ] = pkg
	}

	return nil
}

func (s *externalApplyConfigurationValue) Set(val string) error {
	vs, err := readAsCSV(val)
	if err != nil {
		return err
	}
	if err := s.set(vs); err != nil {
		return err
	}

	return nil
}

func (s *externalApplyConfigurationValue) Type() string {
	return "string"
}

func (s *externalApplyConfigurationValue) String() string {
	var strs []string
	for k, v := range *s.externals {
		strs = append(strs, fmt.Sprintf("%s.%s:%s", k.Package, k.Name, v))
	}
	str, _ := writeAsCSV(strs)
	return "[" + str + "]"
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

func parseExternalMapping(mapping string) (typ types.Name, pkg string, err error) {
	parts := strings.Split(mapping, ":")
	if len(parts) != 2 {
		return types.Name{}, "", fmt.Errorf("expected string of the form <package>.<typeName>:<applyconfiguration-package> but got %s", mapping)
	}
	packageTypeStr := parts[0]
	pkg = parts[1]
	// need to split on the *last* dot, since k8s.io (and other valid packages) have a dot in it
	lastDot := strings.LastIndex(packageTypeStr, ".")
	if lastDot == -1 || lastDot == len(packageTypeStr)-1 {
		return types.Name{}, "", fmt.Errorf("expected package and type of the form <package>.<typeName> but got %s", packageTypeStr)
	}
	structPkg := packageTypeStr[:lastDot]
	structType := packageTypeStr[lastDot+1:]

	return types.Name{Package: structPkg, Name: structType}, pkg, nil
}
