/**
 *  Copyright 2014 Paul Querna
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package ffjsoninception

import (
	"errors"
	"fmt"
	"github.com/pquerna/ffjson/shared"
	"io/ioutil"
	"os"
	"reflect"
	"sort"
)

type Inception struct {
	objs          []*StructInfo
	InputPath     string
	OutputPath    string
	PackageName   string
	PackagePath   string
	OutputImports map[string]bool
	OutputFuncs   []string
	q             ConditionalWrite
}

func NewInception(inputPath string, packageName string, outputPath string) *Inception {
	return &Inception{
		objs:          make([]*StructInfo, 0),
		InputPath:     inputPath,
		OutputPath:    outputPath,
		PackageName:   packageName,
		OutputFuncs:   make([]string, 0),
		OutputImports: make(map[string]bool),
	}
}

func (i *Inception) AddMany(objs []shared.InceptionType) {
	for _, obj := range objs {
		i.Add(obj)
	}
}

func (i *Inception) Add(obj shared.InceptionType) {
	i.objs = append(i.objs, NewStructInfo(obj))
	i.PackagePath = i.objs[0].Typ.PkgPath()
}

func (i *Inception) wantUnmarshal(si *StructInfo) bool {
	if si.Options.SkipDecoder {
		return false
	}
	typ := si.Typ
	umlx := typ.Implements(unmarshalFasterType) || reflect.PtrTo(typ).Implements(unmarshalFasterType)
	umlstd := typ.Implements(unmarshalerType) || reflect.PtrTo(typ).Implements(unmarshalerType)
	if umlstd && !umlx {
		// structure has UnmarshalJSON, but not our faster version -- skip it.
		return false
	}
	return true
}

func (i *Inception) wantMarshal(si *StructInfo) bool {
	if si.Options.SkipEncoder {
		return false
	}
	typ := si.Typ
	mlx := typ.Implements(marshalerFasterType) || reflect.PtrTo(typ).Implements(marshalerFasterType)
	mlstd := typ.Implements(marshalerType) || reflect.PtrTo(typ).Implements(marshalerType)
	if mlstd && !mlx {
		// structure has MarshalJSON, but not our faster version -- skip it.
		return false
	}
	return true
}

type sortedStructs []*StructInfo

func (p sortedStructs) Len() int           { return len(p) }
func (p sortedStructs) Less(i, j int) bool { return p[i].Name < p[j].Name }
func (p sortedStructs) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
func (p sortedStructs) Sort()              { sort.Sort(p) }

func (i *Inception) generateCode() error {
	// We sort the structs by name, so output if predictable.
	sorted := sortedStructs(i.objs)
	sorted.Sort()

	for _, si := range sorted {
		if i.wantMarshal(si) {
			err := CreateMarshalJSON(i, si)
			if err != nil {
				return err
			}
		}

		if i.wantUnmarshal(si) {
			err := CreateUnmarshalJSON(i, si)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func (i *Inception) handleError(err error) {
	fmt.Fprintf(os.Stderr, "Error: %s:\n\n", err)
	os.Exit(1)
}

func (i *Inception) Execute() {
	if len(os.Args) != 1 {
		i.handleError(errors.New(fmt.Sprintf("Internal ffjson error: inception executable takes no args: %v", os.Args)))
		return
	}

	err := i.generateCode()
	if err != nil {
		i.handleError(err)
		return
	}

	data, err := RenderTemplate(i)
	if err != nil {
		i.handleError(err)
		return
	}

	stat, err := os.Stat(i.InputPath)

	if err != nil {
		i.handleError(err)
		return
	}

	err = ioutil.WriteFile(i.OutputPath, data, stat.Mode())

	if err != nil {
		i.handleError(err)
		return
	}

}
