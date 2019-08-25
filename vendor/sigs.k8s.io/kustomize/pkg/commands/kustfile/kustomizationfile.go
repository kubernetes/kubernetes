/*
Copyright 2018 The Kubernetes Authors.

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

package kustfile

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"log"
	"reflect"
	"regexp"
	"strings"

	"sigs.k8s.io/kustomize/pkg/fs"
	"sigs.k8s.io/kustomize/pkg/pgmconfig"
	"sigs.k8s.io/kustomize/pkg/types"
	"sigs.k8s.io/yaml"
)

var fieldMarshallingOrder = determineFieldOrder()

// determineFieldOrder returns a slice of Kustomization field
// names in the preferred order for serialization to a file.
// The field list is checked against the actual struct type
// to confirm that all fields are present, and no unknown
// fields are specified. Deprecated fields are removed from
// the list, meaning they will drop to the bottom on output
// (if present). The ordering and/or deprecation of fields
// in nested structs is not determined or considered.
func determineFieldOrder() []string {
	m := make(map[string]bool)
	s := reflect.ValueOf(&types.Kustomization{}).Elem()
	typeOfT := s.Type()
	for i := 0; i < s.NumField(); i++ {
		m[string(typeOfT.Field(i).Name)] = false
	}

	ordered := []string{
		"Resources",
		"Bases",
		"NamePrefix",
		"NameSuffix",
		"Namespace",
		"Crds",
		"CommonLabels",
		"CommonAnnotations",
		"PatchesStrategicMerge",
		"PatchesJson6902",
		"ConfigMapGenerator",
		"SecretGenerator",
		"GeneratorOptions",
		"Vars",
		"Images",
		"Configurations",
		"Generators",
		"Transformers",
		"Inventory",
	}

	// Add deprecated fields here.
	deprecated := map[string]bool{
		"Patches": true,
	}

	// Account for the inlined TypeMeta fields.
	var result []string
	result = append(result, "APIVersion", "Kind")
	m["TypeMeta"] = true

	// Make sure all these fields are recognized.
	for _, n := range ordered {
		if _, ok := m[n]; ok {
			m[n] = true
		} else {
			log.Fatalf("%s is not a recognized field.", n)
		}
		// Keep if not deprecated.
		if _, f := deprecated[n]; !f {
			result = append(result, n)
		}
	}
	return result
}

// commentedField records the comment associated with a kustomization field
// field has to be a recognized kustomization field
// comment can be empty
type commentedField struct {
	field   string
	comment []byte
}

func (cf *commentedField) appendComment(comment []byte) {
	cf.comment = append(cf.comment, comment...)
}

func squash(x [][]byte) []byte {
	return bytes.Join(x, []byte(``))
}

type kustomizationFile struct {
	path           string
	fSys           fs.FileSystem
	originalFields []*commentedField
}

// NewKustomizationFile returns a new instance.
func NewKustomizationFile(fSys fs.FileSystem) (*kustomizationFile, error) { // nolint
	mf := &kustomizationFile{fSys: fSys}
	err := mf.validate()
	if err != nil {
		return nil, err
	}
	return mf, nil
}

func (mf *kustomizationFile) validate() error {
	match := 0
	var path []string
	for _, kfilename := range pgmconfig.KustomizationFileNames {
		if mf.fSys.Exists(kfilename) {
			match += 1
			path = append(path, kfilename)
		}
	}

	switch match {
	case 0:
		return fmt.Errorf("Missing kustomization file '%s'.\n", pgmconfig.KustomizationFileNames[0])
	case 1:
		mf.path = path[0]
	default:
		return fmt.Errorf("Found multiple kustomization file: %v\n", path)
	}

	if mf.fSys.IsDir(mf.path) {
		return fmt.Errorf("%s should be a file", mf.path)
	}
	return nil
}

func (mf *kustomizationFile) Read() (*types.Kustomization, error) {
	data, err := mf.fSys.ReadFile(mf.path)
	if err != nil {
		return nil, err
	}
	data = types.FixKustomizationPreUnmarshalling(data)
	var k types.Kustomization
	err = yaml.Unmarshal(data, &k)
	if err != nil {
		return nil, err
	}
	k.FixKustomizationPostUnmarshalling()
	err = mf.parseCommentedFields(data)
	if err != nil {
		return nil, err
	}
	return &k, err
}

func (mf *kustomizationFile) Write(kustomization *types.Kustomization) error {
	if kustomization == nil {
		return errors.New("util: kustomization file arg is nil")
	}
	data, err := mf.marshal(kustomization)
	if err != nil {
		return err
	}
	return mf.fSys.WriteFile(mf.path, data)
}

// StringInSlice returns true if the string is in the slice.
func StringInSlice(str string, list []string) bool {
	for _, v := range list {
		if v == str {
			return true
		}
	}
	return false
}

func (mf *kustomizationFile) parseCommentedFields(content []byte) error {
	buffer := bytes.NewBuffer(content)
	var comments [][]byte

	line, err := buffer.ReadBytes('\n')
	for err == nil {
		if isCommentOrBlankLine(line) {
			comments = append(comments, line)
		} else {
			matched, field := findMatchedField(line)
			if matched {
				mf.originalFields = append(mf.originalFields, &commentedField{field: field, comment: squash(comments)})
				comments = [][]byte{}
			} else if len(comments) > 0 {
				mf.originalFields[len(mf.originalFields)-1].appendComment(squash(comments))
				comments = [][]byte{}
			}
		}
		line, err = buffer.ReadBytes('\n')
	}

	if err != io.EOF {
		return err
	}
	return nil
}

// marshal converts a kustomization to a byte stream.
func (mf *kustomizationFile) marshal(kustomization *types.Kustomization) ([]byte, error) {
	var output []byte
	for _, comment := range mf.originalFields {
		output = append(output, comment.comment...)
		content, err := marshalField(comment.field, kustomization)
		if err != nil {
			return content, err
		}
		output = append(output, content...)
	}
	for _, field := range fieldMarshallingOrder {
		if mf.hasField(field) {
			continue
		}
		content, err := marshalField(field, kustomization)
		if err != nil {
			return content, nil
		}
		output = append(output, content...)

	}
	return output, nil
}

func (mf *kustomizationFile) hasField(name string) bool {
	for _, n := range mf.originalFields {
		if n.field == name {
			return true
		}
	}
	return false
}

/*
 isCommentOrBlankLine determines if a line is a comment or blank line
 Return true for following lines
 # This line is a comment
       # This line is also a comment with several leading white spaces

 (The line above is a blank line)
*/
func isCommentOrBlankLine(line []byte) bool {
	s := bytes.TrimRight(bytes.TrimLeft(line, " "), "\n")
	return len(s) == 0 || bytes.HasPrefix(s, []byte(`#`))
}

func findMatchedField(line []byte) (bool, string) {
	for _, field := range fieldMarshallingOrder {
		// (?i) is for case insensitive regexp matching
		r := regexp.MustCompile("^(" + "(?i)" + field + "):")
		if r.Match(line) {
			return true, field
		}
	}
	return false, ""
}

// marshalField marshal a given field of a kustomization object into yaml format.
// If the field wasn't in the original kustomization.yaml file or wasn't added,
// an empty []byte is returned.
func marshalField(field string, kustomization *types.Kustomization) ([]byte, error) {
	r := reflect.ValueOf(*kustomization)
	v := r.FieldByName(strings.Title(field))

	if !v.IsValid() || isEmpty(v) {
		return []byte{}, nil
	}

	k := &types.Kustomization{}
	kr := reflect.ValueOf(k)
	kv := kr.Elem().FieldByName(strings.Title(field))
	kv.Set(v)

	return yaml.Marshal(k)
}

func isEmpty(v reflect.Value) bool {
	// If v is a pointer type
	if v.Type().Kind() == reflect.Ptr {
		return v.IsNil()
	}
	return v.Len() == 0
}
