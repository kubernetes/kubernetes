/*
Copyright 2026 The Kubernetes Authors.

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

package main

import (
	"go/parser"
	"go/token"
)

var atomicFieldFunctions = []functionSpec{
	{
		name: "doList", leanName: "listProgram", receiver: "*toFieldSetWalker", receiverID: "v",
		signature:  "func (v *toFieldSetWalker) doList(t *schema.List) (errs ValidationErrors)",
		parameters: []string{"t"}, namedResults: []string{"errs"}, atomicFields: true,
	},
	{
		name: "doMap", leanName: "mapProgram", receiver: "*toFieldSetWalker", receiverID: "v",
		signature:  "func (v *toFieldSetWalker) doMap(t *schema.Map) (errs ValidationErrors)",
		parameters: []string{"t"}, namedResults: []string{"errs"}, atomicFields: true,
	},
}

func validateAtomicFieldImports(source []byte) error {
	file, err := parser.ParseFile(token.NewFileSet(), "tofieldset.go", source, parser.ImportsOnly)
	if err != nil {
		return err
	}
	expected := map[string]string{}
	for _, name := range []string{"fieldpath", "schema", "value"} {
		expected[name] = "sigs.k8s.io/structured-merge-diff/v6/" + name
	}
	return validateImportBindings(file.Imports, expected)
}
