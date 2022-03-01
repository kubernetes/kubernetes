/*
Copyright 2022 The Kubernetes Authors.

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

package library

import (
	"testing"

	"github.com/google/cel-go/cel"
	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

func TestLibraryCompatibility(t *testing.T) {
	functionNames := map[string]struct{}{}

	decls := map[cel.Library][]*exprpb.Decl{
		urlsLib:  urlLibraryDecls,
		listsLib: listsLibraryDecls,
		regexLib: regexLibraryDecls,
	}
	if len(k8sExtensionLibs) != len(decls) {
		t.Errorf("Expected the same number of libraries in the ExtensionLibs as are tested for compatibility")
	}
	for _, l := range decls {
		for _, d := range l {
			functionNames[d.GetName()] = struct{}{}
		}
	}

	// WARN: All library changes must follow
	// https://github.com/kubernetes/enhancements/tree/master/keps/sig-api-machinery/2876-crd-validation-expression-language#function-library-updates
	// and must track the functions here along with which Kubernetes version introduced them.
	knownFunctions := []string{
		// Kubernetes 1.24:
		"isSorted", "sum", "max", "min", "indexOf", "lastIndexOf", "find", "findAll", "url", "getScheme", "getHost", "getHostname",
		"getPort", "getEscapedPath", "getQuery", "isURL",
		// Kubernetes <1.??>:
	}
	for _, fn := range knownFunctions {
		delete(functionNames, fn)
	}

	if len(functionNames) != 0 {
		t.Errorf("Expected all functions in the libraries to be assigned to a kubernetes release, but found the unassigned function names: %v", functionNames)
	}
}
