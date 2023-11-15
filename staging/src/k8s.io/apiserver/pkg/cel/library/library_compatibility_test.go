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

	"k8s.io/apimachinery/pkg/util/sets"
)

func TestLibraryCompatibility(t *testing.T) {
	var libs []map[string][]cel.FunctionOpt
	libs = append(libs, authzLibraryDecls, listsLibraryDecls, regexLibraryDecls, urlLibraryDecls, quantityLibraryDecls, ipLibraryDecls, cidrLibraryDecls)
	functionNames := sets.New[string]()
	for _, lib := range libs {
		for name := range lib {
			functionNames[name] = struct{}{}
		}
	}

	// WARN: All library changes must follow
	// https://github.com/kubernetes/enhancements/tree/master/keps/sig-api-machinery/2876-crd-validation-expression-language#function-library-updates
	// and must track the functions here along with which Kubernetes version introduced them.
	knownFunctions := sets.New(
		// Kubernetes 1.24:
		"isSorted", "sum", "max", "min", "indexOf", "lastIndexOf", "find", "findAll", "url", "getScheme", "getHost", "getHostname",
		"getPort", "getEscapedPath", "getQuery", "isURL",
		// Kubernetes <1.27>:
		"path", "group", "serviceAccount", "resource", "subresource", "namespace", "name", "check", "allowed", "reason",
		// Kubernetes <1.28>:
		"errored", "error",
		// Kubernetes <1.29>:
		"add", "asApproximateFloat", "asInteger", "compareTo", "isGreaterThan", "isInteger", "isLessThan", "isQuantity", "quantity", "sign", "sub",
		// Kubernetes <1.30>:
		"ip", "family", "isUnspecified", "isLoopback", "isLinkLocalMulticast", "isLinkLocalUnicast", "isGlobalUnicast", "ip.isCanonical", "isIP", "cidr", "containsIP", "containsCIDR", "masked", "prefixLength", "isCIDR", "string",
		// Kubernetes <1.??>:
	)

	// TODO: test celgo function lists

	unexpected := functionNames.Difference(knownFunctions)
	missing := knownFunctions.Difference(functionNames)

	if len(unexpected) != 0 {
		t.Errorf("Expected all functions in the libraries to be assigned to a kubernetes release, but found the unexpected function names: %v", unexpected)
	}
	if len(missing) != 0 {
		t.Errorf("Expected all functions in the libraries to be assigned to a kubernetes release, but found the missing function names: %v", missing)
	}
}
