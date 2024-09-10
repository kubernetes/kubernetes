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
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/decls"
	"github.com/google/cel-go/common/types"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
)

func TestLibraryCompatibility(t *testing.T) {
	functionNames := sets.New[string]()
	for _, lib := range KnownLibraries() {
		if !strings.HasPrefix(lib.LibraryName(), "kubernetes.") {
			t.Errorf("Expected all kubernetes CEL libraries to have a name package with a 'kubernetes.' prefix but got %v", lib.LibraryName())
		}
		for name := range lib.declarations() {
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
		// Kubernetes <1.31>:
		"fieldSelector", "labelSelector", "validate", "format.named", "isSemver", "major", "minor", "patch", "semver",
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

// TestTypeRegistration ensures that all custom types defined and used by Kubernetes CEL libraries
// are returned by library.Types().  Other tests depend on Types() to provide an up-to-date list of
// types declared in a library.
func TestTypeRegistration(t *testing.T) {
	for _, lib := range KnownLibraries() {
		registeredTypes := sets.New[*cel.Type]()
		usedTypes := sets.New[*cel.Type]()
		// scan all registered function declarations for the library
		for _, fn := range lib.declarations() {
			fn, err := decls.NewFunction("placeholder-not-used", fn...)
			if err != nil {
				t.Fatal(err)
			}
			for _, o := range fn.OverloadDecls() {
				// ArgTypes include both the receiver type (if present) and
				// all function argument types.
				for _, at := range o.ArgTypes() {
					switch at.Kind() {
					// User defined types are either Opaque or Struct.
					case types.OpaqueKind, types.StructKind:
						usedTypes.Insert(at)
					default:
						// skip
					}
				}
			}
		}
		for _, lb := range lib.Types() {
			registeredTypes.Insert(lb)
			if !strings.HasPrefix(lb.TypeName(), "kubernetes.") && !legacyTypeNames.Has(lb.TypeName()) {
				t.Errorf("Expected all types in kubernetes CEL libraries to have a type name packaged with a 'kubernetes.' prefix but got %v", lb.TypeName())
			}
		}
		unregistered := usedTypes.Difference(registeredTypes)
		if len(unregistered) != 0 {
			t.Errorf("Expected types to be registered with the %s library Type() functions, but they were not: %v", lib.LibraryName(), unregistered)
		}
	}
}

// TODO: Consider renaming these to "kubernetes.net.IP" and "kubernetes.net.CIDR" if we decide not to promote them to cel-go
var legacyTypeNames = sets.New[string]("net.IP", "net.CIDR")
