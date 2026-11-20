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
	"fmt"
	"strings"
	"testing"

	"cel.dev/cel-go/cel"
	"cel.dev/cel-go/common/decls"
	"cel.dev/cel-go/common/types"
	"cel.dev/cel-go/ext"

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
		// Kubernetes <1.32>:
		"jsonpatch.escapeKey",
		// Kubernetes <1.33>:
		"semver", "isSemver", "major", "minor", "patch",
		// Kubernetes <1.37>:
		"includes",
		"isMask",
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
			if !strings.HasPrefix(lb.TypeName(), "kubernetes.") && !upstreamTypeNames.Has(lb.TypeName()) {
				t.Errorf("Expected all types in kubernetes CEL libraries to have a type name packaged with a 'kubernetes.' prefix but got %v", lb.TypeName())
			}
		}
		unregistered := usedTypes.Difference(registeredTypes)
		if len(unregistered) != 0 {
			t.Errorf("Expected types to be registered with the %s library Type() functions, but they were not: %v", lib.LibraryName(), unregistered)
		}
	}
}

// We have our own implementation of these, but they are also cel-go types
var upstreamTypeNames = sets.New[string]("net.IP", "net.CIDR")

// TestNetworkLibraryOverloadCompatibility verifies that the cel-go network library, which replaces
// the kubernetes IP and CIDR libraries in the base environment from Kubernetes 1.38, declares every
// overload of those libraries with the same overload ID and signature, so that stored expressions
// compiled with the kubernetes libraries still compile.
//
// It also tracks the overloads that the cel-go network library adds on top of the kubernetes
// libraries. The cel-go network library does not gate its functions on NetworkVersion, so a cel-go
// bump that adds a function makes it available in the base environment without a Kubernetes
// version gate. New functions must follow
// https://github.com/kubernetes/enhancements/tree/master/keps/sig-api-machinery/2876-crd-validation-expression-language#function-library-updates
// before they are listed below.
func TestNetworkLibraryOverloadCompatibility(t *testing.T) {
	kubernetesOverloads := overloadSignatures(t, IP(), CIDR(CIDRVersion(1)))
	celGoOverloads := overloadSignatures(t, ext.Network(ext.NetworkVersion(1)))

	for id, signature := range kubernetesOverloads {
		celGoSignature, ok := celGoOverloads[id]
		if !ok {
			t.Errorf("Expected overload %s %q of the kubernetes IP and CIDR libraries to be declared by the cel-go network library", id, signature)
			continue
		}
		if celGoSignature != signature {
			t.Errorf("Expected overload %s to have signature %q in the cel-go network library, matching the kubernetes IP and CIDR libraries, but got %q", id, signature, celGoSignature)
		}
	}

	// WARN: Overloads added by the cel-go network library on top of the kubernetes IP and CIDR
	// libraries must be tracked here along with the Kubernetes version that introduced them.
	expectedAdditions := map[string]string{
		// Kubernetes <1.38>:
	}
	for id, signature := range celGoOverloads {
		if _, ok := kubernetesOverloads[id]; ok {
			continue
		}
		expected, ok := expectedAdditions[id]
		switch {
		case !ok:
			t.Errorf("Unexpected overload %s %q added by the cel-go network library. It must be introduced in a new Kubernetes version and tracked here.", id, signature)
		case expected != signature:
			t.Errorf("Expected overload %s added by the cel-go network library to have signature %q, but got %q", id, expected, signature)
		}
	}
	for id, signature := range expectedAdditions {
		if _, ok := celGoOverloads[id]; !ok {
			t.Errorf("Expected overload %s %q to be added by the cel-go network library, but it was not found", id, signature)
		}
	}
}

// overloadSignatures returns the signatures of all function overloads declared by the given
// environment options, keyed by overload ID.
func overloadSignatures(t *testing.T, opts ...cel.EnvOption) map[string]string {
	t.Helper()
	env, err := cel.NewCustomEnv(opts...)
	if err != nil {
		t.Fatal(err)
	}
	signatures := map[string]string{}
	for name, fn := range env.Functions() {
		for _, o := range fn.OverloadDecls() {
			var args []string
			for _, at := range o.ArgTypes() {
				args = append(args, at.String())
			}
			signature := fmt.Sprintf("%s(%s) -> %s", name, strings.Join(args, ", "), o.ResultType())
			if o.IsMemberFunction() {
				signature = fmt.Sprintf("%s.%s(%s) -> %s", args[0], name, strings.Join(args[1:], ", "), o.ResultType())
			}
			signatures[o.ID()] = signature
		}
	}
	return signatures
}
