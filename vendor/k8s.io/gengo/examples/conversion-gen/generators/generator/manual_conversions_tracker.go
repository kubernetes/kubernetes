/*
Copyright 2019 The Kubernetes Authors.

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

package generator

import (
	"bytes"
	"fmt"
	"strings"

	"k8s.io/klog"

	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"
)

// a ManualConversionsTracker keeps track of manually defined conversion functions.
type ManualConversionsTracker struct {
	// see the explanation on NewManualConversionsTracker.
	additionalConversionArguments []NamedVariable

	// processedPackages keeps track of which packages have already been processed, as there
	// is no need to ever process the same package twice.
	processedPackages map[string][]error

	// conversionFunctions keeps track of the manual function definitions known to this tracker.
	conversionFunctions map[ConversionPair]*types.Type

	// see conversionFunctionName
	buffer          *bytes.Buffer
	conversionNamer *namer.NameStrategy
}

// NewManualConversionsTracker builds a new ManualConversionsTracker.
// Additional conversion arguments allow users to set which arguments should be part of
// a conversion function signature.
// When generating conversion code, those will be added to the signature of each conversion function,
// and then passed down to conversion functions for embedded types. This allows to generate
// conversion code with additional argument, eg
//    Convert_a_X_To_b_Y(in *a.X, out *b.Y, s conversion.Scope) error
// Manually defined conversion functions will also be expected to have similar signatures.
func NewManualConversionsTracker(additionalConversionArguments ...NamedVariable) *ManualConversionsTracker {
	return &ManualConversionsTracker{
		additionalConversionArguments: additionalConversionArguments,
		processedPackages:             make(map[string][]error),
		conversionFunctions:           make(map[ConversionPair]*types.Type),
		buffer:                        &bytes.Buffer{},
		conversionNamer:               ConversionNamer(),
	}
}

var errorName = types.Ref("", "error").Name

// findManualConversionFunctions looks for conversion functions in the given package.
func (t *ManualConversionsTracker) findManualConversionFunctions(context *generator.Context, packagePath string) (errors []error) {
	if e, present := t.processedPackages[packagePath]; present {
		// already processed
		return e
	}

	pkg, err := context.AddDirectory(packagePath)
	if err != nil {
		return []error{fmt.Errorf("unable to add directory %q to context: %v", packagePath, err)}
	}
	if pkg == nil {
		klog.Warningf("Skipping nil package passed to getManualConversionFunctions")
		return
	}
	klog.V(5).Infof("Scanning for conversion functions in %v", pkg.Path)

	for _, function := range pkg.Functions {
		if function.Underlying == nil || function.Underlying.Kind != types.Func {
			errors = append(errors, fmt.Errorf("malformed function: %#v", function))
			continue
		}
		if function.Underlying.Signature == nil {
			errors = append(errors, fmt.Errorf("function without signature: %#v", function))
			continue
		}

		klog.V(8).Infof("Considering function %s", function.Name)

		isConversionFunc, inType, outType := t.isConversionFunction(function)
		if !isConversionFunc {
			if strings.HasPrefix(function.Name.Name, conversionFunctionPrefix) {
				errors = append(errors, fmt.Errorf("function %s %s does not match expected conversion signature",
					function.Name.Package, function.Name.Name))
			}
			continue
		}

		// it is a conversion function
		key := ConversionPair{inType.Elem, outType.Elem}
		if previousConversionFunc, present := t.conversionFunctions[key]; present {
			errors = append(errors, fmt.Errorf("duplicate static conversion defined: %s -> %s from:\n%s.%s\n%s.%s",
				inType, outType, previousConversionFunc.Name.Package, previousConversionFunc.Name.Name, function.Name.Package, function.Name.Name))
			continue
		}
		t.conversionFunctions[key] = function
	}

	t.processedPackages[packagePath] = errors
	return
}

// isConversionFunction returns true iff the given function is a conversion function; that is of the form
// func Convert_a_X_To_b_Y(in *a.X, out *b.Y, additionalConversionArguments...) error
// If it is a signature functions, also returns the inType and outType.
func (t *ManualConversionsTracker) isConversionFunction(function *types.Type) (bool, *types.Type, *types.Type) {
	signature := function.Underlying.Signature

	if signature.Receiver != nil {
		klog.V(8).Infof("%s has a receiver", function.Name)
		return false, nil, nil
	}
	if len(signature.Results) != 1 || signature.Results[0].Name != errorName {
		klog.V(8).Infof("%s has wrong results", function.Name)
		return false, nil, nil
	}
	// 2 (in and out) + additionalConversionArguments
	if len(signature.Parameters) != 2+len(t.additionalConversionArguments) {
		klog.V(8).Infof("%s has wrong number of parameters", function.Name)
		return false, nil, nil
	}
	inType := signature.Parameters[0]
	outType := signature.Parameters[1]
	if inType.Kind != types.Pointer || outType.Kind != types.Pointer {
		klog.V(8).Infof("%s does not have pointers parameters for in/out", function.Name)
		return false, nil, nil
	}
	for i, extraArg := range t.additionalConversionArguments {
		if signature.Parameters[i+2].Name != extraArg.Type.Name {
			klog.V(8).Infof("%s's %d-th parameter has wrong type: %q VS %q",
				function.Name, i+2, signature.Parameters[i+2].Name, extraArg.Type.Name)
			return false, nil, nil
		}
	}

	// check it satisfies the naming convention
	if function.Name.Name != t.conversionFunctionName(inType.Elem, outType.Elem) {
		return false, nil, nil
	}

	return true, inType, outType
}

func (t *ManualConversionsTracker) preexists(inType, outType *types.Type) (*types.Type, bool) {
	function, ok := t.conversionFunctions[ConversionPair{inType, outType}]
	return function, ok
}

// conversionFunctionName returns the name of the conversion function for in to out.
func (t *ManualConversionsTracker) conversionFunctionName(in, out *types.Type) string {
	return conversionFunctionName(in, out, t.conversionNamer, t.buffer)
}
