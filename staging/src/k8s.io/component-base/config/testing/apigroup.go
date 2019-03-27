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

package testing

import (
	"fmt"
	"reflect"
	"regexp"
	"strings"

	apinamingtest "k8s.io/apimachinery/pkg/api/apitesting/naming"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
)

// APIVersionRegexp is used to match external APIVersions (ex: v1, v1alpha2, v1beta2)
var APIVersionRegexp = regexp.MustCompile(`^v\d+((alpha|beta){1}\d+)?$`)

// ComponentConfigPackage details the dependencies and feature flags for testing a ComponentConfig API
type ComponentConfigPackage struct {
	ComponentName               string
	GroupName                   string
	SchemeGroupVersion          schema.GroupVersion
	AddToScheme                 func(*runtime.Scheme) error
	SkipTests                   sets.String
	AllowedNoJSONTags           map[reflect.Type]sets.String
	AllowedNonstandardJSONNames map[reflect.Type]sets.String
}

type testingFunc func(*runtime.Scheme, *ComponentConfigPackage) error

// These constants may be used to specify test functions to skip for ComponentConfigPackage.SkipTests
const (
	// default for both internal and external
	VerifyTagNaming                 = "VerifyTagNaming"
	VerifyGroupNameSuffix           = "VerifyGroupNameSuffix"
	VerifyGroupNameMatch            = "VerifyGroupNameMatch"
	VerifyCorrectGroupName          = "VerifyCorrectGroupName"
	VerifyComponentConfigKindExists = "VerifyComponentConfigKindExists"
	VerifySchemeBuilderInit         = "VerifySchemeBuilderInit"
	VerifyTypeRegistration          = "VerifyTypeRegistration"
	// non-default
	VerifyExternalAPIVersion = "VerifyExternalAPIVersion"
	VerifyInternalAPIVersion = "VerifyInternalAPIVersion"
)

// This is the default collection of test functions to run, regardless of whether we are testing the internal or external API
var testingFuncs = map[string]testingFunc{
	VerifyTagNaming:                 verifyTagNaming,
	VerifyGroupNameSuffix:           verifyGroupNameSuffix,
	VerifyGroupNameMatch:            verifyGroupNameMatch,
	VerifyCorrectGroupName:          verifyCorrectGroupName,
	VerifyComponentConfigKindExists: verifyComponentConfigKindExists,
	VerifySchemeBuilderInit:         verifySchemeBuilderInit,
	VerifyTypeRegistration:          verifyTypeRegistration,
}

// VerifyExternalTypePackage validates the ComponentConfigPackage and runs all of the default testingFuncs.
// Additionally, it also runs VerifyExternalAPIVersion.
func VerifyExternalTypePackage(pkginfo *ComponentConfigPackage) error {
	// Test external tag naming (json name should match Go name)
	// Test GroupName has the k8s.io suffix
	// Test GroupName == SchemeGroupVersion.GroupName
	// Test GroupName is named correctly (based on ComponentName)
	// Test there is a {Component}Configuration kind in the scheme
	// Test that the SchemeBuilder contains exactly one init func: addKnownTypes
	// Test that addKnownTypes and AddToScheme registers at least one type and doesn't error

	scheme, err := setup(pkginfo)
	if err != nil {
		return fmt.Errorf("test setup error: %v", err)
	}

	// Test that the API version follows the right pattern and isn't internal
	extraFns := map[string]testingFunc{
		VerifyExternalAPIVersion: verifyExternalAPIVersion,
	}

	return runFuncs(scheme, pkginfo, extraFns)
}

// VerifyInternalTypePackage validates the ComponentConfigPackage and runs all of the default testingFuncs.
// Additionally, it also runs VerifyInternalAPIVersion.
func VerifyInternalTypePackage(pkginfo *ComponentConfigPackage) error {
	// Test internal tag naming (no tags allowed)
	// Test GroupName has the k8s.io suffix
	// Test GroupName == SchemeGroupVersion.GroupName
	// Test GroupName is named correctly (based on ComponentName)
	// Test there is a {Component}Configuration kind in the scheme
	// Test that the SchemeBuilder contains exactly one init func: addKnownTypes
	// Test that addKnownTypes and AddToScheme registers at least one type and doesn't error

	scheme, err := setup(pkginfo)
	if err != nil {
		return fmt.Errorf("test setup error: %v", err)
	}

	// Test that API version should be internal
	extraFns := map[string]testingFunc{
		VerifyInternalAPIVersion: verifyInternalAPIVersion,
	}

	return runFuncs(scheme, pkginfo, extraFns)
}

// setup validates that all necessary inputs are present in pkginfo and that the pkginfo can be added to a scheme.
// On success, a new scheme mutated by pkginfo.AddToScheme is returned for use with the testing functions.
func setup(pkginfo *ComponentConfigPackage) (*runtime.Scheme, error) {
	if len(pkginfo.ComponentName) == 0 ||
		len(pkginfo.GroupName) == 0 ||
		pkginfo.SchemeGroupVersion.Empty() ||
		pkginfo.AddToScheme == nil {
		return nil, fmt.Errorf("invalid argument: not all parameters were passed correctly to the function")
	}

	scheme := runtime.NewScheme()
	if err := pkginfo.AddToScheme(scheme); err != nil {
		return nil, fmt.Errorf("AddToScheme must not return an error")
	}
	return scheme, nil
}

// runFuncs runs all of the default testFuncs and non-default extraFuncs.
// Test functions whose keys are contained in pkginfo.SkipTests are skipped.
// All functions are guaranteed to run (regardless of previous error conditions), and all errors are aggregated.
func runFuncs(scheme *runtime.Scheme, pkginfo *ComponentConfigPackage, extraFns map[string]testingFunc) error {
	verifyFns := []testingFunc{}
	for name, fn := range testingFuncs {
		if pkginfo.SkipTests.Has(name) {
			continue
		}
		verifyFns = append(verifyFns, fn)
	}
	for name, fn := range extraFns {
		if pkginfo.SkipTests.Has(name) {
			continue
		}
		verifyFns = append(verifyFns, fn)
	}
	errs := []error{}
	for _, fn := range verifyFns {
		if err := fn(scheme, pkginfo); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.NewAggregate(errs)
}

func verifyTagNaming(scheme *runtime.Scheme, pkginfo *ComponentConfigPackage) error {
	return apinamingtest.VerifyTagNaming(scheme, map[reflect.Type]bool{}, pkginfo.AllowedNonstandardJSONNames, pkginfo.AllowedNoJSONTags)
}

func verifyGroupNameSuffix(scheme *runtime.Scheme, _ *ComponentConfigPackage) error {
	return apinamingtest.VerifyGroupNames(scheme, sets.NewString())
}

func verifyGroupNameMatch(_ *runtime.Scheme, pkginfo *ComponentConfigPackage) error {
	if pkginfo.GroupName != pkginfo.SchemeGroupVersion.Group {
		return fmt.Errorf("GroupName must equal SchemeGroupVersion.Group")
	}
	return nil
}

func verifyCorrectGroupName(_ *runtime.Scheme, pkginfo *ComponentConfigPackage) error {
	desiredGroupName := fmt.Sprintf("%s.config.k8s.io", lowercaseWithoutDashes(pkginfo.ComponentName))
	if pkginfo.SchemeGroupVersion.Group != desiredGroupName {
		return fmt.Errorf("GroupName isn't the expected value %q", desiredGroupName)
	}
	return nil
}

func verifyComponentConfigKindExists(scheme *runtime.Scheme, pkginfo *ComponentConfigPackage) error {
	expectedKind := fmt.Sprintf("%sConfiguration", dashesToCapitalCase(pkginfo.ComponentName))
	expectedGVK := pkginfo.SchemeGroupVersion.WithKind(expectedKind)
	if !scheme.Recognizes(expectedGVK) {
		return fmt.Errorf("Kind %s not registered in the scheme as expected", expectedKind)
	}
	return nil
}

func verifySchemeBuilderInit(_ *runtime.Scheme, pkginfo *ComponentConfigPackage) error {
	// TODO(stealthybox): implement missing SchemeBuilder verification
	//   Test that the SchemeBuilder contains exactly one init func: addKnownTypes
	return nil
}

func verifyTypeRegistration(_ *runtime.Scheme, pkginfo *ComponentConfigPackage) error {
	// TODO(stealthybox): implement missing SchemeBuilder verification
	//   Test that addKnownTypes and AddToScheme registers at least one type and doesn't error
	return nil
}

func verifyExternalAPIVersion(_ *runtime.Scheme, pkginfo *ComponentConfigPackage) error {
	if !APIVersionRegexp.MatchString(pkginfo.SchemeGroupVersion.Version) {
		return fmt.Errorf("API version for package invalid, needs to either match %q", APIVersionRegexp.String())
	}
	return nil
}

func verifyInternalAPIVersion(_ *runtime.Scheme, pkginfo *ComponentConfigPackage) error {
	if pkginfo.SchemeGroupVersion.Version != runtime.APIVersionInternal {
		return fmt.Errorf("API version must be %q", runtime.APIVersionInternal)
	}
	return nil
}

func lowercaseWithoutDashes(str string) string {
	return strings.Replace(strings.ToLower(str), "-", "", -1)
}

func dashesToCapitalCase(str string) string {
	segments := strings.Split(str, "-")
	result := ""
	for _, segment := range segments {
		result += strings.Title(segment)
	}
	return result
}
