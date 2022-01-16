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

// APIVersionRegexp is the regular expression that matches with valid apiversion
var APIVersionRegexp = regexp.MustCompile(`^v\d+((alpha|beta){1}\d+)?$`)

// ComponentConfigPackage is used in APIGroup Testing
type ComponentConfigPackage struct {
	ComponentName               string
	GroupName                   string
	SchemeGroupVersion          schema.GroupVersion
	AddToScheme                 func(*runtime.Scheme) error
	SkipTests                   sets.String
	AllowedTags                 map[reflect.Type]bool
	AllowedNonstandardJSONNames map[reflect.Type]string
}

type testingFunc func(*runtime.Scheme, *ComponentConfigPackage) error

const (
	verifyTagNaming                 = "verifyTagNaming"
	verifyGroupNameSuffix           = "verifyGroupNameSuffix"
	verifyGroupNameMatch            = "verifyGroupNameMatch"
	verifyCorrectGroupName          = "verifyCorrectGroupName"
	verifyComponentConfigKindExists = "verifyComponentConfigKindExists"
	verifyExternalAPIVersion        = "verifyExternalAPIVersion"
	verifyInternalAPIVersion        = "verifyInternalAPIVersion"
)

var testingFuncs = map[string]testingFunc{
	verifyTagNaming:                 verifyTagNamingFunc,
	verifyGroupNameSuffix:           verifyGroupNameSuffixFunc,
	verifyGroupNameMatch:            verifyGroupNameMatchFunc,
	verifyCorrectGroupName:          verifyCorrectGroupNameFunc,
	verifyComponentConfigKindExists: verifyComponentConfigKindExistsFunc,
}

// VerifyExternalTypePackage tests if external component config package is defined correctly
// Test tag naming (json name should match Go name)
// Test that GroupName has the k8s.io suffix
// Test that GroupName == SchemeGroupVersion.GroupName
// Test that the API version follows the right pattern and isn't internal
// Test that addKnownTypes and AddToScheme registers at least one type and doesn't error
// Test that the GroupName is named correctly (based on ComponentName), and there is a {Component}Configuration kind in the scheme
func VerifyExternalTypePackage(pkginfo *ComponentConfigPackage) error {
	scheme, err := setup(pkginfo)
	if err != nil {
		return fmt.Errorf("test setup error: %v", err)
	}
	extraFns := map[string]testingFunc{
		verifyExternalAPIVersion: verifyExternalAPIVersionFunc,
	}
	return runFuncs(scheme, pkginfo, extraFns)
}

// VerifyInternalTypePackage tests if internal component config package is defined correctly
// Test tag naming (no tags allowed)
// Test that GroupName has the k8s.io suffix
// Test that GroupName == SchemeGroupVersion.GroupName
// API version should be internal
// Test that addKnownTypes and AddToScheme registers at least one type and doesn't error
// Test that the GroupName is named correctly (based on ComponentName), and there is a {Component}Configuration kind in the scheme
func VerifyInternalTypePackage(pkginfo *ComponentConfigPackage) error {
	scheme, err := setup(pkginfo)
	if err != nil {
		return fmt.Errorf("test setup error: %v", err)
	}
	extraFns := map[string]testingFunc{
		verifyInternalAPIVersion: verifyInternalAPIVersionFunc,
	}
	return runFuncs(scheme, pkginfo, extraFns)
}

func setup(pkginfo *ComponentConfigPackage) (*runtime.Scheme, error) {
	if len(pkginfo.ComponentName) == 0 ||
		len(pkginfo.GroupName) == 0 ||
		pkginfo.SchemeGroupVersion.Empty() ||
		pkginfo.AddToScheme == nil {
		return nil, fmt.Errorf("invalid argument: not all parameters were passed correctly to the function")
	}

	scheme := runtime.NewScheme()
	if err := pkginfo.AddToScheme(scheme); err != nil {
		return nil, fmt.Errorf("AddToScheme must not return an error: %v", err)
	}
	if len(scheme.AllKnownTypes()) == 0 {
		return nil, fmt.Errorf("AddToScheme doesn't register any type")
	}
	return scheme, nil
}

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

func verifyTagNamingFunc(scheme *runtime.Scheme, pkginfo *ComponentConfigPackage) error {
	return apinamingtest.VerifyTagNaming(scheme, pkginfo.AllowedTags, pkginfo.AllowedNonstandardJSONNames)
}

func verifyGroupNameSuffixFunc(scheme *runtime.Scheme, _ *ComponentConfigPackage) error {
	return apinamingtest.VerifyGroupNames(scheme, sets.NewString())
}

func verifyGroupNameMatchFunc(_ *runtime.Scheme, pkginfo *ComponentConfigPackage) error {
	if pkginfo.GroupName != pkginfo.SchemeGroupVersion.Group {
		return fmt.Errorf("GroupName must be equal to SchemeGroupVersion.Group, GroupName: %v,SchemeGroupVersion.Group: %v",
			pkginfo.GroupName, pkginfo.SchemeGroupVersion.Group)
	}
	return nil
}

func verifyCorrectGroupNameFunc(_ *runtime.Scheme, pkginfo *ComponentConfigPackage) error {
	desiredGroupName := fmt.Sprintf("%s.config.k8s.io", lowercaseWithoutDashes(pkginfo.ComponentName))
	if pkginfo.SchemeGroupVersion.Group != desiredGroupName {
		return fmt.Errorf("got GroupName %q, want %q", pkginfo.SchemeGroupVersion.Group, desiredGroupName)

	}
	return nil
}

func verifyComponentConfigKindExistsFunc(scheme *runtime.Scheme, pkginfo *ComponentConfigPackage) error {
	expectedKind := fmt.Sprintf("%sConfiguration", dashesToCapitalCase(pkginfo.ComponentName))
	expectedGVK := pkginfo.SchemeGroupVersion.WithKind(expectedKind)
	if !scheme.Recognizes(expectedGVK) {
		registeredKinds := sets.NewString()
		for gvk := range scheme.AllKnownTypes() {
			registeredKinds.Insert(gvk.Kind)
		}
		return fmt.Errorf("Kind %s not registered in the scheme, registered kinds are %v", expectedKind, registeredKinds.List())
	}
	return nil
}

func verifyExternalAPIVersionFunc(_ *runtime.Scheme, pkginfo *ComponentConfigPackage) error {
	if !APIVersionRegexp.MatchString(pkginfo.SchemeGroupVersion.Version) {
		return fmt.Errorf("invalid API version %q, must match %q", pkginfo.SchemeGroupVersion.Version, APIVersionRegexp.String())
	}
	return nil
}

func verifyInternalAPIVersionFunc(_ *runtime.Scheme, pkginfo *ComponentConfigPackage) error {
	if pkginfo.SchemeGroupVersion.Version != runtime.APIVersionInternal {
		return fmt.Errorf("internal API version must be %q, got %q",
			runtime.APIVersionInternal, pkginfo.SchemeGroupVersion.Version)
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
