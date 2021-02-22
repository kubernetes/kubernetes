/*
Copyright 2021 The Kubernetes Authors.

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

package generators

import (
	"fmt"
	"strings"

	"k8s.io/gengo/examples/set-gen/sets"
	"k8s.io/gengo/types"
)

// x-kubernetes-field-version contains info about the prerelease status,
// minium version for the prerelease and the feature gate.
// If the field has multiple prerelease statuses, the extension
// will include all of them.
const fieldVersionExtension = "x-kubernetes-field-version"

// tagPrerelease is the comment tag prefix for specifying
// the field version info. Example tag format:
// +k8s:openapi-gen:prerelease=alpha,version=v1.20,featureGate=EndpointSliceNodeName
const tagPrerelease = "k8s:openapi-gen:prerelease"

// the prerelease names must match the prereleases defined in
// k8s.io/component-base/featuregate
var allowedPrereleaseNames = sets.NewString("alpha", "beta", "stable", "deprecated")

// fieldVersion contains the following information about a particular field:
// 1. prerelease status
// 2. minimum k8s version for the prerelase status
// 3. feature gate
type fieldVersion struct {
	prerelease  string
	minVersion  string
	featureGate string
}

func parseFieldVersions(comments []string) ([]fieldVersion, []error) {
	fieldVersions := []fieldVersion{}
	errors := []error{}

	for _, comment := range types.ExtractCommentTags("+", comments)[tagPrerelease] {
		itemMap := map[string]string{}
		itemValues := strings.Split(comment, ",")

		prereleaseName := strings.TrimSpace(itemValues[0])
		if !allowedPrereleaseNames.Has(prereleaseName) {
			errors = append(errors, fmt.Errorf("unrecognized prerelease: %q. supported names are: %q", prereleaseName, allowedPrereleaseNames.List()))
		}

		for _, itemValue := range itemValues[1:] {
			itemValueParts := strings.Split(itemValue, "=")
			if len(itemValueParts) == 2 {
				itemMap[strings.TrimSpace(itemValueParts[0])] = strings.TrimSpace(itemValueParts[1])
			} else {
				errors = append(errors, fmt.Errorf("unrecognized item in prerelease: %v", itemValue))
			}
		}
		if itemMap["minVersion"] == "" {
			errors = append(errors, fmt.Errorf("prerelease item %s is missing minVersion info", prereleaseName))
		}
		if itemMap["featureGate"] == "" {
			errors = append(errors, fmt.Errorf("prerelease item %s is missing feature gate info", prereleaseName))
		}

		fv := fieldVersion{
			prerelease:  prereleaseName,
			minVersion:  itemMap["minVersion"],
			featureGate: itemMap["featureGate"],
		}
		fieldVersions = append(fieldVersions, fv)
	}

	return fieldVersions, errors
}

func (fv fieldVersion) validate() []error {
	errors := []error{}
	if !allowedPrereleaseNames.Has(fv.prerelease) {
		errors = append(errors, fmt.Errorf("unrecognized prerelease: %q. supported names are: %q", fv.prerelease, allowedPrereleaseNames.List()))
	}
	if fv.minVersion == "" {
		errors = append(errors, fmt.Errorf("prerelease item %s is missing minVersion info", fv.prerelease))
	}
	if fv.featureGate == "" {
		errors = append(errors, fmt.Errorf("prerelease item %s is missing feature gate info", fv.prerelease))
	}
	return errors
}

func validateFieldVersions(fieldVersions []fieldVersion) []error {
	errors := []error{}
	for _, fv := range fieldVersions {
		if err := fv.validate(); err != nil {
			errors = append(errors, err...)
		}
	}
	return errors
}

// emit prints the fieldVersion, can be called on a nil fieldVersion (emits nothing)
func (fv *fieldVersion) emit(g openAPITypeWriter) {
	if fv == nil {
		return
	}
	g.Do("map[string]interface{}{\n", nil)
	if fv.prerelease != "" {
		g.Do("\"prerelease\": \"$.$\",\n", fv.prerelease)
	}
	if fv.minVersion != "" {
		g.Do("\"minVersion\": \"$.$\",\n", fv.minVersion)
	}
	if fv.featureGate != "" {
		g.Do("\"featureGate\": \"$.$\",\n", fv.featureGate)
	}
	g.Do("},\n", nil)
}
