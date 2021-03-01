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

package main

import (
	"fmt"
	"os"
	"regexp"

	kruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/component-base/featuregate"
	kfeatures "k8s.io/kubernetes/pkg/features"

	flag "github.com/spf13/pflag"
	"k8s.io/klog/v2"
)

var (
	typeSrc              = flag.StringP("type-src", "s", "", "From where we are going to read the types")
	featureGateRegex     = regexp.MustCompile(`featureGate=([A-Za-z]*)`)
	minVersionRegex      = regexp.MustCompile(`minVersion=([A-Za-z0-9.]*)`)
	prereleaseRegex      = regexp.MustCompile(`prerelease=([A-Za-z]*)`)
	minVersionValueRegex = regexp.MustCompile(`v1\.[0-9]*`)
	prereleaseValues     = []string{"alpha", "beta", "stable", "deprecated"}
)

func main() {
	flag.Parse()

	if *typeSrc == "" {
		klog.Fatalf("Please define -s flag as it is the source file")
	}

	docsForTypes := kruntime.ParseDocumentationFrom(*typeSrc, false)

	errors := validateFieldVersionTagsInComments(docsForTypes)
	if len(errors) > 0 {
		for _, e := range errors {
			klog.V(0).Error(e, "invalid feature gate tags detected:")
		}
		os.Exit(len(errors))
	}
}

// validateFieldVersionTagsInComments applies the following validations
// for field version tags on comments on fields:
// 1. prelease value is one of prereleaseValues
// 2. minVersion satisfies the regex minVersionValueRegex
// 3. featureGate value is a valid feature gate
// TODO: validate that prerelease and minVersion exist
func validateFieldVersionTagsInComments(kubeTypes []kruntime.KubeTypes) []error {
	errors := []error{}
	for _, kubeType := range kubeTypes {
		structName := kubeType[0].Name
		kubeType = kubeType[1:]

		for _, pair := range kubeType { // Iterate only the fields
			invalidPrereleases := []string{}
			for _, m := range prereleaseRegex.FindAllStringSubmatch(pair.Doc, -1) {
				var validPrerelease bool
				for _, p := range prereleaseValues {
					if m[1] == p {
						validPrerelease = true
					}
				}
				if !validPrerelease {
					invalidPrereleases = append(invalidPrereleases, m[1])
				}
			}

			invalidMinVersions := []string{}
			for _, m := range minVersionRegex.FindAllStringSubmatch(pair.Doc, -1) {
				if !minVersionValueRegex.MatchString(m[1]) {
					invalidMinVersions = append(invalidMinVersions, m[1])
				}
			}

			invalidFeatureGates := []string{}
			for _, m := range featureGateRegex.FindAllStringSubmatch(pair.Doc, -1) {
				if _, ok := kfeatures.DefaultKubernetesFeatureGates[featuregate.Feature(m[1])]; !ok {
					invalidFeatureGates = append(invalidFeatureGates, m[1])
				}
			}

			if len(invalidPrereleases) != 0 || len(invalidMinVersions) != 0 || len(invalidFeatureGates) != 0 {
				msg := fmt.Sprintf("In struct: %s, field %s has:\n", structName, pair.Name)
				if len(invalidPrereleases) != 0 {
					msg = msg + fmt.Sprintf("- %d invalid prerelease comment tag(s): %v\n", len(invalidPrereleases), invalidPrereleases)
				}
				if len(invalidMinVersions) != 0 {
					msg = msg + fmt.Sprintf("- %d invalid minVersion comment tag(s): %v\n", len(invalidMinVersions), invalidMinVersions)
				}
				if len(invalidFeatureGates) != 0 {
					msg = msg + fmt.Sprintf("- %d invalid featureGate comment tag(s): %v", len(invalidFeatureGates), invalidFeatureGates)
				}
				errors = append(errors, fmt.Errorf("%s", msg))
			}
		}
	}
	return errors
}
