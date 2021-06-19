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

	"k8s.io/gengo/types"

	"k8s.io/kube-openapi/pkg/generators/rules"
)

const (
	// apiLifecycleExtension contains info about the component defining the
	// API field, the prerelease status of the API field, the minimum
	// component version for the prerelease and the feature gate associated
	// with the field.
	apiLifecycleExtension = "x-kubernetes-api-lifecycle"
)

// parseAPILifecycle returns a map of lifecycle tag keys to their values.
// If an error is encountered for a particular key, it returns a nil map.
func parseAPILifecycle(comments []string) (map[string]string, []error) {
	errors := []error{}
	apiLifecycle := map[string]string{}

	commentTag := types.ExtractCommentTags("+", comments)[rules.TagLifecycle]
	if commentTag != nil {
		// only consider the first occurance of tagLifecycleComponent
		// commentValues is of the form ["kubernetes", "status=alpha", "minVersion=1.22", "featureGate=Foo"]
		commentValues := strings.Split(commentTag[0], ",")
		apiLifecycle[rules.TagComponent] = commentValues[0]

		if len(commentValues) > 1 {
			for _, commentValue := range commentValues[1:] {
				commentValueParts := strings.Split(commentValue, "=")
				if len(commentValueParts) != 2 {
					errors = append(errors, fmt.Errorf("unrecognized key \"%s\" for extension %s, must of the form foo=bar", commentValueParts, apiLifecycleExtension))
					continue
				}
				if !rules.AllowedTagKeys.Has(commentValueParts[0]) {
					errors = append(errors, fmt.Errorf("unrecognized key \"%s\" for extension %s, must be one of %q", commentValueParts[0], apiLifecycleExtension, rules.AllowedTagKeys.List()))
					continue
				}
				apiLifecycle[commentValueParts[0]] = commentValueParts[1]
			}
		}

		if len(errors) > 0 {
			return nil, errors
		}
	}
	return apiLifecycle, errors
}

func validateAPILifecycleTags(tags map[string]string) []error {
	errors := []error{}
	for tag, value := range tags {
		if !rules.AllowedTagKeys.Has(tag) {
			errors = append(errors, fmt.Errorf("unrecognized tag key: %s. supported values are: %q", tag, rules.AllowedTagKeys.List()))
		}
		if tag == rules.TagStatus && !rules.AllowedStatusNames.Has(value) {
			errors = append(errors, fmt.Errorf("unrecognized status: %s. supported names are: %q", value, rules.AllowedStatusNames.List()))
		}
	}
	return errors
}

// emit prints the apiLifecycleExtension
func emitAPILifecycle(apiLifecycle map[string]string, g openAPITypeWriter) {
	if apiLifecycle == nil {
		return
	}

	componentValue, exists := apiLifecycle[rules.TagComponent]
	if !exists {
		return
	}
	g.Do("\"$.$\": map[string]interface{}{\n", componentValue)

	if statusValue, exists := apiLifecycle[rules.TagStatus]; exists {
		g.Do("\"$.$\": ", rules.TagStatus)
		g.Do("\"$.$\",\n", statusValue)
	}
	if minVersionValue, exists := apiLifecycle[rules.TagMinVersion]; exists {
		g.Do("\"$.$\": ", rules.TagMinVersion)
		g.Do("\"$.$\",\n", minVersionValue)
	}
	if featureGateValue, exists := apiLifecycle[rules.TagFeatureGate]; exists {
		g.Do("\"$.$\": ", rules.TagFeatureGate)
		g.Do("\"$.$\",\n", featureGateValue)
	}
	g.Do("},\n", nil)
}
