/*
Copyright 2014 Google Inc. All rights reserved.

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

package v1beta3

import (
	"fmt"

	newer "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func init() {
	// Add field conversion funcs.
	err := newer.Scheme.AddFieldLabelConversionFunc("v1beta3", "pods",
		func(label, value string) (string, string, error) {
			switch label {
			case "name":
				fallthrough
			case "Status.Phase":
				fallthrough
			case "Status.Host":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		})
	if err != nil {
		// If one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}
}
