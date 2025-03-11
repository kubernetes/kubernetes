/*
Copyright 2014 The Kubernetes Authors.

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

package content

import (
	"fmt"

	"k8s.io/apimachinery/pkg/api/validate/constraints"
)

// MinError returns a string explanation of a "must be greater than or equal"
// validation failure.
func MinError[T constraints.Integer](min T) string {
	return fmt.Sprintf("must be greater than or equal to %d", min)
}
