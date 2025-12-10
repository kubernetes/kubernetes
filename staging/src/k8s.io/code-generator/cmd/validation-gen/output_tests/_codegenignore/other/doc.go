/*
Copyright 2024 The Kubernetes Authors.

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

// +k8s:validation-gen=*

// This is a test package.  It exists to demonstrate references to types that
// are not part of the gengo args.  Even though this package purports to have
// validations, it is outside of the args used when generating output_tests,
// and so the generated could should NOT descend into these.
package other

// +k8s:validateFalse="you should not see this outside of this pkg"
type StringType string

// +k8s:validateFalse="you should not see this outside of this pkg"
type IntType int

// +k8s:validateFalse="you should not see this outside of this pkg"
type StructType struct {
	// +k8s:validateFalse="you should not see this outside of this pkg"
	StringField string `json:"stringField"`
}
