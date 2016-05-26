/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package admission

import "k8s.io/kubernetes/pkg/controller/framework"

// Validator should be implemented by admission plugins that can validate themselves
// after initialization has happened.
type Validator interface {
	Validate() error
}

// WantsNamespaceInformer should be implemented by admission plugins that need
// a NamespaceInformer
type WantsNamespaceInformer interface {
	SetNamespaceInformer(framework.SharedInformer)
}
