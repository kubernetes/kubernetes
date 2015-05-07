/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

// Attributes is an interface used by AdmissionController to get information about a request
// that is used to make an admission decision.
type Attributes interface {
	GetNamespace() string
	GetResource() string
	GetOperation() string
	GetObject() runtime.Object
	GetKind() string
}

// Interface is an abstract, pluggable interface for Admission Control decisions.
type Interface interface {
	// Admit makes an admission decision based on the request attributes
	Admit(a Attributes) (err error)
}
