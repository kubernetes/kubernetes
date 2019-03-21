/*
Copyright 2018 The Kubernetes Authors.

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

package generic

import (
	"context"

	"k8s.io/api/admissionregistration/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
)

// Source can list dynamic webhook plugins.
type Source interface {
	Webhooks() []v1beta1.Webhook
	HasSynced() bool
}

// VersionedAttributes is a wrapper around the original admission attributes, adding versioned
// variants of the object and old object.
type VersionedAttributes struct {
	admission.Attributes
	VersionedOldObject runtime.Object
	VersionedObject    runtime.Object
}

// Dispatcher dispatches webhook call to a list of webhooks with admission attributes as argument.
type Dispatcher interface {
	// Dispatch a request to the webhooks using the given webhooks. A non-nil error means the request is rejected.
	Dispatch(ctx context.Context, a *VersionedAttributes, o admission.ObjectInterfaces, hooks []*v1beta1.Webhook) error
}
