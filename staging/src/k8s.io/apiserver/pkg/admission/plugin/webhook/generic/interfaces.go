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

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/webhook"
)

// Source can list dynamic webhook plugins.
type Source interface {
	Webhooks() []webhook.WebhookAccessor
	HasSynced() bool
}

// VersionedAttributes is a wrapper around the original admission attributes, adding versioned
// variants of the object and old object.
type VersionedAttributes struct {
	// Attributes holds the original admission attributes
	admission.Attributes
	// VersionedOldObject holds Attributes.OldObject (if non-nil), converted to VersionedKind.
	// It must never be mutated.
	VersionedOldObject runtime.Object
	// VersionedObject holds Attributes.Object (if non-nil), converted to VersionedKind.
	// If mutated, Dirty must be set to true by the mutator.
	VersionedObject runtime.Object
	// VersionedKind holds the fully qualified kind
	VersionedKind schema.GroupVersionKind
	// Dirty indicates VersionedObject has been modified since being converted from Attributes.Object
	Dirty bool
}

// GetObject overrides the Attributes.GetObject()
func (v *VersionedAttributes) GetObject() runtime.Object {
	if v.VersionedObject != nil {
		return v.VersionedObject
	}
	return v.Attributes.GetObject()
}

// WebhookInvocation describes how to call a webhook, including the resource and subresource the webhook registered for,
// and the kind that should be sent to the webhook.
type WebhookInvocation struct {
	Webhook     webhook.WebhookAccessor
	Resource    schema.GroupVersionResource
	Subresource string
	Kind        schema.GroupVersionKind
}

// Dispatcher dispatches webhook call to a list of webhooks with admission attributes as argument.
type Dispatcher interface {
	// Dispatch a request to the webhooks. Dispatcher may choose not to
	// call a hook, either because the rules of the hook does not match, or
	// the namespaceSelector or the objectSelector of the hook does not
	// match. A non-nil error means the request is rejected.
	Dispatch(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces, hooks []webhook.WebhookAccessor) error
}
