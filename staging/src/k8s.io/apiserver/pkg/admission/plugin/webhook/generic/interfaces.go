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

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/webhook"
)

type VersionedAttributeAccessor interface {
	VersionedAttribute(gvk schema.GroupVersionKind) (*admission.VersionedAttributes, error)
}

// Source can list dynamic webhook plugins.
type Source interface {
	Webhooks() []webhook.WebhookAccessor
	HasSynced() bool
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
