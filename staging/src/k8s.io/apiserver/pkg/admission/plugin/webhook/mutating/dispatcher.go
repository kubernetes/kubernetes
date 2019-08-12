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

// Package mutating delegates admission checks to dynamically configured
// mutating webhooks.
package mutating

import (
	"context"
	"fmt"
	"time"

	jsonpatch "github.com/evanphx/json-patch"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/klog"

	admissionv1 "k8s.io/api/admission/v1"
	"k8s.io/api/admissionregistration/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/admission"
	admissionmetrics "k8s.io/apiserver/pkg/admission/metrics"
	"k8s.io/apiserver/pkg/admission/plugin/webhook"
	webhookerrors "k8s.io/apiserver/pkg/admission/plugin/webhook/errors"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/generic"
	webhookrequest "k8s.io/apiserver/pkg/admission/plugin/webhook/request"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/util"
	webhookutil "k8s.io/apiserver/pkg/util/webhook"
	utiltrace "k8s.io/utils/trace"
)

type mutatingDispatcher struct {
	cm     *webhookutil.ClientManager
	plugin *Plugin
}

func newMutatingDispatcher(p *Plugin) func(cm *webhookutil.ClientManager) generic.Dispatcher {
	return func(cm *webhookutil.ClientManager) generic.Dispatcher {
		return &mutatingDispatcher{cm, p}
	}
}

var _ generic.Dispatcher = &mutatingDispatcher{}

func (a *mutatingDispatcher) Dispatch(ctx context.Context, attr admission.Attributes, o admission.ObjectInterfaces, hooks []webhook.WebhookAccessor) error {
	reinvokeCtx := attr.GetReinvocationContext()
	var webhookReinvokeCtx *webhookReinvokeContext
	if v := reinvokeCtx.Value(PluginName); v != nil {
		webhookReinvokeCtx = v.(*webhookReinvokeContext)
	} else {
		webhookReinvokeCtx = &webhookReinvokeContext{}
		reinvokeCtx.SetValue(PluginName, webhookReinvokeCtx)
	}

	if reinvokeCtx.IsReinvoke() && webhookReinvokeCtx.IsOutputChangedSinceLastWebhookInvocation(attr.GetObject()) {
		// If the object has changed, we know the in-tree plugin re-invocations have mutated the object,
		// and we need to reinvoke all eligible webhooks.
		webhookReinvokeCtx.RequireReinvokingPreviouslyInvokedPlugins()
	}
	defer func() {
		webhookReinvokeCtx.SetLastWebhookInvocationOutput(attr.GetObject())
	}()
	var versionedAttr *generic.VersionedAttributes
	for _, hook := range hooks {
		attrForCheck := attr
		if versionedAttr != nil {
			attrForCheck = versionedAttr
		}
		invocation, statusErr := a.plugin.ShouldCallHook(hook, attrForCheck, o)
		if statusErr != nil {
			return statusErr
		}
		if invocation == nil {
			continue
		}
		hook, ok := invocation.Webhook.GetMutatingWebhook()
		if !ok {
			return fmt.Errorf("mutating webhook dispatch requires v1beta1.MutatingWebhook, but got %T", hook)
		}
		// This means that during reinvocation, a webhook will not be
		// called for the first time. For example, if the webhook is
		// skipped in the first round because of mismatching labels,
		// even if the labels become matching, the webhook does not
		// get called during reinvocation.
		if reinvokeCtx.IsReinvoke() && !webhookReinvokeCtx.ShouldReinvokeWebhook(invocation.Webhook.GetUID()) {
			continue
		}

		if versionedAttr == nil {
			// First webhook, create versioned attributes
			var err error
			if versionedAttr, err = generic.NewVersionedAttributes(attr, invocation.Kind, o); err != nil {
				return apierrors.NewInternalError(err)
			}
		} else {
			// Subsequent webhook, convert existing versioned attributes to this webhook's version
			if err := generic.ConvertVersionedAttributes(versionedAttr, invocation.Kind, o); err != nil {
				return apierrors.NewInternalError(err)
			}
		}

		t := time.Now()

		changed, err := a.callAttrMutatingHook(ctx, hook, invocation, versionedAttr, o)
		admissionmetrics.Metrics.ObserveWebhook(time.Since(t), err != nil, versionedAttr.Attributes, "admit", hook.Name)
		if changed {
			// Patch had changed the object. Prepare to reinvoke all previous webhooks that are eligible for re-invocation.
			webhookReinvokeCtx.RequireReinvokingPreviouslyInvokedPlugins()
			reinvokeCtx.SetShouldReinvoke()
		}
		if hook.ReinvocationPolicy != nil && *hook.ReinvocationPolicy == v1beta1.IfNeededReinvocationPolicy {
			webhookReinvokeCtx.AddReinvocableWebhookToPreviouslyInvoked(invocation.Webhook.GetUID())
		}
		if err == nil {
			continue
		}

		ignoreClientCallFailures := hook.FailurePolicy != nil && *hook.FailurePolicy == v1beta1.Ignore
		if callErr, ok := err.(*webhookutil.ErrCallingWebhook); ok {
			if ignoreClientCallFailures {
				klog.Warningf("Failed calling webhook, failing open %v: %v", hook.Name, callErr)
				utilruntime.HandleError(callErr)
				continue
			}
			klog.Warningf("Failed calling webhook, failing closed %v: %v", hook.Name, err)
			return apierrors.NewInternalError(err)
		}
		return err
	}

	// convert versionedAttr.VersionedObject to the internal version in the underlying admission.Attributes
	if versionedAttr != nil && versionedAttr.VersionedObject != nil && versionedAttr.Dirty {
		return o.GetObjectConvertor().Convert(versionedAttr.VersionedObject, versionedAttr.Attributes.GetObject(), nil)
	}

	return nil
}

// note that callAttrMutatingHook updates attr

func (a *mutatingDispatcher) callAttrMutatingHook(ctx context.Context, h *v1beta1.MutatingWebhook, invocation *generic.WebhookInvocation, attr *generic.VersionedAttributes, o admission.ObjectInterfaces) (bool, error) {
	if attr.Attributes.IsDryRun() {
		if h.SideEffects == nil {
			return false, &webhookutil.ErrCallingWebhook{WebhookName: h.Name, Reason: fmt.Errorf("Webhook SideEffects is nil")}
		}
		if !(*h.SideEffects == v1beta1.SideEffectClassNone || *h.SideEffects == v1beta1.SideEffectClassNoneOnDryRun) {
			return false, webhookerrors.NewDryRunUnsupportedErr(h.Name)
		}
	}

	uid, request, response, err := webhookrequest.CreateAdmissionObjects(attr, invocation)
	if err != nil {
		return false, &webhookutil.ErrCallingWebhook{WebhookName: h.Name, Reason: err}
	}
	// Make the webhook request
	client, err := a.cm.HookClient(util.HookClientConfigForWebhook(invocation.Webhook))
	if err != nil {
		return false, &webhookutil.ErrCallingWebhook{WebhookName: h.Name, Reason: err}
	}
	trace := utiltrace.New("Call mutating webhook",
		utiltrace.Field{"configuration", invocation.Webhook.GetConfigurationName()},
		utiltrace.Field{"webhook", h.Name},
		utiltrace.Field{"resource", attr.GetResource()},
		utiltrace.Field{"subresource", attr.GetSubresource()},
		utiltrace.Field{"operation", attr.GetOperation()},
		utiltrace.Field{"UID", uid})
	defer trace.LogIfLong(500 * time.Millisecond)
	r := client.Post().Context(ctx).Body(request)
	if h.TimeoutSeconds != nil {
		r = r.Timeout(time.Duration(*h.TimeoutSeconds) * time.Second)
	}
	if err := r.Do().Into(response); err != nil {
		return false, &webhookutil.ErrCallingWebhook{WebhookName: h.Name, Reason: err}
	}
	trace.Step("Request completed")

	result, err := webhookrequest.VerifyAdmissionResponse(uid, true, response)
	if err != nil {
		return false, &webhookutil.ErrCallingWebhook{WebhookName: h.Name, Reason: err}
	}

	for k, v := range result.AuditAnnotations {
		key := h.Name + "/" + k
		if err := attr.Attributes.AddAnnotation(key, v); err != nil {
			klog.Warningf("Failed to set admission audit annotation %s to %s for mutating webhook %s: %v", key, v, h.Name, err)
		}
	}

	if !result.Allowed {
		return false, webhookerrors.ToStatusErr(h.Name, result.Result)
	}

	if len(result.Patch) == 0 {
		return false, nil
	}
	patchObj, err := jsonpatch.DecodePatch(result.Patch)
	if err != nil {
		return false, apierrors.NewInternalError(err)
	}
	if len(patchObj) == 0 {
		return false, nil
	}

	// if a non-empty patch was provided, and we have no object we can apply it to (e.g. a DELETE admission operation), error
	if attr.VersionedObject == nil {
		return false, apierrors.NewInternalError(fmt.Errorf("admission webhook %q attempted to modify the object, which is not supported for this operation", h.Name))
	}

	var patchedJS []byte
	jsonSerializer := json.NewSerializer(json.DefaultMetaFactory, o.GetObjectCreater(), o.GetObjectTyper(), false)
	switch result.PatchType {
	// VerifyAdmissionResponse normalizes to v1 patch types, regardless of the AdmissionReview version used
	case admissionv1.PatchTypeJSONPatch:
		objJS, err := runtime.Encode(jsonSerializer, attr.VersionedObject)
		if err != nil {
			return false, apierrors.NewInternalError(err)
		}
		patchedJS, err = patchObj.Apply(objJS)
		if err != nil {
			return false, apierrors.NewInternalError(err)
		}
	default:
		return false, &webhookutil.ErrCallingWebhook{WebhookName: h.Name, Reason: fmt.Errorf("unsupported patch type %q", result.PatchType)}
	}

	var newVersionedObject runtime.Object
	if _, ok := attr.VersionedObject.(*unstructured.Unstructured); ok {
		// Custom Resources don't have corresponding Go struct's.
		// They are represented as Unstructured.
		newVersionedObject = &unstructured.Unstructured{}
	} else {
		newVersionedObject, err = o.GetObjectCreater().New(attr.VersionedKind)
		if err != nil {
			return false, apierrors.NewInternalError(err)
		}
	}

	// TODO: if we have multiple mutating webhooks, we can remember the json
	// instead of encoding and decoding for each one.
	if newVersionedObject, _, err = jsonSerializer.Decode(patchedJS, nil, newVersionedObject); err != nil {
		return false, apierrors.NewInternalError(err)
	}

	changed := !apiequality.Semantic.DeepEqual(attr.VersionedObject, newVersionedObject)
	trace.Step("Patch applied")
	attr.Dirty = true
	attr.VersionedObject = newVersionedObject
	o.GetObjectDefaulter().Default(attr.VersionedObject)
	return changed, nil
}
