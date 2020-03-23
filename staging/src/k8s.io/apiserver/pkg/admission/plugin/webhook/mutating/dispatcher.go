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
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
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
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	webhookutil "k8s.io/apiserver/pkg/util/webhook"
	utiltrace "k8s.io/utils/trace"
)

const (
	// PatchAuditAnnotationPrefix is a prefix for persisting webhook patch in audit annotation.
	// Audit handler decides whether annotation with this prefix should be logged based on audit level.
	// Since mutating webhook patches the request body, audit level must be greater or equal to Request
	// for the annotation to be logged
	PatchAuditAnnotationPrefix = "patch.webhook.admission.k8s.io/"
	// MutationAuditAnnotationPrefix is a prefix for presisting webhook mutation existence in audit annotation.
	MutationAuditAnnotationPrefix = "mutation.webhook.admission.k8s.io/"
)

var encodingjson = json.CaseSensitiveJsonIterator()

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
	for i, hook := range hooks {
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
			return fmt.Errorf("mutating webhook dispatch requires v1.MutatingWebhook, but got %T", hook)
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
		round := 0
		if reinvokeCtx.IsReinvoke() {
			round = 1
		}
		changed, err := a.callAttrMutatingHook(ctx, hook, invocation, versionedAttr, o, round, i)
		ignoreClientCallFailures := hook.FailurePolicy != nil && *hook.FailurePolicy == admissionregistrationv1.Ignore
		rejected := false
		if err != nil {
			switch err := err.(type) {
			case *webhookutil.ErrCallingWebhook:
				if !ignoreClientCallFailures {
					rejected = true
					admissionmetrics.Metrics.ObserveWebhookRejection(hook.Name, "admit", string(versionedAttr.Attributes.GetOperation()), admissionmetrics.WebhookRejectionCallingWebhookError, 0)
				}
			case *webhookutil.ErrWebhookRejection:
				rejected = true
				admissionmetrics.Metrics.ObserveWebhookRejection(hook.Name, "admit", string(versionedAttr.Attributes.GetOperation()), admissionmetrics.WebhookRejectionNoError, int(err.Status.ErrStatus.Code))
			default:
				rejected = true
				admissionmetrics.Metrics.ObserveWebhookRejection(hook.Name, "admit", string(versionedAttr.Attributes.GetOperation()), admissionmetrics.WebhookRejectionAPIServerInternalError, 0)
			}
		}
		admissionmetrics.Metrics.ObserveWebhook(time.Since(t), rejected, versionedAttr.Attributes, "admit", hook.Name)
		if changed {
			// Patch had changed the object. Prepare to reinvoke all previous webhooks that are eligible for re-invocation.
			webhookReinvokeCtx.RequireReinvokingPreviouslyInvokedPlugins()
			reinvokeCtx.SetShouldReinvoke()
		}
		if hook.ReinvocationPolicy != nil && *hook.ReinvocationPolicy == admissionregistrationv1.IfNeededReinvocationPolicy {
			webhookReinvokeCtx.AddReinvocableWebhookToPreviouslyInvoked(invocation.Webhook.GetUID())
		}
		if err == nil {
			continue
		}

		if callErr, ok := err.(*webhookutil.ErrCallingWebhook); ok {
			if ignoreClientCallFailures {
				klog.Warningf("Failed calling webhook, failing open %v: %v", hook.Name, callErr)
				utilruntime.HandleError(callErr)

				select {
				case <-ctx.Done():
					// parent context is canceled or timed out, no point in continuing
					return apierrors.NewTimeoutError("request did not complete within requested timeout", 0)
				default:
					// individual webhook timed out, but parent context did not, continue
					continue
				}
			}
			klog.Warningf("Failed calling webhook, failing closed %v: %v", hook.Name, err)
			return apierrors.NewInternalError(err)
		}
		if rejectionErr, ok := err.(*webhookutil.ErrWebhookRejection); ok {
			return rejectionErr.Status
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

func (a *mutatingDispatcher) callAttrMutatingHook(ctx context.Context, h *admissionregistrationv1.MutatingWebhook, invocation *generic.WebhookInvocation, attr *generic.VersionedAttributes, o admission.ObjectInterfaces, round, idx int) (bool, error) {
	configurationName := invocation.Webhook.GetConfigurationName()
	annotator := newWebhookAnnotator(attr, round, idx, h.Name, configurationName)
	changed := false
	defer func() { annotator.addMutationAnnotation(changed) }()
	if attr.Attributes.IsDryRun() {
		if h.SideEffects == nil {
			return false, &webhookutil.ErrCallingWebhook{WebhookName: h.Name, Reason: fmt.Errorf("Webhook SideEffects is nil")}
		}
		if !(*h.SideEffects == admissionregistrationv1.SideEffectClassNone || *h.SideEffects == admissionregistrationv1.SideEffectClassNoneOnDryRun) {
			return false, webhookerrors.NewDryRunUnsupportedErr(h.Name)
		}
	}

	uid, request, response, err := webhookrequest.CreateAdmissionObjects(attr, invocation)
	if err != nil {
		return false, &webhookutil.ErrCallingWebhook{WebhookName: h.Name, Reason: err}
	}
	// Make the webhook request
	client, err := invocation.Webhook.GetRESTClient(a.cm)
	if err != nil {
		return false, &webhookutil.ErrCallingWebhook{WebhookName: h.Name, Reason: err}
	}
	trace := utiltrace.New("Call mutating webhook",
		utiltrace.Field{"configuration", configurationName},
		utiltrace.Field{"webhook", h.Name},
		utiltrace.Field{"resource", attr.GetResource()},
		utiltrace.Field{"subresource", attr.GetSubresource()},
		utiltrace.Field{"operation", attr.GetOperation()},
		utiltrace.Field{"UID", uid})
	defer trace.LogIfLong(500 * time.Millisecond)

	// if the webhook has a specific timeout, wrap the context to apply it
	if h.TimeoutSeconds != nil {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Duration(*h.TimeoutSeconds)*time.Second)
		defer cancel()
	}

	r := client.Post().Body(request)

	// if the context has a deadline, set it as a parameter to inform the backend
	if deadline, hasDeadline := ctx.Deadline(); hasDeadline {
		// compute the timeout
		if timeout := time.Until(deadline); timeout > 0 {
			// if it's not an even number of seconds, round up to the nearest second
			if truncated := timeout.Truncate(time.Second); truncated != timeout {
				timeout = truncated + time.Second
			}
			// set the timeout
			r.Timeout(timeout)
		}
	}

	if err := r.Do(ctx).Into(response); err != nil {
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
		return false, &webhookutil.ErrWebhookRejection{Status: webhookerrors.ToStatusErr(h.Name, result.Result)}
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

	changed = !apiequality.Semantic.DeepEqual(attr.VersionedObject, newVersionedObject)
	trace.Step("Patch applied")
	annotator.addPatchAnnotation(patchObj, result.PatchType)
	attr.Dirty = true
	attr.VersionedObject = newVersionedObject
	o.GetObjectDefaulter().Default(attr.VersionedObject)
	return changed, nil
}

type webhookAnnotator struct {
	attr                  *generic.VersionedAttributes
	patchAnnotationKey    string
	mutationAnnotationKey string
	webhook               string
	configuration         string
}

func newWebhookAnnotator(attr *generic.VersionedAttributes, round, idx int, webhook, configuration string) *webhookAnnotator {
	return &webhookAnnotator{
		attr:                  attr,
		patchAnnotationKey:    fmt.Sprintf("%sround_%d_index_%d", PatchAuditAnnotationPrefix, round, idx),
		mutationAnnotationKey: fmt.Sprintf("%sround_%d_index_%d", MutationAuditAnnotationPrefix, round, idx),
		webhook:               webhook,
		configuration:         configuration,
	}
}

func (w *webhookAnnotator) addMutationAnnotation(mutated bool) {
	if w.attr == nil || w.attr.Attributes == nil {
		return
	}
	value, err := mutationAnnotationValue(w.configuration, w.webhook, mutated)
	if err != nil {
		klog.Warningf("unexpected error composing mutating webhook annotation: %v", err)
		return
	}
	if err := w.attr.Attributes.AddAnnotation(w.mutationAnnotationKey, value); err != nil {
		klog.Warningf("failed to set mutation annotation for mutating webhook key %s to %s: %v", w.mutationAnnotationKey, value, err)
	}
}

func (w *webhookAnnotator) addPatchAnnotation(patch interface{}, patchType admissionv1.PatchType) {
	if w.attr == nil || w.attr.Attributes == nil {
		return
	}
	var value string
	var err error
	switch patchType {
	case admissionv1.PatchTypeJSONPatch:
		value, err = jsonPatchAnnotationValue(w.configuration, w.webhook, patch)
		if err != nil {
			klog.Warningf("unexpected error composing mutating webhook JSON patch annotation: %v", err)
			return
		}
	default:
		klog.Warningf("unsupported patch type for mutating webhook annotation: %v", patchType)
		return
	}
	if err := w.attr.Attributes.AddAnnotationWithLevel(w.patchAnnotationKey, value, auditinternal.LevelRequest); err != nil {
		// NOTE: we don't log actual patch in kube-apiserver log to avoid potentially
		// leaking information
		klog.Warningf("failed to set patch annotation for mutating webhook key %s; confugiration name: %s, webhook name: %s", w.patchAnnotationKey, w.configuration, w.webhook)
	}
}

// MutationAuditAnnotation logs if a webhook invocation mutated the request object
type MutationAuditAnnotation struct {
	Configuration string `json:"configuration"`
	Webhook       string `json:"webhook"`
	Mutated       bool   `json:"mutated"`
}

// PatchAuditAnnotation logs a patch from a mutating webhook
type PatchAuditAnnotation struct {
	Configuration string      `json:"configuration"`
	Webhook       string      `json:"webhook"`
	Patch         interface{} `json:"patch,omitempty"`
	PatchType     string      `json:"patchType,omitempty"`
}

func mutationAnnotationValue(configuration, webhook string, mutated bool) (string, error) {
	m := MutationAuditAnnotation{
		Configuration: configuration,
		Webhook:       webhook,
		Mutated:       mutated,
	}
	bytes, err := encodingjson.Marshal(m)
	return string(bytes), err
}

func jsonPatchAnnotationValue(configuration, webhook string, patch interface{}) (string, error) {
	p := PatchAuditAnnotation{
		Configuration: configuration,
		Webhook:       webhook,
		Patch:         patch,
		PatchType:     string(admissionv1.PatchTypeJSONPatch),
	}
	bytes, err := encodingjson.Marshal(p)
	return string(bytes), err
}
