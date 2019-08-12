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

package validating

import (
	"context"
	"fmt"
	"sync"
	"time"

	"k8s.io/api/admissionregistration/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/admission"
	admissionmetrics "k8s.io/apiserver/pkg/admission/metrics"
	"k8s.io/apiserver/pkg/admission/plugin/webhook"
	webhookerrors "k8s.io/apiserver/pkg/admission/plugin/webhook/errors"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/generic"
	webhookrequest "k8s.io/apiserver/pkg/admission/plugin/webhook/request"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/util"
	webhookutil "k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/klog"
	utiltrace "k8s.io/utils/trace"
)

type validatingDispatcher struct {
	cm     *webhookutil.ClientManager
	plugin *Plugin
}

func newValidatingDispatcher(p *Plugin) func(cm *webhookutil.ClientManager) generic.Dispatcher {
	return func(cm *webhookutil.ClientManager) generic.Dispatcher {
		return &validatingDispatcher{cm, p}
	}
}

var _ generic.Dispatcher = &validatingDispatcher{}

func (d *validatingDispatcher) Dispatch(ctx context.Context, attr admission.Attributes, o admission.ObjectInterfaces, hooks []webhook.WebhookAccessor) error {
	var relevantHooks []*generic.WebhookInvocation
	// Construct all the versions we need to call our webhooks
	versionedAttrs := map[schema.GroupVersionKind]*generic.VersionedAttributes{}
	for _, hook := range hooks {
		invocation, statusError := d.plugin.ShouldCallHook(hook, attr, o)
		if statusError != nil {
			return statusError
		}
		if invocation == nil {
			continue
		}
		relevantHooks = append(relevantHooks, invocation)
		// If we already have this version, continue
		if _, ok := versionedAttrs[invocation.Kind]; ok {
			continue
		}
		versionedAttr, err := generic.NewVersionedAttributes(attr, invocation.Kind, o)
		if err != nil {
			return apierrors.NewInternalError(err)
		}
		versionedAttrs[invocation.Kind] = versionedAttr
	}

	if len(relevantHooks) == 0 {
		// no matching hooks
		return nil
	}

	wg := sync.WaitGroup{}
	errCh := make(chan error, len(relevantHooks))
	wg.Add(len(relevantHooks))
	for i := range relevantHooks {
		go func(invocation *generic.WebhookInvocation) {
			defer wg.Done()
			hook, ok := invocation.Webhook.GetValidatingWebhook()
			if !ok {
				utilruntime.HandleError(fmt.Errorf("validating webhook dispatch requires v1beta1.ValidatingWebhook, but got %T", hook))
				return
			}
			versionedAttr := versionedAttrs[invocation.Kind]
			t := time.Now()
			err := d.callHook(ctx, hook, invocation, versionedAttr)
			admissionmetrics.Metrics.ObserveWebhook(time.Since(t), err != nil, versionedAttr.Attributes, "validating", hook.Name)
			if err == nil {
				return
			}

			ignoreClientCallFailures := hook.FailurePolicy != nil && *hook.FailurePolicy == v1beta1.Ignore
			if callErr, ok := err.(*webhookutil.ErrCallingWebhook); ok {
				if ignoreClientCallFailures {
					klog.Warningf("Failed calling webhook, failing open %v: %v", hook.Name, callErr)
					utilruntime.HandleError(callErr)
					return
				}

				klog.Warningf("Failed calling webhook, failing closed %v: %v", hook.Name, err)
				errCh <- apierrors.NewInternalError(err)
				return
			}

			klog.Warningf("rejected by webhook %q: %#v", hook.Name, err)
			errCh <- err
		}(relevantHooks[i])
	}
	wg.Wait()
	close(errCh)

	var errs []error
	for e := range errCh {
		errs = append(errs, e)
	}
	if len(errs) == 0 {
		return nil
	}
	if len(errs) > 1 {
		for i := 1; i < len(errs); i++ {
			// TODO: merge status errors; until then, just return the first one.
			utilruntime.HandleError(errs[i])
		}
	}
	return errs[0]
}

func (d *validatingDispatcher) callHook(ctx context.Context, h *v1beta1.ValidatingWebhook, invocation *generic.WebhookInvocation, attr *generic.VersionedAttributes) error {
	if attr.Attributes.IsDryRun() {
		if h.SideEffects == nil {
			return &webhookutil.ErrCallingWebhook{WebhookName: h.Name, Reason: fmt.Errorf("Webhook SideEffects is nil")}
		}
		if !(*h.SideEffects == v1beta1.SideEffectClassNone || *h.SideEffects == v1beta1.SideEffectClassNoneOnDryRun) {
			return webhookerrors.NewDryRunUnsupportedErr(h.Name)
		}
	}

	uid, request, response, err := webhookrequest.CreateAdmissionObjects(attr, invocation)
	if err != nil {
		return &webhookutil.ErrCallingWebhook{WebhookName: h.Name, Reason: err}
	}
	// Make the webhook request
	client, err := d.cm.HookClient(util.HookClientConfigForWebhook(invocation.Webhook))
	if err != nil {
		return &webhookutil.ErrCallingWebhook{WebhookName: h.Name, Reason: err}
	}
	trace := utiltrace.New("Call validating webhook",
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
		return &webhookutil.ErrCallingWebhook{WebhookName: h.Name, Reason: err}
	}
	trace.Step("Request completed")

	result, err := webhookrequest.VerifyAdmissionResponse(uid, false, response)
	if err != nil {
		return &webhookutil.ErrCallingWebhook{WebhookName: h.Name, Reason: err}
	}

	for k, v := range result.AuditAnnotations {
		key := h.Name + "/" + k
		if err := attr.Attributes.AddAnnotation(key, v); err != nil {
			klog.Warningf("Failed to set admission audit annotation %s to %s for validating webhook %s: %v", key, v, h.Name, err)
		}
	}
	if result.Allowed {
		return nil
	}
	return webhookerrors.ToStatusErr(h.Name, result.Result)
}
