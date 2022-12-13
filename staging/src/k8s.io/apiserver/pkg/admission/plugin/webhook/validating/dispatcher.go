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

	"go.opentelemetry.io/otel/attribute"

	v1 "k8s.io/api/admissionregistration/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/admission"
	admissionmetrics "k8s.io/apiserver/pkg/admission/metrics"
	"k8s.io/apiserver/pkg/admission/plugin/webhook"
	webhookerrors "k8s.io/apiserver/pkg/admission/plugin/webhook/errors"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/generic"
	webhookrequest "k8s.io/apiserver/pkg/admission/plugin/webhook/request"
	endpointsrequest "k8s.io/apiserver/pkg/endpoints/request"
	webhookutil "k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/apiserver/pkg/warning"
	"k8s.io/component-base/tracing"
	"k8s.io/klog/v2"
)

const (
	// ValidatingAuditAnnotationPrefix is a prefix for keeping noteworthy
	// validating audit annotations.
	ValidatingAuditAnnotationPrefix = "validating.webhook.admission.k8s.io/"
	// ValidatingAuditAnnotationFailedOpenKeyPrefix in an annotation indicates
	// the validating webhook failed open when the webhook backend connection
	// failed or returned an internal server error.
	ValidatingAuditAnnotationFailedOpenKeyPrefix = "failed-open." + ValidatingAuditAnnotationPrefix
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

	// Check if the request has already timed out before spawning remote calls
	select {
	case <-ctx.Done():
		// parent context is canceled or timed out, no point in continuing
		return apierrors.NewTimeoutError("request did not complete within requested timeout", 0)
	default:
	}

	wg := sync.WaitGroup{}
	errCh := make(chan error, 2*len(relevantHooks)) // double the length to handle extra errors for panics in the gofunc
	wg.Add(len(relevantHooks))
	for i := range relevantHooks {
		go func(invocation *generic.WebhookInvocation, idx int) {
			ignoreClientCallFailures := false
			hookName := "unknown"
			versionedAttr := versionedAttrs[invocation.Kind]
			// The ordering of these two defers is critical. The wg.Done will release the parent go func to close the errCh
			// that is used by the second defer to report errors. The recovery and error reporting must be done first.
			defer wg.Done()
			defer func() {
				// HandleCrash has already called the crash handlers and it has been configured to utilruntime.ReallyCrash
				// This block prevents the second panic from failing our process.
				// This failure mode for the handler functions properly using the channel below.
				recover()
			}()
			defer utilruntime.HandleCrash(
				func(r interface{}) {
					if r == nil {
						return
					}
					if ignoreClientCallFailures {
						// if failures are supposed to ignored, ignore it
						klog.Warningf("Panic calling webhook, failing open %v: %v", hookName, r)
						admissionmetrics.Metrics.ObserveWebhookFailOpen(ctx, hookName, "validating")
						key := fmt.Sprintf("%sround_0_index_%d", ValidatingAuditAnnotationFailedOpenKeyPrefix, idx)
						value := hookName
						if err := versionedAttr.Attributes.AddAnnotation(key, value); err != nil {
							klog.Warningf("Failed to set admission audit annotation %s to %s for validating webhook %s: %v", key, value, hookName, err)
						}
						return
					}
					// this ensures that the admission request fails and a message is provided.
					errCh <- apierrors.NewInternalError(fmt.Errorf("ValidatingAdmissionWebhook/%v has panicked: %v", hookName, r))
				},
			)

			hook, ok := invocation.Webhook.GetValidatingWebhook()
			if !ok {
				utilruntime.HandleError(fmt.Errorf("validating webhook dispatch requires v1.ValidatingWebhook, but got %T", hook))
				return
			}
			hookName = hook.Name
			ignoreClientCallFailures = hook.FailurePolicy != nil && *hook.FailurePolicy == v1.Ignore
			t := time.Now()
			err := d.callHook(ctx, hook, invocation, versionedAttr)
			rejected := false
			if err != nil {
				switch err := err.(type) {
				case *webhookutil.ErrCallingWebhook:
					if !ignoreClientCallFailures {
						rejected = true
						admissionmetrics.Metrics.ObserveWebhookRejection(ctx, hook.Name, "validating", string(versionedAttr.Attributes.GetOperation()), admissionmetrics.WebhookRejectionCallingWebhookError, int(err.Status.ErrStatus.Code))
					}
					admissionmetrics.Metrics.ObserveWebhook(ctx, hook.Name, time.Since(t), rejected, versionedAttr.Attributes, "validating", int(err.Status.ErrStatus.Code))
				case *webhookutil.ErrWebhookRejection:
					rejected = true
					admissionmetrics.Metrics.ObserveWebhookRejection(ctx, hook.Name, "validating", string(versionedAttr.Attributes.GetOperation()), admissionmetrics.WebhookRejectionNoError, int(err.Status.ErrStatus.Code))
					admissionmetrics.Metrics.ObserveWebhook(ctx, hook.Name, time.Since(t), rejected, versionedAttr.Attributes, "validating", int(err.Status.ErrStatus.Code))
				default:
					rejected = true
					admissionmetrics.Metrics.ObserveWebhookRejection(ctx, hook.Name, "validating", string(versionedAttr.Attributes.GetOperation()), admissionmetrics.WebhookRejectionAPIServerInternalError, 0)
					admissionmetrics.Metrics.ObserveWebhook(ctx, hook.Name, time.Since(t), rejected, versionedAttr.Attributes, "validating", 0)
				}
			} else {
				admissionmetrics.Metrics.ObserveWebhook(ctx, hook.Name, time.Since(t), rejected, versionedAttr.Attributes, "validating", 200)
				return
			}

			if callErr, ok := err.(*webhookutil.ErrCallingWebhook); ok {
				if ignoreClientCallFailures {
					klog.Warningf("Failed calling webhook, failing open %v: %v", hook.Name, callErr)
					admissionmetrics.Metrics.ObserveWebhookFailOpen(ctx, hook.Name, "validating")
					key := fmt.Sprintf("%sround_0_index_%d", ValidatingAuditAnnotationFailedOpenKeyPrefix, idx)
					value := hook.Name
					if err := versionedAttr.Attributes.AddAnnotation(key, value); err != nil {
						klog.Warningf("Failed to set admission audit annotation %s to %s for validating webhook %s: %v", key, value, hook.Name, err)
					}
					utilruntime.HandleError(callErr)
					return
				}

				klog.Warningf("Failed calling webhook, failing closed %v: %v", hook.Name, err)
				errCh <- apierrors.NewInternalError(err)
				return
			}

			if rejectionErr, ok := err.(*webhookutil.ErrWebhookRejection); ok {
				err = rejectionErr.Status
			}
			klog.Warningf("rejected by webhook %q: %#v", hook.Name, err)
			errCh <- err
		}(relevantHooks[i], i)
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

func (d *validatingDispatcher) callHook(ctx context.Context, h *v1.ValidatingWebhook, invocation *generic.WebhookInvocation, attr *generic.VersionedAttributes) error {
	if attr.Attributes.IsDryRun() {
		if h.SideEffects == nil {
			return &webhookutil.ErrCallingWebhook{WebhookName: h.Name, Reason: fmt.Errorf("Webhook SideEffects is nil"), Status: apierrors.NewBadRequest("Webhook SideEffects is nil")}
		}
		if !(*h.SideEffects == v1.SideEffectClassNone || *h.SideEffects == v1.SideEffectClassNoneOnDryRun) {
			return webhookerrors.NewDryRunUnsupportedErr(h.Name)
		}
	}

	uid, request, response, err := webhookrequest.CreateAdmissionObjects(attr, invocation)
	if err != nil {
		return &webhookutil.ErrCallingWebhook{WebhookName: h.Name, Reason: fmt.Errorf("could not create admission objects: %w", err), Status: apierrors.NewBadRequest("error creating admission objects")}
	}
	// Make the webhook request
	client, err := invocation.Webhook.GetRESTClient(d.cm)
	if err != nil {
		return &webhookutil.ErrCallingWebhook{WebhookName: h.Name, Reason: fmt.Errorf("could not get REST client: %w", err), Status: apierrors.NewBadRequest("error getting REST client")}
	}
	ctx, span := tracing.Start(ctx, "Call validating webhook",
		attribute.String("configuration", invocation.Webhook.GetConfigurationName()),
		attribute.String("webhook", h.Name),
		attribute.Stringer("resource", attr.GetResource()),
		attribute.String("subresource", attr.GetSubresource()),
		attribute.String("operation", string(attr.GetOperation())),
		attribute.String("UID", string(uid)))
	defer span.End(500 * time.Millisecond)

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

	do := func() { err = r.Do(ctx).Into(response) }
	if wd, ok := endpointsrequest.LatencyTrackersFrom(ctx); ok {
		tmp := do
		do = func() { wd.ValidatingWebhookTracker.Track(tmp) }
	}
	do()
	if err != nil {
		var status *apierrors.StatusError
		if se, ok := err.(*apierrors.StatusError); ok {
			status = se
		} else {
			status = apierrors.NewBadRequest("error calling webhook")
		}
		return &webhookutil.ErrCallingWebhook{WebhookName: h.Name, Reason: fmt.Errorf("failed to call webhook: %w", err), Status: status}
	}
	span.AddEvent("Request completed")

	result, err := webhookrequest.VerifyAdmissionResponse(uid, false, response)
	if err != nil {
		return &webhookutil.ErrCallingWebhook{WebhookName: h.Name, Reason: fmt.Errorf("received invalid webhook response: %w", err), Status: apierrors.NewServiceUnavailable("error validating webhook response")}
	}

	for k, v := range result.AuditAnnotations {
		key := h.Name + "/" + k
		if err := attr.Attributes.AddAnnotation(key, v); err != nil {
			klog.Warningf("Failed to set admission audit annotation %s to %s for validating webhook %s: %v", key, v, h.Name, err)
		}
	}
	for _, w := range result.Warnings {
		warning.AddWarning(ctx, "", w)
	}
	if result.Allowed {
		return nil
	}
	return &webhookutil.ErrWebhookRejection{Status: webhookerrors.ToStatusErr(h.Name, result.Result)}
}
