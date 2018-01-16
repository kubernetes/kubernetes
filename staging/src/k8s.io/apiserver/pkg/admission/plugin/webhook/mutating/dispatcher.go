/*
Copyright 2017 The Kubernetes Authors.

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
	"time"

	jsonpatch "github.com/evanphx/json-patch"
	"github.com/golang/glog"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/config"

	admissionv1beta1 "k8s.io/api/admission/v1beta1"
	"k8s.io/api/admissionregistration/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	admissionmetrics "k8s.io/apiserver/pkg/admission/metrics"
	webhookerrors "k8s.io/apiserver/pkg/admission/plugin/webhook/errors"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/generic"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/request"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/versioned"
)

type mutatingDispatcher struct {
	cm     *config.ClientManager
	plugin *Plugin
}

func newMutatingDispatcher(p *Plugin) func(cm *config.ClientManager) generic.Dispatcher {
	return func(cm *config.ClientManager) generic.Dispatcher {
		return &mutatingDispatcher{cm, p}
	}
}

var _ generic.Dispatcher = &mutatingDispatcher{}

func (a *mutatingDispatcher) Dispatch(ctx context.Context, versionedAttr *versioned.Attributes, relevantHooks []*v1beta1.Webhook) error {
	for _, hook := range relevantHooks {
		t := time.Now()
		err := a.callAttrMutatingHook(ctx, hook, versionedAttr)
		admissionmetrics.Metrics.ObserveWebhook(time.Since(t), err != nil, versionedAttr.Attributes, "admit", hook.Name)
		if err == nil {
			continue
		}

		ignoreClientCallFailures := hook.FailurePolicy != nil && *hook.FailurePolicy == v1beta1.Ignore
		if callErr, ok := err.(*webhookerrors.ErrCallingWebhook); ok {
			if ignoreClientCallFailures {
				glog.Warningf("Failed calling webhook, failing open %v: %v", hook.Name, callErr)
				utilruntime.HandleError(callErr)
				continue
			}
			glog.Warningf("Failed calling webhook, failing closed %v: %v", hook.Name, err)
		}
		return apierrors.NewInternalError(err)
	}

	// convert attr.Object to the internal version
	return a.plugin.convertor.Convert(versionedAttr.Object, versionedAttr.Attributes.GetObject())
}

// note that callAttrMutatingHook updates attr
func (a *mutatingDispatcher) callAttrMutatingHook(ctx context.Context, h *v1beta1.Webhook, attr *versioned.Attributes) error {
	// Make the webhook request
	request := request.CreateAdmissionReview(attr)
	client, err := a.cm.HookClient(h)
	if err != nil {
		return &webhookerrors.ErrCallingWebhook{WebhookName: h.Name, Reason: err}
	}
	response := &admissionv1beta1.AdmissionReview{}
	if err := client.Post().Context(ctx).Body(&request).Do().Into(response); err != nil {
		return &webhookerrors.ErrCallingWebhook{WebhookName: h.Name, Reason: err}
	}

	if !response.Response.Allowed {
		return webhookerrors.ToStatusErr(h.Name, response.Response.Result)
	}

	patchJS := response.Response.Patch
	if len(patchJS) == 0 {
		return nil
	}
	patchObj, err := jsonpatch.DecodePatch(patchJS)
	if err != nil {
		return apierrors.NewInternalError(err)
	}
	objJS, err := runtime.Encode(a.plugin.jsonSerializer, attr.Object)
	if err != nil {
		return apierrors.NewInternalError(err)
	}
	patchedJS, err := patchObj.Apply(objJS)
	if err != nil {
		return apierrors.NewInternalError(err)
	}
	// TODO: if we have multiple mutating webhooks, we can remember the json
	// instead of encoding and decoding for each one.
	if _, _, err := a.plugin.jsonSerializer.Decode(patchedJS, nil, attr.Object); err != nil {
		return apierrors.NewInternalError(err)
	}
	a.plugin.defaulter.Default(attr.Object)
	return nil
}
