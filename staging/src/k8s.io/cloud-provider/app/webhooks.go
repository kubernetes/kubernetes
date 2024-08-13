/*
Copyright 2022 The Kubernetes Authors.

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

package app

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"strconv"
	"time"

	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/server/mux"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/cloud-provider/app/config"
	genericcontrollermanager "k8s.io/controller-manager/app"
	"k8s.io/klog/v2"
)

var (
	runtimeScheme = runtime.NewScheme()
	codecs        = serializer.NewCodecFactory(runtimeScheme)
	deserializer  = codecs.UniversalDeserializer()
	encoder       runtime.Encoder
)

func init() {
	_ = corev1.AddToScheme(runtimeScheme)
	_ = admissionv1.AddToScheme(runtimeScheme)
	serializerInfo, _ := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
	encoder = serializerInfo.Serializer
}

// WebhooksDisabledByDefault is the webhooks disabled default when starting cloud-controller managers.
var WebhooksDisabledByDefault = sets.NewString()

type WebhookConfig struct {
	Path             string
	AdmissionHandler func(*admissionv1.AdmissionRequest) (*admissionv1.AdmissionResponse, error)
}

type WebhookHandler struct {
	Name string
	Path string
	http.Handler
	AdmissionHandler func(*admissionv1.AdmissionRequest) (*admissionv1.AdmissionResponse, error)
	CompletedConfig  *config.CompletedConfig
	Cloud            cloudprovider.Interface
}

func NewWebhookHandlers(webhookConfigs map[string]WebhookConfig, completedConfig *config.CompletedConfig, cloud cloudprovider.Interface) map[string]WebhookHandler {
	webhookHandlers := make(map[string]WebhookHandler)
	for name, config := range webhookConfigs {
		if !genericcontrollermanager.IsControllerEnabled(name, WebhooksDisabledByDefault, completedConfig.ComponentConfig.Webhook.Webhooks) {
			klog.Warningf("Webhook %q is disabled", name)
			continue
		}
		klog.Infof("Webhook enabled: %q", name)
		webhookHandlers[name] = WebhookHandler{
			Name:             name,
			Path:             config.Path,
			AdmissionHandler: config.AdmissionHandler,
			CompletedConfig:  completedConfig,
			Cloud:            cloud,
		}
	}
	return webhookHandlers
}

func WebhookNames(webhooks map[string]WebhookConfig) []string {
	ret := sets.StringKeySet(webhooks)
	return ret.List()
}

func newHandler(webhooks map[string]WebhookHandler) *mux.PathRecorderMux {
	mux := mux.NewPathRecorderMux("controller-manager-webhook")

	for _, handler := range webhooks {
		mux.Handle(handler.Path, handler)
	}

	return mux
}

func (h WebhookHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	ctx := context.Background()
	klog.Infof("Received validation request: %q", r.RequestURI)

	start := time.Now()
	var (
		statusCode        int
		err               error
		in                *admissionv1.AdmissionReview
		admissionResponse *admissionv1.AdmissionResponse
	)
	defer func() {
		latency := time.Since(start)

		if statusCode != 0 {
			recordRequestTotal(ctx, strconv.Itoa(statusCode), h.Name)
			recordRequestLatency(ctx, strconv.Itoa(statusCode), h.Name, latency.Seconds())
			return
		}

		if err != nil {
			recordRequestTotal(ctx, "<error>", h.Name)
			recordRequestLatency(ctx, "<error>", h.Name, latency.Seconds())
		}
	}()

	in, err = parseRequest(r)
	if err != nil {
		klog.Error(err)
		statusCode = http.StatusBadRequest
		http.Error(w, err.Error(), statusCode)
		return
	}

	admissionResponse, err = h.AdmissionHandler(in.Request)
	if err != nil {
		e := fmt.Sprintf("error generating admission response: %v", err)
		klog.Errorf(e)
		statusCode = http.StatusInternalServerError
		http.Error(w, e, statusCode)
		return
	} else if admissionResponse == nil {
		e := fmt.Sprintf("admission response cannot be nil")
		klog.Error(e)
		statusCode = http.StatusInternalServerError
		http.Error(w, e, statusCode)
		return
	}

	admissionReview := &admissionv1.AdmissionReview{
		Response: admissionResponse,
	}
	admissionReview.Response.UID = in.Request.UID
	w.Header().Set("Content-Type", "application/json")

	codec := codecs.EncoderForVersion(encoder, admissionv1.SchemeGroupVersion)
	out, err := runtime.Encode(codec, admissionReview)
	if err != nil {
		e := fmt.Sprintf("error parsing admission response: %v", err)
		klog.Error(e)
		statusCode = http.StatusInternalServerError
		http.Error(w, e, statusCode)
		return
	}

	klog.Infof("%s", out)
	fmt.Fprintf(w, "%s", out)
}

// parseRequest extracts an AdmissionReview from an http.Request if possible
func parseRequest(r *http.Request) (*admissionv1.AdmissionReview, error) {

	review := &admissionv1.AdmissionReview{}

	if r.Header.Get("Content-Type") != "application/json" {
		return nil, fmt.Errorf("Content-Type: %q should be %q",
			r.Header.Get("Content-Type"), "application/json")
	}

	bodybuf := new(bytes.Buffer)
	bodybuf.ReadFrom(r.Body)
	body := bodybuf.Bytes()

	if len(body) == 0 {
		return nil, fmt.Errorf("admission request HTTP body is empty")
	}

	if _, _, err := deserializer.Decode(body, nil, review); err != nil {
		return nil, fmt.Errorf("could not deserialize incoming admission review: %v", err)
	}

	if review.Request == nil {
		return nil, fmt.Errorf("admission review can't be used: Request field is nil")
	}

	return review, nil
}
