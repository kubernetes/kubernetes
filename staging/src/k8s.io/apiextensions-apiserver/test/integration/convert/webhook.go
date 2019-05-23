/*
Copyright 2019 The Kubernetes Authors.

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

package convert

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/uuid"
)

// WaitReadyFunc calls triggerConversionFn periodically and waits until it detects that the webhook
// conversion server has handled at least 1 conversion request or the timeout is exceeded, in which
// case an error is returned.
type WaitReadyFunc func(timeout time.Duration, triggerConversionFn func() error) error

// StartConversionWebhookServerWithWaitReady starts an http server with the provided handler and returns the WebhookClientConfig
// needed to configure a CRD to use this conversion webhook as its converter.
// It also returns a WaitReadyFunc to be called after the CRD is configured to wait until the conversion webhook handler
// accepts at least one conversion request. If the server fails to start, an error is returned.
// WaitReady is useful when changing the conversion webhook config of an existing CRD because the update does not take effect immediately.
func StartConversionWebhookServerWithWaitReady(handler http.Handler) (func(), *apiextensionsv1beta1.WebhookClientConfig, WaitReadyFunc, error) {
	var once sync.Once
	handlerReadyC := make(chan struct{})
	readyNotifyHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		once.Do(func() {
			close(handlerReadyC)
		})
		handler.ServeHTTP(w, r)
	})

	tearDown, webhookConfig, err := StartConversionWebhookServer(readyNotifyHandler)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("error starting webhook server: %v", err)
	}

	waitReady := func(timeout time.Duration, triggerConversionFn func() error) error {
		var err error
		for {
			select {
			case <-handlerReadyC:
				return nil
			case <-time.After(timeout):
				return fmt.Errorf("Timed out waiting for CRD webhook converter update, last trigger conversion error: %v", err)
			case <-time.After(100 * time.Millisecond):
				err = triggerConversionFn()
			}
		}
	}
	return tearDown, webhookConfig, waitReady, err
}

// StartConversionWebhookServer starts an http server with the provided handler and returns the WebhookClientConfig
// needed to configure a CRD to use this conversion webhook as its converter.
func StartConversionWebhookServer(handler http.Handler) (func(), *apiextensionsv1beta1.WebhookClientConfig, error) {
	// Use a unique path for each webhook server. This ensures that all conversion requests
	// received by the handler are intended for it; if a WebhookClientConfig other than this one
	// is applied in the api server, conversion requests will not reach the handler (if they
	// reach the server they will be returned at 404). This helps prevent tests that require a
	// specific conversion webhook from accidentally using a different one, which could otherwise
	// cause a test to flake or pass when it should fail. Since updating the conversion client
	// config of a custom resource definition does not take effect immediately, this is needed
	// by the WaitReady returned StartConversionWebhookServerWithWaitReady to detect when a
	// conversion client config change in the api server has taken effect.
	path := fmt.Sprintf("/conversionwebhook-%s", uuid.NewUUID())
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(localhostCert) {
		return nil, nil, fmt.Errorf("Failed to append Cert from PEM")
	}
	cert, err := tls.X509KeyPair(localhostCert, localhostKey)
	if err != nil {
		return nil, nil, fmt.Errorf("Failed to build cert with error: %+v", err)
	}
	webhookMux := http.NewServeMux()
	webhookMux.Handle(path, handler)
	webhookServer := httptest.NewUnstartedServer(webhookMux)
	webhookServer.TLS = &tls.Config{
		RootCAs:      roots,
		Certificates: []tls.Certificate{cert},
	}
	webhookServer.StartTLS()
	endpoint := webhookServer.URL + path
	webhookConfig := &apiextensionsv1beta1.WebhookClientConfig{
		CABundle: localhostCert,
		URL:      &endpoint,
	}
	return webhookServer.Close, webhookConfig, nil
}

// ReviewConverterFunc converts an entire ConversionReview.
type ReviewConverterFunc func(review apiextensionsv1beta1.ConversionReview) (apiextensionsv1beta1.ConversionReview, error)

// NewReviewWebhookHandler creates a handler that delegates the review conversion to the provided ReviewConverterFunc.
func NewReviewWebhookHandler(t *testing.T, converterFunc ReviewConverterFunc) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		data, err := ioutil.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
			return
		}
		if contentType := r.Header.Get("Content-Type"); contentType != "application/json" {
			t.Errorf("contentType=%s, expect application/json", contentType)
			return
		}

		review := apiextensionsv1beta1.ConversionReview{}
		if err := json.Unmarshal(data, &review); err != nil {
			t.Errorf("Fail to deserialize object: %s with error: %v", string(data), err)
			http.Error(w, err.Error(), 400)
			return
		}

		review, err = converterFunc(review)
		if err != nil {
			t.Errorf("Error converting review: %v", err)
			http.Error(w, err.Error(), 500)
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(review); err != nil {
			t.Errorf("Marshal of response failed with error: %v", err)
		}
	})
}

// ObjectConverterFunc converts a single custom resource to the desiredAPIVersion and returns it or returns an error.
type ObjectConverterFunc func(desiredAPIVersion string, customResource runtime.RawExtension) (runtime.RawExtension, error)

// NewObjectConverterWebhookHandler creates a handler that delegates custom resource conversion to the provided ConverterFunc.
func NewObjectConverterWebhookHandler(t *testing.T, converterFunc ObjectConverterFunc) http.Handler {
	return NewReviewWebhookHandler(t, func(review apiextensionsv1beta1.ConversionReview) (apiextensionsv1beta1.ConversionReview, error) {
		converted := []runtime.RawExtension{}
		errMsgs := []string{}
		for _, obj := range review.Request.Objects {
			convertedObj, err := converterFunc(review.Request.DesiredAPIVersion, obj)
			if err != nil {
				errMsgs = append(errMsgs, err.Error())
			}

			converted = append(converted, convertedObj)
		}

		review.Response = &apiextensionsv1beta1.ConversionResponse{
			UID:              review.Request.UID,
			ConvertedObjects: converted,
		}
		if len(errMsgs) == 0 {
			review.Response.Result = metav1.Status{Status: "Success"}
		} else {
			review.Response.Result = metav1.Status{Status: "Failure", Message: strings.Join(errMsgs, ", ")}
		}
		return review, nil
	})
}

// localhostCert was generated from crypto/tls/generate_cert.go with the following command:
//     go run generate_cert.go  --rsa-bits 512 --host 127.0.0.1,::1,example.com --ca --start-date "Jan 1 00:00:00 1970" --duration=1000000h
var localhostCert = []byte(`-----BEGIN CERTIFICATE-----
MIIBjzCCATmgAwIBAgIRAKpi2WmTcFrVjxrl5n5YDUEwDQYJKoZIhvcNAQELBQAw
EjEQMA4GA1UEChMHQWNtZSBDbzAgFw03MDAxMDEwMDAwMDBaGA8yMDg0MDEyOTE2
MDAwMFowEjEQMA4GA1UEChMHQWNtZSBDbzBcMA0GCSqGSIb3DQEBAQUAA0sAMEgC
QQC9fEbRszP3t14Gr4oahV7zFObBI4TfA5i7YnlMXeLinb7MnvT4bkfOJzE6zktn
59zP7UiHs3l4YOuqrjiwM413AgMBAAGjaDBmMA4GA1UdDwEB/wQEAwICpDATBgNV
HSUEDDAKBggrBgEFBQcDATAPBgNVHRMBAf8EBTADAQH/MC4GA1UdEQQnMCWCC2V4
YW1wbGUuY29thwR/AAABhxAAAAAAAAAAAAAAAAAAAAABMA0GCSqGSIb3DQEBCwUA
A0EAUsVE6KMnza/ZbodLlyeMzdo7EM/5nb5ywyOxgIOCf0OOLHsPS9ueGLQX9HEG
//yjTXuhNcUugExIjM/AIwAZPQ==
-----END CERTIFICATE-----`)

// localhostKey is the private key for localhostCert.
var localhostKey = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIIBOwIBAAJBAL18RtGzM/e3XgavihqFXvMU5sEjhN8DmLtieUxd4uKdvsye9Phu
R84nMTrOS2fn3M/tSIezeXhg66quOLAzjXcCAwEAAQJBAKcRxH9wuglYLBdI/0OT
BLzfWPZCEw1vZmMR2FF1Fm8nkNOVDPleeVGTWoOEcYYlQbpTmkGSxJ6ya+hqRi6x
goECIQDx3+X49fwpL6B5qpJIJMyZBSCuMhH4B7JevhGGFENi3wIhAMiNJN5Q3UkL
IuSvv03kaPR5XVQ99/UeEetUgGvBcABpAiBJSBzVITIVCGkGc7d+RCf49KTCIklv
bGWObufAR8Ni4QIgWpILjW8dkGg8GOUZ0zaNA6Nvt6TIv2UWGJ4v5PoV98kCIQDx
rIiZs5QbKdycsv9gQJzwQAogC8o04X3Zz3dsoX+h4A==
-----END RSA PRIVATE KEY-----`)
