/*
Copyright 2016 The Kubernetes Authors.

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

package filters

import (
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/sets"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	genericapifilters "k8s.io/kubernetes/pkg/genericapiserver/endpoints/filters"
)

func TestTimeout(t *testing.T) {
	sendResponse := make(chan struct{}, 1)
	writeErrors := make(chan error, 1)
	resp := "test response"
	timeoutErr := apierrors.NewServerTimeout(schema.GroupResource{Group: "foo", Resource: "bar"}, "get", 0)
	mapper := apirequest.NewRequestContextMapper()
	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)

	scheme.AddKnownTypes(schema.GroupVersion{Group: "", Version: "v1"}, &metav1.Status{})
	scheme.AddKnownTypes(schema.GroupVersion{Group: "core", Version: "v1"}, &metav1.Status{})

	var handler http.Handler
	handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-sendResponse
		_, err := w.Write([]byte(resp))
		writeErrors <- err
	})
	handler = WithTimeout(handler, time.Second, mapper, codecs)
	handler = genericapifilters.WithRequestInfo(handler, newTestRequestInfoResolver(), mapper)
	handler = apirequest.WithRequestContext(handler, mapper)

	ts := httptest.NewServer(handler)
	defer ts.Close()

	// No timeouts
	sendResponse <- struct{}{}
	res, err := http.Get(ts.URL)
	if err != nil {
		t.Error(err)
	}
	if res.StatusCode != http.StatusOK {
		t.Errorf("got res.StatusCode %d; expected %d", res.StatusCode, http.StatusOK)
	}
	body, _ := ioutil.ReadAll(res.Body)
	if string(body) != resp {
		t.Errorf("got body %q; expected %q", string(body), resp)
	}
	if err := <-writeErrors; err != nil {
		t.Errorf("got unexpected Write error on first request: %v", err)
	}

	// Times out without version
	res, err = http.Get(ts.URL)
	if err != nil {
		t.Error(err)
	}
	if res.StatusCode != http.StatusGatewayTimeout {
		t.Errorf("got res.StatusCode %d; expected %d", res.StatusCode, http.StatusServiceUnavailable)
	}
	body, _ = ioutil.ReadAll(res.Body)
	if !strings.Contains(string(body), timeoutErr.Error()) {
		t.Errorf("got body %q; expected it to contain %q", string(body), timeoutErr.Error())
	}

	// Now try to send a response
	sendResponse <- struct{}{}
	if err := <-writeErrors; err != http.ErrHandlerTimeout {
		t.Errorf("got Write error of %v; expected %v", err, http.ErrHandlerTimeout)
	}

	// Times out _with_ version
	res, err = http.Get(ts.URL + "/apis/core/v1/pods")
	if err != nil {
		t.Error(err)
	}
	if res.StatusCode != http.StatusGatewayTimeout {
		t.Errorf("got res.StatusCode %d; expected %d", res.StatusCode, http.StatusServiceUnavailable)
	}
	body, _ = ioutil.ReadAll(res.Body)
	if !strings.Contains(string(body), timeoutErr.Error()) {
		t.Errorf("got body %q; expected it to contain %q", string(body), timeoutErr.Error())
	}

	// Now try to send a response
	sendResponse <- struct{}{}
	if err := <-writeErrors; err != http.ErrHandlerTimeout {
		t.Errorf("got Write error of %v; expected %v", err, http.ErrHandlerTimeout)
	}
}

func newTestRequestInfoResolver() *apirequest.RequestInfoFactory {
	return &apirequest.RequestInfoFactory{
		APIPrefixes:          sets.NewString("api", "apis"),
		GrouplessAPIPrefixes: sets.NewString("api"),
	}
}
