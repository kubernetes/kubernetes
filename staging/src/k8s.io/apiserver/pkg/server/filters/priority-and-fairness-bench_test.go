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
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"k8s.io/api/flowcontrol/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	fcboot "k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
	"k8s.io/apiserver/pkg/authentication/user"
	apifilters "k8s.io/apiserver/pkg/endpoints/filters"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilflowcontrol "k8s.io/apiserver/pkg/util/flowcontrol"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
)

func BenchmarkAPFFilter(b *testing.B) {
	const testConcurrencyLimit = 100
	const parallel = false
	apfConfig := []runtime.Object{}
	for _, fs := range append(append([]*v1beta1.FlowSchema{}, fcboot.SuggestedFlowSchemas...), fcboot.MandatoryFlowSchemas...) {
		apfConfig = append(apfConfig, fs)
	}
	for _, plc := range append(append([]*v1beta1.PriorityLevelConfiguration{}, fcboot.SuggestedPriorityLevelConfigurations...), fcboot.MandatoryPriorityLevelConfigurations...) {
		apfConfig = append(apfConfig, plc)
	}
	cs := clientsetfake.NewSimpleClientset(apfConfig...)
	informerFactory := informers.NewSharedInformerFactory(cs, 0)
	flowcontrolClient := cs.FlowcontrolV1beta1()
	apfIfc := utilflowcontrol.New(informerFactory, flowcontrolClient, testConcurrencyLimit, time.Minute)
	longRunningCheck := BasicLongRunningRequestCheck(
		sets.NewString("watch", "proxy"),
		sets.NewString("attach", "exec", "proxy", "log", "portforward"),
	)
	innerHandler := http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		rw.WriteHeader(http.StatusOK)
	})

	apfHandler := WithPriorityAndFairness(innerHandler, longRunningCheck, apfIfc)
	requestInfoFactory := &apirequest.RequestInfoFactory{APIPrefixes: sets.NewString("apis", "api"), GrouplessAPIPrefixes: sets.NewString("api")}
	handler := apifilters.WithRequestInfo(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r = r.WithContext(apirequest.WithUser(r.Context(), &user.DefaultInfo{
			Name:   "Bond",
			UID:    "007",
			Groups: []string{user.AllUnauthenticated},
		}))
		apfHandler.ServeHTTP(w, r)
	}), requestInfoFactory)

	b.ResetTimer()
	if parallel {
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				req := httptest.NewRequest("GET", "/healthz", nil)
				rr := httptest.NewRecorder()
				handler.ServeHTTP(rr, req)
				if rr.Code != http.StatusOK {
					b.Errorf("Got status code %d instead of %d", rr.Code, http.StatusOK)
				}
			}
		})
	} else {
		for i := 0; i < b.N; i++ {
			req := httptest.NewRequest("GET", "/healthz", nil)
			rr := httptest.NewRecorder()
			handler.ServeHTTP(rr, req)
			if rr.Code != http.StatusOK {
				b.Errorf("Got status code %d instead of %d", rr.Code, http.StatusOK)
			}
		}
	}
}
