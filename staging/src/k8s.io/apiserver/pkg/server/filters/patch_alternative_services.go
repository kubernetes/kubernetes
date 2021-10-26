/*
Copyright 2021 The Kubernetes Authors.

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
	"fmt"
	"net"
	"net/http"
	"sync"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/informers"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
)

type AlternativeServicesInfo struct {
	endpointsLister corelisters.EndpointsLister
	endpointsSynced func() bool
	// generated Alt-Svc header
	mu           sync.Mutex
	altSvcHeader string
}

func (a *AlternativeServicesInfo) sync() {
	endpoint, err := a.endpointsLister.Endpoints("default").Get("kubernetes")
	if err != nil {
		a.set("")
		return
	}

	ips := getEndpointIPs(endpoint)
	// only publish alternative services if there are 2 or more api servers
	if len(ips) < 2 {
		a.set("")
		return
	}

	// Generate Alt-Svc header
	// ref: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Alt-Svc
	// Example: h2="10.0.0.2:6443", h2="10.0.0.3:6443", h2="10.0.0.4:6443"
	var hdr string
	for i, a := range ips {
		if i != 0 {
			hdr += ", "
		}
		hdr += fmt.Sprintf(`h2="%s"`, net.JoinHostPort(a, "6443"))
	}
	a.set(hdr)
}

func (a *AlternativeServicesInfo) set(header string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.altSvcHeader = header
}

func (a *AlternativeServicesInfo) get() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.altSvcHeader
}

func NewAlternativeServerInfo(factory informers.SharedInformerFactory) *AlternativeServicesInfo {
	a := &AlternativeServicesInfo{}
	if factory == nil {
		a.endpointsSynced = func() bool { return false }
		return a
	}

	informer := factory.Core().V1().Endpoints()
	informer.Informer().GetController()
	informer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			key, err := cache.MetaNamespaceKeyFunc(obj)
			if err == nil && key == "default/kubernetes" {
				a.sync()
			}
		},
		UpdateFunc: func(old, cur interface{}) {
			key, err := cache.MetaNamespaceKeyFunc(cur)
			if err == nil && key == "default/kubernetes" {
				a.sync()
			}
		},
		DeleteFunc: func(obj interface{}) {
			key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
			if err == nil && key == "default/kubernetes" {
				a.sync()
			}
		},
	})

	a.endpointsLister = informer.Lister()
	a.endpointsSynced = informer.Informer().HasSynced
	return a
}

// WithAternativeServices sets the Alt-Svc header based on the available api servers
// See RFC7838
func WithAternativeServices(handler http.Handler, a *AlternativeServicesInfo) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if !a.endpointsSynced() {
			handler.ServeHTTP(w, req)
			return
		}

		// https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Alt-Svc
		hdr := a.get()
		if len(hdr) > 0 {
			w.Header().Set("Alt-Svc", hdr)
		}
		handler.ServeHTTP(w, req)
	})
}

// return the unique endpoint IPs
func getEndpointIPs(endpoints *corev1.Endpoints) []string {
	endpointMap := make(map[string]bool)
	ips := make([]string, 0)
	if endpoints == nil || len(endpoints.Subsets) == 0 {
		return ips
	}
	for _, subset := range endpoints.Subsets {
		for _, address := range subset.Addresses {
			if _, ok := endpointMap[address.IP]; !ok {
				endpointMap[address.IP] = true
				ips = append(ips, address.IP)
			}
		}
	}
	return ips
}
