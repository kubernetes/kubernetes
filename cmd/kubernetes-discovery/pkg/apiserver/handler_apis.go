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

package apiserver

import (
	"net/http"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apiserver"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"

	apiregistrationapi "k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/apis/apiregistration"
	apiregistrationv1alpha1api "k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/apis/apiregistration/v1alpha1"
	informers "k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/client/informers/apiregistration/internalversion"
	listers "k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/client/listers/apiregistration/internalversion"
)

// WithAPIs adds the handling for /apis and /apis/<group: -apiregistration.k8s.io>.
func WithAPIs(handler http.Handler, informer informers.APIServiceInformer) http.Handler {
	apisHandler := &apisHandler{
		lister:   informer.Lister(),
		delegate: handler,
	}
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		apisHandler.ServeHTTP(w, req)
	})
}

// apisHandler serves the `/apis` endpoint.
// This is registered as a filter so that it never collides with any explictly registered endpoints
type apisHandler struct {
	lister listers.APIServiceLister

	delegate http.Handler
}

var discoveryGroup = metav1.APIGroup{
	Name: apiregistrationapi.GroupName,
	Versions: []metav1.GroupVersionForDiscovery{
		{
			GroupVersion: apiregistrationv1alpha1api.SchemeGroupVersion.String(),
			Version:      apiregistrationv1alpha1api.SchemeGroupVersion.Version,
		},
	},
	PreferredVersion: metav1.GroupVersionForDiscovery{
		GroupVersion: apiregistrationv1alpha1api.SchemeGroupVersion.String(),
		Version:      apiregistrationv1alpha1api.SchemeGroupVersion.Version,
	},
}

func (r *apisHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	// if the URL is for OUR api group, serve it normally
	if strings.HasPrefix(req.URL.Path+"/", "/apis/"+apiregistrationapi.GroupName+"/") {
		r.delegate.ServeHTTP(w, req)
		return
	}
	// don't handle URLs that aren't /apis
	if req.URL.Path != "/apis" && req.URL.Path != "/apis/" {
		r.delegate.ServeHTTP(w, req)
		return
	}

	discoveryGroupList := &metav1.APIGroupList{
		// always add OUR api group to the list first.  Since we'll never have a registered APIService for it
		// and since this is the crux of the API, having this first will give our names priority.  It's good to be king.
		Groups: []metav1.APIGroup{discoveryGroup},
	}

	apiServices, err := r.lister.List(labels.Everything())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	apiServicesByGroup := apiregistrationapi.SortedByGroup(apiServices)
	for _, apiGroupServers := range apiServicesByGroup {
		// skip the legacy group
		if len(apiGroupServers[0].Spec.Group) == 0 {
			continue
		}
		discoveryGroupList.Groups = append(discoveryGroupList.Groups, *newDiscoveryAPIGroup(apiGroupServers))
	}

	json, err := runtime.Encode(api.Codecs.LegacyCodec(), discoveryGroupList)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if _, err := w.Write(json); err != nil {
		panic(err)
	}
}

func newDiscoveryAPIGroup(apiServices []*apiregistrationapi.APIService) *metav1.APIGroup {
	apiServicesByGroup := apiregistrationapi.SortedByGroup(apiServices)[0]

	discoveryGroup := &metav1.APIGroup{
		Name: apiServicesByGroup[0].Spec.Group,
		PreferredVersion: metav1.GroupVersionForDiscovery{
			GroupVersion: apiServicesByGroup[0].Spec.Group + "/" + apiServicesByGroup[0].Spec.Version,
			Version:      apiServicesByGroup[0].Spec.Version,
		},
	}

	for _, apiService := range apiServicesByGroup {
		discoveryGroup.Versions = append(discoveryGroup.Versions,
			metav1.GroupVersionForDiscovery{
				GroupVersion: apiService.Spec.Group + "/" + apiService.Spec.Version,
				Version:      apiService.Spec.Version,
			},
		)
	}

	return discoveryGroup
}

// apiGroupHandler serves the `/apis/<group>` endpoint.
type apiGroupHandler struct {
	groupName string

	lister listers.APIServiceLister
}

func (r *apiGroupHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	// don't handle URLs that aren't /apis/<groupName>
	if req.URL.Path != "/apis/"+r.groupName && req.URL.Path != "/apis/"+r.groupName+"/" {
		http.Error(w, "", http.StatusNotFound)
		return
	}

	apiServices, err := r.lister.List(labels.Everything())
	if statusErr, ok := err.(*apierrors.StatusError); ok && err != nil {
		apiserver.WriteRawJSON(int(statusErr.Status().Code), statusErr.Status(), w)
		return
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	apiServicesForGroup := []*apiregistrationapi.APIService{}
	for _, apiService := range apiServices {
		if apiService.Spec.Group == r.groupName {
			apiServicesForGroup = append(apiServicesForGroup, apiService)
		}
	}

	if len(apiServicesForGroup) == 0 {
		http.Error(w, "", http.StatusNotFound)
		return
	}

	json, err := runtime.Encode(api.Codecs.LegacyCodec(), newDiscoveryAPIGroup(apiServicesForGroup))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if _, err := w.Write(json); err != nil {
		panic(err)
	}
}
