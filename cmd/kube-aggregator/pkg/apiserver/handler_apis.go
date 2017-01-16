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

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
	v1listers "k8s.io/kubernetes/pkg/client/listers/core/v1"
	"k8s.io/kubernetes/pkg/genericapiserver/api/handlers/responsewriters"

	apiregistrationapi "k8s.io/kubernetes/cmd/kube-aggregator/pkg/apis/apiregistration"
	apiregistrationv1alpha1api "k8s.io/kubernetes/cmd/kube-aggregator/pkg/apis/apiregistration/v1alpha1"
	informers "k8s.io/kubernetes/cmd/kube-aggregator/pkg/client/informers/apiregistration/internalversion"
	listers "k8s.io/kubernetes/cmd/kube-aggregator/pkg/client/listers/apiregistration/internalversion"
)

// WithAPIs adds the handling for /apis and /apis/<group: -apiregistration.k8s.io>.
func WithAPIs(handler http.Handler, informer informers.APIServiceInformer, serviceLister v1listers.ServiceLister, endpointsLister v1listers.EndpointsLister) http.Handler {
	apisHandler := &apisHandler{
		lister:          informer.Lister(),
		delegate:        handler,
		serviceLister:   serviceLister,
		endpointsLister: endpointsLister,
	}
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		apisHandler.ServeHTTP(w, req)
	})
}

// apisHandler serves the `/apis` endpoint.
// This is registered as a filter so that it never collides with any explictly registered endpoints
type apisHandler struct {
	lister listers.APIServiceLister

	serviceLister   v1listers.ServiceLister
	endpointsLister v1listers.EndpointsLister

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
		discoveryGroup := convertToDiscoveryAPIGroup(apiGroupServers, r.serviceLister, r.endpointsLister)
		if discoveryGroup != nil {
			discoveryGroupList.Groups = append(discoveryGroupList.Groups, *discoveryGroup)
		}
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

// convertToDiscoveryAPIGroup takes apiservices in a single group and returns a discovery compatible object.
// if none of the services are available, it will return nil.
func convertToDiscoveryAPIGroup(apiServices []*apiregistrationapi.APIService, serviceLister v1listers.ServiceLister, endpointsLister v1listers.EndpointsLister) *metav1.APIGroup {
	apiServicesByGroup := apiregistrationapi.SortedByGroup(apiServices)[0]

	var discoveryGroup *metav1.APIGroup

	for _, apiService := range apiServicesByGroup {
		// skip any API services without actual services
		if _, err := serviceLister.Services(apiService.Spec.Service.Namespace).Get(apiService.Spec.Service.Name); err != nil {
			continue
		}

		hasActiveEndpoints := false
		endpoints, err := endpointsLister.Endpoints(apiService.Spec.Service.Namespace).Get(apiService.Spec.Service.Name)
		// skip any API services without endpoints
		if err != nil {
			continue
		}
		for _, subset := range endpoints.Subsets {
			if len(subset.Addresses) > 0 {
				hasActiveEndpoints = true
				break
			}
		}
		if !hasActiveEndpoints {
			continue
		}

		// the first APIService which is valid becomes the default
		if discoveryGroup == nil {
			discoveryGroup = &metav1.APIGroup{
				Name: apiService.Spec.Group,
				PreferredVersion: metav1.GroupVersionForDiscovery{
					GroupVersion: apiService.Spec.Group + "/" + apiService.Spec.Version,
					Version:      apiService.Spec.Version,
				},
			}
		}

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

	serviceLister   v1listers.ServiceLister
	endpointsLister v1listers.EndpointsLister
}

func (r *apiGroupHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	// don't handle URLs that aren't /apis/<groupName>
	if req.URL.Path != "/apis/"+r.groupName && req.URL.Path != "/apis/"+r.groupName+"/" {
		http.Error(w, "", http.StatusNotFound)
		return
	}

	apiServices, err := r.lister.List(labels.Everything())
	if statusErr, ok := err.(*apierrors.StatusError); ok && err != nil {
		responsewriters.WriteRawJSON(int(statusErr.Status().Code), statusErr.Status(), w)
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

	discoveryGroup := convertToDiscoveryAPIGroup(apiServicesForGroup, r.serviceLister, r.endpointsLister)
	if discoveryGroup == nil {
		http.Error(w, "", http.StatusNotFound)
		return
	}
	json, err := runtime.Encode(api.Codecs.LegacyCodec(), discoveryGroup)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if _, err := w.Write(json); err != nil {
		panic(err)
	}
}
