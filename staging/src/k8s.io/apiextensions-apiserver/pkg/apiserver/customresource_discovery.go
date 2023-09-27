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

package apiserver

import (
	"context"
	"net/http"
	"sort"
	"strings"

	"github.com/kcp-dev/logicalcluster/v3"

	autoscaling "k8s.io/api/autoscaling/v1"
	apiextensionshelpers "k8s.io/apiextensions-apiserver/pkg/apihelpers"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/kcp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/version"
	"k8s.io/apiserver/pkg/endpoints/discovery"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
)

type versionDiscoveryHandler struct {
	crdLister kcp.ClusterAwareCRDClusterLister
	delegate  http.Handler
}

func (r *versionDiscoveryHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	pathParts := splitPath(req.URL.Path)
	// only match /apis/<group>/<version>
	if len(pathParts) != 3 || pathParts[0] != "apis" {
		r.delegate.ServeHTTP(w, req)
		return
	}

	clusterName, wildcard, err := genericapirequest.ClusterNameOrWildcardFrom(req.Context())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if wildcard {
		// this is the only case where wildcard works for a list because this is our special CRD lister that handles it.
		clusterName = "*"
	}

	requestedGroup := pathParts[1]
	requestedVersion := pathParts[2]

	crds, err := r.crdLister.Cluster(clusterName).List(req.Context(), labels.Everything())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	apiResources := APIResourcesForGroupVersion(requestedGroup, requestedVersion, crds)

	resourceListerFunc := discovery.APIResourceListerFunc(func() []metav1.APIResource {
		return apiResources
	})

	discovery.NewAPIVersionHandler(Codecs, schema.GroupVersion{Group: requestedGroup, Version: requestedVersion}, resourceListerFunc).ServeHTTP(w, req)
}

func APIResourcesForGroupVersion(requestedGroup, requestedVersion string, crds []*apiextensionsv1.CustomResourceDefinition) []metav1.APIResource {
	apiResourcesForDiscovery := []metav1.APIResource{}

	for _, crd := range crds {
		if requestedGroup != crd.Spec.Group {
			continue
		}

		if !apiextensionshelpers.IsCRDConditionTrue(crd, apiextensionsv1.Established) {
			continue
		}

		var (
			storageVersionHash string
			subresources       *apiextensionsv1.CustomResourceSubresources
			foundVersion       = false
		)

		for _, v := range crd.Spec.Versions {
			if !v.Served {
				continue
			}

			// HACK: support the case when we add core resources through CRDs (KCP scenario)
			groupVersion := crd.Spec.Group + "/" + v.Name
			if crd.Spec.Group == "" {
				groupVersion = v.Name
			}

			gv := metav1.GroupVersion{Group: groupVersion, Version: v.Name}

			if v.Name == requestedVersion {
				foundVersion = true
				subresources = v.Subresources
			}
			if v.Storage {
				storageVersionHash = discovery.StorageVersionHash(logicalcluster.From(crd), gv.Group, gv.Version, crd.Spec.Names.Kind)
			}
		}

		if !foundVersion {
			// This CRD doesn't have the requested version
			continue
		}

		verbs := metav1.Verbs([]string{"delete", "deletecollection", "get", "list", "patch", "create", "update", "watch"})
		// if we're terminating we don't allow some verbs
		if apiextensionshelpers.IsCRDConditionTrue(crd, apiextensionsv1.Terminating) {
			verbs = metav1.Verbs([]string{"delete", "deletecollection", "get", "list", "watch"})
		}

		apiResourcesForDiscovery = append(apiResourcesForDiscovery, metav1.APIResource{
			Name:               crd.Status.AcceptedNames.Plural,
			SingularName:       crd.Status.AcceptedNames.Singular,
			Namespaced:         crd.Spec.Scope == apiextensionsv1.NamespaceScoped,
			Kind:               crd.Status.AcceptedNames.Kind,
			Verbs:              verbs,
			ShortNames:         crd.Status.AcceptedNames.ShortNames,
			Categories:         crd.Status.AcceptedNames.Categories,
			StorageVersionHash: storageVersionHash,
		})

		if subresources != nil && subresources.Status != nil {
			apiResourcesForDiscovery = append(apiResourcesForDiscovery, metav1.APIResource{
				Name:       crd.Status.AcceptedNames.Plural + "/status",
				Namespaced: crd.Spec.Scope == apiextensionsv1.NamespaceScoped,
				Kind:       crd.Status.AcceptedNames.Kind,
				Verbs:      metav1.Verbs([]string{"get", "patch", "update"}),
			})
		}

		if subresources != nil && subresources.Scale != nil {
			apiResourcesForDiscovery = append(apiResourcesForDiscovery, metav1.APIResource{
				Group:      autoscaling.GroupName,
				Version:    "v1",
				Kind:       "Scale",
				Name:       crd.Status.AcceptedNames.Plural + "/scale",
				Namespaced: crd.Spec.Scope == apiextensionsv1.NamespaceScoped,
				Verbs:      metav1.Verbs([]string{"get", "patch", "update"}),
			})
		}
	}

	return apiResourcesForDiscovery
}

type groupDiscoveryHandler struct {
	crdLister kcp.ClusterAwareCRDClusterLister
	delegate  http.Handler
}

func (r *groupDiscoveryHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	pathParts := splitPath(req.URL.Path)
	// only match /apis/<group>
	if len(pathParts) != 2 || pathParts[0] != "apis" {
		r.delegate.ServeHTTP(w, req)
		return
	}

	clusterName, wildcard, err := genericapirequest.ClusterNameOrWildcardFrom(req.Context())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if wildcard {
		// this is the only case where wildcard works for a list because this is our special CRD lister that handles it.
		clusterName = "*"
	}

	apiVersionsForDiscovery := []metav1.GroupVersionForDiscovery{}
	versionsForDiscoveryMap := map[metav1.GroupVersion]bool{}

	requestedGroup := pathParts[1]

	crds, err := r.crdLister.Cluster(clusterName).List(req.Context(), labels.Everything())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	foundGroup := false
	for _, crd := range crds {
		if requestedGroup != crd.Spec.Group {
			continue
		}

		if !apiextensionshelpers.IsCRDConditionTrue(crd, apiextensionsv1.Established) {
			continue
		}

		for _, v := range crd.Spec.Versions {
			if !v.Served {
				continue
			}
			// If there is any Served version, that means the group should show up in discovery
			foundGroup = true

			// HACK: support the case when we add core resources through CRDs (KCP scenario)
			groupVersion := crd.Spec.Group + "/" + v.Name
			if crd.Spec.Group == "" {
				groupVersion = v.Name
			}

			gv := metav1.GroupVersion{Group: crd.Spec.Group, Version: v.Name}

			if !versionsForDiscoveryMap[gv] {
				versionsForDiscoveryMap[gv] = true
				apiVersionsForDiscovery = append(apiVersionsForDiscovery, metav1.GroupVersionForDiscovery{
					GroupVersion: groupVersion,
					Version:      v.Name,
				})
			}
		}
	}

	sortGroupDiscoveryByKubeAwareVersion(apiVersionsForDiscovery)

	if !foundGroup {
		r.delegate.ServeHTTP(w, req)
		return
	}

	apiGroup := metav1.APIGroup{
		Name:     requestedGroup,
		Versions: apiVersionsForDiscovery,
		// the preferred versions for a group is the first item in
		// apiVersionsForDiscovery after it put in the right ordered
		PreferredVersion: apiVersionsForDiscovery[0],
	}

	discovery.NewAPIGroupHandler(Codecs, apiGroup).ServeHTTP(w, req)
}

type rootDiscoveryHandler struct {
	crdLister kcp.ClusterAwareCRDClusterLister
	delegate  http.Handler
}

func (r *rootDiscoveryHandler) Groups(ctx context.Context, _ *http.Request) ([]metav1.APIGroup, error) {
	apiVersionsForDiscovery := map[string][]metav1.GroupVersionForDiscovery{}
	versionsForDiscoveryMap := map[string]map[metav1.GroupVersion]bool{}

	clusterName, wildcard, err := genericapirequest.ClusterNameOrWildcardFrom(ctx)
	if err != nil {
		return nil, err
	}
	if wildcard {
		// this is the only case where wildcard works for a list because this is our special CRD lister that handles it.
		clusterName = "*"
	}

	crds, err := r.crdLister.Cluster(clusterName).List(ctx, labels.Everything())
	if err != nil {
		return []metav1.APIGroup{}, err
	}
	for _, crd := range crds {
		if !apiextensionshelpers.IsCRDConditionTrue(crd, apiextensionsv1.Established) {
			continue
		}

		for _, v := range crd.Spec.Versions {
			if !v.Served {
				continue
			}

			if crd.Spec.Group == "" {
				// Don't include CRDs in the core ("") group in /apis discovery. They
				// instead are in /api/v1 handled elsewhere.
				continue
			}
			groupVersion := crd.Spec.Group + "/" + v.Name

			gv := metav1.GroupVersion{Group: crd.Spec.Group, Version: v.Name}

			m, ok := versionsForDiscoveryMap[crd.Spec.Group]
			if !ok {
				m = make(map[metav1.GroupVersion]bool)
			}

			if !m[gv] {
				m[gv] = true
				groupVersions := apiVersionsForDiscovery[crd.Spec.Group]
				groupVersions = append(groupVersions, metav1.GroupVersionForDiscovery{
					GroupVersion: groupVersion,
					Version:      v.Name,
				})
				apiVersionsForDiscovery[crd.Spec.Group] = groupVersions
			}

			versionsForDiscoveryMap[crd.Spec.Group] = m
		}
	}

	for _, versions := range apiVersionsForDiscovery {
		sortGroupDiscoveryByKubeAwareVersion(versions)

	}

	groupList := make([]metav1.APIGroup, 0, len(apiVersionsForDiscovery))
	for group, versions := range apiVersionsForDiscovery {
		g := metav1.APIGroup{
			Name:             group,
			Versions:         versions,
			PreferredVersion: versions[0],
		}
		groupList = append(groupList, g)
	}
	return groupList, nil
}

// splitPath returns the segments for a URL path.
func splitPath(path string) []string {
	path = strings.Trim(path, "/")
	if path == "" {
		return []string{}
	}
	return strings.Split(path, "/")
}

func sortGroupDiscoveryByKubeAwareVersion(gd []metav1.GroupVersionForDiscovery) {
	sort.Slice(gd, func(i, j int) bool {
		return version.CompareKubeAwareVersionStrings(gd[i].Version, gd[j].Version) > 0
	})
}
