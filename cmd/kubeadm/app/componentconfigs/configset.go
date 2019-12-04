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

package componentconfigs

import (
	"sort"

	"github.com/pkg/errors"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

// handler is a package internal type that handles component config factory and common functionality.
// Every component config group should have exactly one static instance of handler.
type handler struct {
	// GroupVersion holds this handler's group name and preferred version
	GroupVersion schema.GroupVersion

	// AddToScheme points to a func that should add the GV types to a schema
	AddToScheme func(*runtime.Scheme) error

	// CreateEmpty returns an empty kubeadmapi.ComponentConfig (not even defaulted)
	CreateEmpty func() kubeadmapi.ComponentConfig

	// fromCluster should load the component config from a config map on the cluster.
	// Don't use this directly! Use FromCluster instead!
	fromCluster func(*handler, clientset.Interface, *kubeadmapi.ClusterConfiguration) (kubeadmapi.ComponentConfig, error)
}

// FromDocumentMap looks in the document map for documents with this handler's group.
// If such are found a new component config is instantiated and the documents are loaded into it.
// No error is returned if no documents are found.
func (h *handler) FromDocumentMap(docmap kubeadmapi.DocumentMap) (kubeadmapi.ComponentConfig, error) {
	for gvk := range docmap {
		if gvk.Group == h.GroupVersion.Group {
			cfg := h.CreateEmpty()
			if err := cfg.Unmarshal(docmap); err != nil {
				return nil, err
			}
			return cfg, nil
		}
	}
	return nil, nil
}

// fromConfigMap is an utility function, which will load the value of a key of a config map and use h.FromDocumentMap() to perform the parsing
// This is an utility func. Used by the component config support implementations. Don't use it outside of that context.
func (h *handler) fromConfigMap(client clientset.Interface, cmName, cmKey string, mustExist bool) (kubeadmapi.ComponentConfig, error) {
	configMap, err := apiclient.GetConfigMapWithRetry(client, metav1.NamespaceSystem, cmName)
	if err != nil {
		if !mustExist && (apierrors.IsNotFound(err) || apierrors.IsForbidden(err)) {
			klog.Warningf("Warning: No %s config is loaded. Continuing without it: %v", h.GroupVersion, err)
			return nil, nil
		}
		return nil, err
	}

	configData, ok := configMap.Data[cmKey]
	if !ok {
		return nil, errors.Errorf("unexpected error when reading %s ConfigMap: %s key value pair missing", cmName, cmKey)
	}

	gvkmap, err := kubeadmutil.SplitYAMLDocuments([]byte(configData))
	if err != nil {
		return nil, err
	}

	return h.FromDocumentMap(gvkmap)
}

// FromCluster loads a component from a config map in the cluster
func (h *handler) FromCluster(clientset clientset.Interface, clusterCfg *kubeadmapi.ClusterConfiguration) (kubeadmapi.ComponentConfig, error) {
	return h.fromCluster(h, clientset, clusterCfg)
}

// Marshal is an utility function, used by the component config support implementations to marshal a runtime.Object to YAML with the
// correct group and version
func (h *handler) Marshal(object runtime.Object) ([]byte, error) {
	return kubeadmutil.MarshalToYamlForCodecs(object, h.GroupVersion, Codecs)
}

// Unmarshal attempts to unmarshal a runtime.Object from a document map. If no object is found, no error is returned.
// If a matching group is found, but no matching version an error is returned indicating that users should do manual conversion.
func (h *handler) Unmarshal(from kubeadmapi.DocumentMap, into runtime.Object) error {
	for gvk, yaml := range from {
		// If this is a different group, we ignore it
		if gvk.Group != h.GroupVersion.Group {
			continue
		}

		// If this is the correct group, but different version, we return an error
		if gvk.Version != h.GroupVersion.Version {
			// TODO: Replace this with a special error type and make UX better around it
			return errors.Errorf("unexpected apiVersion %q, you may have to do manual conversion to %q and execute kubeadm again", gvk.GroupVersion(), h.GroupVersion)
		}

		// As long as we support only component configs with a single kind, this is allowed
		return runtime.DecodeInto(Codecs.UniversalDecoder(), yaml, into)
	}

	return nil
}

// known holds the known component config handlers. Add new component configs here.
var known = []*handler{
	&kubeProxyHandler,
	&kubeletHandler,
}

// ensureInitializedComponentConfigs is an utility func to initialize the ComponentConfigMap in ClusterConfiguration prior to possible writes to it
func ensureInitializedComponentConfigs(clusterCfg *kubeadmapi.ClusterConfiguration) {
	if clusterCfg.ComponentConfigs == nil {
		clusterCfg.ComponentConfigs = kubeadmapi.ComponentConfigMap{}
	}
}

// Default sets up defaulted component configs in the supplied ClusterConfiguration
func Default(clusterCfg *kubeadmapi.ClusterConfiguration, localAPIEndpoint *kubeadmapi.APIEndpoint) {
	ensureInitializedComponentConfigs(clusterCfg)

	for _, handler := range known {
		// If the component config exists, simply default it. Otherwise, create it before defaulting.
		group := handler.GroupVersion.Group
		if componentCfg, ok := clusterCfg.ComponentConfigs[group]; ok {
			componentCfg.Default(clusterCfg, localAPIEndpoint)
		} else {
			componentCfg := handler.CreateEmpty()
			componentCfg.Default(clusterCfg, localAPIEndpoint)
			clusterCfg.ComponentConfigs[group] = componentCfg
		}
	}
}

// FetchFromCluster attempts to fetch all known component configs from their config maps and store them in the supplied ClusterConfiguration
func FetchFromCluster(clusterCfg *kubeadmapi.ClusterConfiguration, client clientset.Interface) error {
	ensureInitializedComponentConfigs(clusterCfg)

	for _, handler := range known {
		componentCfg, err := handler.FromCluster(client, clusterCfg)
		if err != nil {
			return err
		}

		if componentCfg != nil {
			clusterCfg.ComponentConfigs[handler.GroupVersion.Group] = componentCfg
		}
	}

	return nil
}

// FetchFromDocumentMap attempts to load all known component configs from a document map into the supplied ClusterConfiguration
func FetchFromDocumentMap(clusterCfg *kubeadmapi.ClusterConfiguration, docmap kubeadmapi.DocumentMap) error {
	ensureInitializedComponentConfigs(clusterCfg)

	for _, handler := range known {
		componentCfg, err := handler.FromDocumentMap(docmap)
		if err != nil {
			return err
		}

		if componentCfg != nil {
			clusterCfg.ComponentConfigs[handler.GroupVersion.Group] = componentCfg
		}
	}

	return nil
}

// Validate is a placeholder for performing a validation on an already loaded component configs in a ClusterConfiguration
// Currently it prints a warning that no validation was performed
func Validate(clusterCfg *kubeadmapi.ClusterConfiguration) field.ErrorList {
	groups := []string{}
	for group := range clusterCfg.ComponentConfigs {
		groups = append(groups, group)
	}
	sort.Strings(groups) // The sort is needed to make the output predictable
	klog.Warningf("WARNING: kubeadm cannot validate component configs for API groups %v", groups)
	return field.ErrorList{}
}
