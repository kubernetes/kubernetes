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
	"github.com/pkg/errors"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/output"
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
			// consider all successfully loaded configs from a document map as user supplied
			cfg.SetUserSupplied(true)
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

	// If the checksum comes up neatly we assume the config was generated
	generatedConfig := VerifyConfigMapSignature(configMap)

	componentCfg, err := h.FromDocumentMap(gvkmap)
	if err != nil {
		// If the config was generated and we get UnsupportedConfigVersionError, we skip loading it.
		// This will force us to use the generated default current version (effectively regenerating the config with the current version).
		if _, ok := err.(*UnsupportedConfigVersionError); ok && generatedConfig {
			return nil, nil
		}
		return nil, err
	}

	if componentCfg != nil {
		componentCfg.SetUserSupplied(!generatedConfig)
	}

	return componentCfg, nil
}

// FromCluster loads a component from a config map in the cluster
func (h *handler) FromCluster(clientset clientset.Interface, clusterCfg *kubeadmapi.ClusterConfiguration) (kubeadmapi.ComponentConfig, error) {
	return h.fromCluster(h, clientset, clusterCfg)
}

// known holds the known component config handlers. Add new component configs here.
var known = []*handler{
	&kubeProxyHandler,
	&kubeletHandler,
}

// configBase is the base type for all component config implementations
type configBase struct {
	// GroupVersion holds the supported GroupVersion for the inheriting config
	GroupVersion schema.GroupVersion

	// userSupplied tells us if the config is user supplied (invalid checksum) or not
	userSupplied bool
}

func (cb *configBase) IsUserSupplied() bool {
	return cb.userSupplied
}

func (cb *configBase) SetUserSupplied(userSupplied bool) {
	cb.userSupplied = userSupplied
}

func (cb *configBase) DeepCopyInto(other *configBase) {
	*other = *cb
}

func cloneBytes(in []byte) []byte {
	out := make([]byte, len(in))
	copy(out, in)
	return out
}

// Marshal is an utility function, used by the component config support implementations to marshal a runtime.Object to YAML with the
// correct group and version
func (cb *configBase) Marshal(object runtime.Object) ([]byte, error) {
	return kubeadmutil.MarshalToYamlForCodecs(object, cb.GroupVersion, Codecs)
}

// Unmarshal attempts to unmarshal a runtime.Object from a document map. If no object is found, no error is returned.
// If a matching group is found, but no matching version an error is returned indicating that users should do manual conversion.
func (cb *configBase) Unmarshal(from kubeadmapi.DocumentMap, into runtime.Object) error {
	for gvk, yaml := range from {
		// If this is a different group, we ignore it
		if gvk.Group != cb.GroupVersion.Group {
			continue
		}

		if gvk.Version != cb.GroupVersion.Version {
			return &UnsupportedConfigVersionError{
				OldVersion:     gvk.GroupVersion(),
				CurrentVersion: cb.GroupVersion,
				Document:       cloneBytes(yaml),
			}
		}

		// As long as we support only component configs with a single kind, this is allowed
		return runtime.DecodeInto(Codecs.UniversalDecoder(), yaml, into)
	}

	return nil
}

// ensureInitializedComponentConfigs is an utility func to initialize the ComponentConfigMap in ClusterConfiguration prior to possible writes to it
func ensureInitializedComponentConfigs(clusterCfg *kubeadmapi.ClusterConfiguration) {
	if clusterCfg.ComponentConfigs == nil {
		clusterCfg.ComponentConfigs = kubeadmapi.ComponentConfigMap{}
	}
}

// Default sets up defaulted component configs in the supplied ClusterConfiguration
func Default(clusterCfg *kubeadmapi.ClusterConfiguration, localAPIEndpoint *kubeadmapi.APIEndpoint, nodeRegOpts *kubeadmapi.NodeRegistrationOptions) {
	ensureInitializedComponentConfigs(clusterCfg)

	for _, handler := range known {
		// If the component config exists, simply default it. Otherwise, create it before defaulting.
		group := handler.GroupVersion.Group
		if componentCfg, ok := clusterCfg.ComponentConfigs[group]; ok {
			componentCfg.Default(clusterCfg, localAPIEndpoint, nodeRegOpts)
		} else {
			componentCfg := handler.CreateEmpty()
			componentCfg.Default(clusterCfg, localAPIEndpoint, nodeRegOpts)
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

// FetchFromClusterWithLocalOverwrites fetches component configs from a cluster and overwrites them locally with
// the ones present in the supplied document map. If any UnsupportedConfigVersionError are not handled by the configs
// in the document map, the function returns them all as a single UnsupportedConfigVersionsErrorMap.
// This function is normally called only in some specific cases during upgrade.
func FetchFromClusterWithLocalOverwrites(clusterCfg *kubeadmapi.ClusterConfiguration, client clientset.Interface, docmap kubeadmapi.DocumentMap) error {
	ensureInitializedComponentConfigs(clusterCfg)

	oldVersionErrs := UnsupportedConfigVersionsErrorMap{}

	for _, handler := range known {
		componentCfg, err := handler.FromCluster(client, clusterCfg)
		if err != nil {
			if vererr, ok := err.(*UnsupportedConfigVersionError); ok {
				oldVersionErrs[handler.GroupVersion.Group] = vererr
			} else {
				return err
			}
		} else if componentCfg != nil {
			clusterCfg.ComponentConfigs[handler.GroupVersion.Group] = componentCfg
		}
	}

	for _, handler := range known {
		componentCfg, err := handler.FromDocumentMap(docmap)
		if err != nil {
			if vererr, ok := err.(*UnsupportedConfigVersionError); ok {
				oldVersionErrs[handler.GroupVersion.Group] = vererr
			} else {
				return err
			}
		} else if componentCfg != nil {
			clusterCfg.ComponentConfigs[handler.GroupVersion.Group] = componentCfg
			delete(oldVersionErrs, handler.GroupVersion.Group)
		}
	}

	if len(oldVersionErrs) != 0 {
		return oldVersionErrs
	}

	return nil
}

// GetVersionStates returns a slice of ComponentConfigVersionState structs
// describing all supported component config groups that were identified on the cluster
func GetVersionStates(clusterCfg *kubeadmapi.ClusterConfiguration, client clientset.Interface, docmap kubeadmapi.DocumentMap) ([]output.ComponentConfigVersionState, error) {
	// We don't want to modify clusterCfg so we make a working deep copy of it.
	// Also, we don't want the defaulted component configs so we get rid of them.
	scratchClusterCfg := clusterCfg.DeepCopy()
	scratchClusterCfg.ComponentConfigs = kubeadmapi.ComponentConfigMap{}

	// Call FetchFromClusterWithLocalOverwrites. This will populate the configs it can load and will return all
	// UnsupportedConfigVersionError(s) in a sinle instance of a MultipleUnsupportedConfigVersionsError.
	var multipleVerErrs UnsupportedConfigVersionsErrorMap
	err := FetchFromClusterWithLocalOverwrites(scratchClusterCfg, client, docmap)
	if err != nil {
		if vererrs, ok := err.(UnsupportedConfigVersionsErrorMap); ok {
			multipleVerErrs = vererrs
		} else {
			// This seems to be a genuine error so we end here
			return nil, err
		}
	}

	results := []output.ComponentConfigVersionState{}
	for _, handler := range known {
		group := handler.GroupVersion.Group
		if vererr, ok := multipleVerErrs[group]; ok {
			// If there is an UnsupportedConfigVersionError then we are dealing with a case where the config was user
			// supplied and requires manual upgrade
			results = append(results, output.ComponentConfigVersionState{
				Group:                 group,
				CurrentVersion:        vererr.OldVersion.Version,
				PreferredVersion:      vererr.CurrentVersion.Version,
				ManualUpgradeRequired: true,
			})
		} else if _, ok := scratchClusterCfg.ComponentConfigs[group]; ok {
			// Normally loaded component config. No manual upgrade required on behalf of users.
			results = append(results, output.ComponentConfigVersionState{
				Group:            group,
				CurrentVersion:   handler.GroupVersion.Version, // Currently kubeadm supports only one version per API
				PreferredVersion: handler.GroupVersion.Version, // group so we can get away with these being the same
			})
		} else {
			// This config was either not present (user did not install an addon) or the config was unsupported kubeadm
			// generated one and is therefore skipped so we can automatically re-generate it (no action required on
			// behalf of the user).
			results = append(results, output.ComponentConfigVersionState{
				Group:            group,
				PreferredVersion: handler.GroupVersion.Version,
			})
		}
	}

	return results, nil
}

// Validate is a placeholder for performing a validation on an already loaded component configs in a ClusterConfiguration
// TODO: investigate if the function can be repurposed for validating component config via CLI
func Validate(clusterCfg *kubeadmapi.ClusterConfiguration) field.ErrorList {
	return field.ErrorList{}
}
