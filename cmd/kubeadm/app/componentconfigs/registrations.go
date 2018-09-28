/*
Copyright 2018 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"
	kubeproxyconfigv1alpha1 "k8s.io/kube-proxy/config/v1alpha1"
	kubeletconfigv1beta1 "k8s.io/kubelet/config/v1beta1"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
)

// AddToSchemeFunc is a function that adds known types and API GroupVersions to a scheme
type AddToSchemeFunc func(*runtime.Scheme) error

// Registration is an object for registering a Kubernetes ComponentConfig type to be recognized and handled by kubeadm
type Registration struct {
	// MarshalGroupVersion is the preferred external API version to use when marshalling the ComponentConfig
	MarshalGroupVersion schema.GroupVersion
	// AddToSchemeFuncs are a set of functions that register APIs to the scheme
	AddToSchemeFuncs []AddToSchemeFunc
	// DefaulterFunc is a function that based on the internal kubeadm configuration defaults the ComponentConfig struct
	DefaulterFunc func(*kubeadmapi.ClusterConfiguration)
	// ValidateFunc is a function that should validate the ComponentConfig type embedded in the internal kubeadm config struct
	ValidateFunc func(*kubeadmapi.ClusterConfiguration, *field.Path) field.ErrorList
	// EmptyValue holds a pointer to an empty struct of the internal ComponentConfig type
	EmptyValue runtime.Object
	// GetFromInternalConfig returns the pointer to the ComponentConfig API object from the internal kubeadm config struct
	GetFromInternalConfig func(*kubeadmapi.ClusterConfiguration) (runtime.Object, bool)
	// SetToInternalConfig sets the pointer to a ComponentConfig API object embedded in the internal kubeadm config struct
	SetToInternalConfig func(runtime.Object, *kubeadmapi.ClusterConfiguration) bool
	// GetFromConfigMap returns the pointer to the ComponentConfig API object read from the config map stored in the cluster
	GetFromConfigMap func(clientset.Interface, *version.Version) (runtime.Object, error)
}

// Marshal marshals obj to bytes for the current Registration
func (r Registration) Marshal(obj runtime.Object) ([]byte, error) {
	return kubeadmutil.MarshalToYamlForCodecs(obj, r.MarshalGroupVersion, Codecs)
}

// Unmarshal unmarshals the bytes to a runtime.Object using the Codecs registered in this Scheme
func (r Registration) Unmarshal(fileContent []byte) (runtime.Object, error) {
	// Do a deepcopy of the empty value so we don't mutate it, which could lead to strange errors
	obj := r.EmptyValue.DeepCopyObject()

	// Decode the file content into obj which is a pointer to an empty struct of the internal ComponentConfig
	if err := unmarshalObject(obj, fileContent); err != nil {
		return nil, err
	}
	return obj, nil
}

func unmarshalObject(obj runtime.Object, fileContent []byte) error {
	// Decode the file content  using the componentconfig Codecs that knows about all APIs
	if err := runtime.DecodeInto(Codecs.UniversalDecoder(), fileContent, obj); err != nil {
		return err
	}
	return nil
}

const (
	// KubeletConfigurationKind is the kind for the kubelet ComponentConfig
	KubeletConfigurationKind RegistrationKind = "KubeletConfiguration"
	// KubeProxyConfigurationKind is the kind for the kubelet ComponentConfig
	KubeProxyConfigurationKind RegistrationKind = "KubeProxyConfiguration"
)

// RegistrationKind is a string type to ensure not any string can be a key in the Registrations map
type RegistrationKind string

// Registrations holds a set of ComponentConfig Registration objects, where the map key is the kind
type Registrations map[RegistrationKind]Registration

// Known contains the known ComponentConfig registrations to kubeadm
var Known Registrations = map[RegistrationKind]Registration{
	KubeProxyConfigurationKind: {
		// TODO: When a beta version of the kube-proxy ComponentConfig API is available, start using it
		MarshalGroupVersion: kubeproxyconfigv1alpha1.SchemeGroupVersion,
		AddToSchemeFuncs:    []AddToSchemeFunc{kubeproxyconfig.AddToScheme, kubeproxyconfigv1alpha1.AddToScheme},
		DefaulterFunc:       DefaultKubeProxyConfiguration,
		ValidateFunc:        ValidateKubeProxyConfiguration,
		EmptyValue:          &kubeproxyconfig.KubeProxyConfiguration{},
		GetFromInternalConfig: func(cfg *kubeadmapi.ClusterConfiguration) (runtime.Object, bool) {
			return cfg.ComponentConfigs.KubeProxy, cfg.ComponentConfigs.KubeProxy != nil
		},
		SetToInternalConfig: func(obj runtime.Object, cfg *kubeadmapi.ClusterConfiguration) bool {
			kubeproxyConfig, ok := obj.(*kubeproxyconfig.KubeProxyConfiguration)
			if ok {
				cfg.ComponentConfigs.KubeProxy = kubeproxyConfig
			}
			return ok
		},
		GetFromConfigMap: GetFromKubeProxyConfigMap,
	},
	KubeletConfigurationKind: {
		MarshalGroupVersion: kubeletconfigv1beta1.SchemeGroupVersion,
		AddToSchemeFuncs:    []AddToSchemeFunc{kubeletconfig.AddToScheme, kubeletconfigv1beta1.AddToScheme},
		DefaulterFunc:       DefaultKubeletConfiguration,
		ValidateFunc:        ValidateKubeletConfiguration,
		EmptyValue:          &kubeletconfig.KubeletConfiguration{},
		GetFromInternalConfig: func(cfg *kubeadmapi.ClusterConfiguration) (runtime.Object, bool) {
			return cfg.ComponentConfigs.Kubelet, cfg.ComponentConfigs.Kubelet != nil
		},
		SetToInternalConfig: func(obj runtime.Object, cfg *kubeadmapi.ClusterConfiguration) bool {
			kubeletConfig, ok := obj.(*kubeletconfig.KubeletConfiguration)
			if ok {
				cfg.ComponentConfigs.Kubelet = kubeletConfig
			}
			return ok
		},
		GetFromConfigMap: GetFromKubeletConfigMap,
	},
}

// AddToScheme adds all the known ComponentConfig API types referenced in the Registrations object to the scheme
func (rs *Registrations) AddToScheme(scheme *runtime.Scheme) error {
	for _, registration := range *rs {
		for _, addToSchemeFunc := range registration.AddToSchemeFuncs {
			if err := addToSchemeFunc(scheme); err != nil {
				return err
			}
		}
	}
	return nil
}

// Default applies to the ComponentConfig defaults to the internal kubeadm API type
func (rs *Registrations) Default(internalcfg *kubeadmapi.ClusterConfiguration) {
	for _, registration := range *rs {
		registration.DefaulterFunc(internalcfg)
	}
}

// Validate validates the ComponentConfig parts of the internal kubeadm API type
func (rs *Registrations) Validate(internalcfg *kubeadmapi.ClusterConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}
	for kind, registration := range *rs {
		allErrs = append(allErrs, registration.ValidateFunc(internalcfg, field.NewPath(string(kind)))...)
	}
	return allErrs
}
