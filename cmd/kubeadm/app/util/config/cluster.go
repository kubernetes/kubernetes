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

package config

import (
	"crypto/x509"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	certutil "k8s.io/client-go/util/cert"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

// FetchInitConfigurationFromCluster fetches configuration from a ConfigMap in the cluster
func FetchInitConfigurationFromCluster(client clientset.Interface, w io.Writer, logPrefix string, newControlPlane bool) (*kubeadmapi.InitConfiguration, error) {
	fmt.Fprintf(w, "[%s] Reading configuration from the cluster...\n", logPrefix)
	fmt.Fprintf(w, "[%s] FYI: You can look at this config file with 'kubectl -n %s get cm %s -oyaml'\n", logPrefix, metav1.NamespaceSystem, constants.KubeadmConfigConfigMap)

	// Fetch the actual config from cluster
	cfg, err := getInitConfigurationFromCluster(constants.KubernetesDir, client, newControlPlane)
	if err != nil {
		return nil, err
	}

	// Apply dynamic defaults
	if err := SetInitDynamicDefaults(cfg); err != nil {
		return nil, err
	}

	return cfg, nil
}

// getInitConfigurationFromCluster is separate only for testing purposes, don't call it directly, use FetchInitConfigurationFromCluster instead
func getInitConfigurationFromCluster(kubeconfigDir string, client clientset.Interface, newControlPlane bool) (*kubeadmapi.InitConfiguration, error) {
	// Also, the config map really should be KubeadmConfigConfigMap...
	configMap, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(constants.KubeadmConfigConfigMap, metav1.GetOptions{})
	if err != nil {
		return nil, errors.Wrap(err, "failed to get config map")
	}

	// InitConfiguration is composed with data from different places
	initcfg := &kubeadmapi.InitConfiguration{}

	// gets ClusterConfiguration from kubeadm-config
	clusterConfigurationData, ok := configMap.Data[constants.ClusterConfigurationConfigMapKey]
	if !ok {
		return nil, errors.Errorf("unexpected error when reading kubeadm-config ConfigMap: %s key value pair missing", constants.ClusterConfigurationConfigMapKey)
	}
	if err := runtime.DecodeInto(kubeadmscheme.Codecs.UniversalDecoder(), []byte(clusterConfigurationData), &initcfg.ClusterConfiguration); err != nil {
		return nil, errors.Wrap(err, "failed to decode cluster configuration data")
	}

	// gets the component configs from the corresponding config maps
	if err := getComponentConfigs(client, &initcfg.ClusterConfiguration); err != nil {
		return nil, errors.Wrap(err, "failed to get component configs")
	}

	// if this isn't a new controlplane instance (e.g. in case of kubeadm upgrades)
	// get nodes specific information as well
	if !newControlPlane {
		// gets the nodeRegistration for the current from the node object
		if err := getNodeRegistration(kubeconfigDir, client, &initcfg.NodeRegistration); err != nil {
			return nil, errors.Wrap(err, "failed to get node registration")
		}
		// gets the APIEndpoint for the current node from then ClusterStatus in the kubeadm-config ConfigMap
		if err := getAPIEndpoint(configMap.Data, initcfg.NodeRegistration.Name, &initcfg.LocalAPIEndpoint); err != nil {
			return nil, errors.Wrap(err, "failed to getAPIEndpoint")
		}
	} else {
		// In the case where newControlPlane is true we don't go through getNodeRegistration() and initcfg.NodeRegistration.CRISocket is empty.
		// This forces DetectCRISocket() to be called later on, and if there is more than one CRI installed on the system, it will error out,
		// while asking for the user to provide an override for the CRI socket. Even if the user provides an override, the call to
		// DetectCRISocket() can happen too early and thus ignore it (while still erroring out).
		// However, if newControlPlane == true, initcfg.NodeRegistration is not used at all and it's overwritten later on.
		// Thus it's necessary to supply some default value, that will avoid the call to DetectCRISocket() and as
		// initcfg.NodeRegistration is discarded, setting whatever value here is harmless.
		initcfg.NodeRegistration.CRISocket = constants.DefaultDockerCRISocket
	}
	return initcfg, nil
}

// getNodeRegistration returns the nodeRegistration for the current node
func getNodeRegistration(kubeconfigDir string, client clientset.Interface, nodeRegistration *kubeadmapi.NodeRegistrationOptions) error {
	// gets the name of the current node
	nodeName, err := getNodeNameFromKubeletConfig(kubeconfigDir)
	if err != nil {
		return errors.Wrap(err, "failed to get node name from kubelet config")
	}

	// gets the corresponding node and retrives attributes stored there.
	node, err := client.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
	if err != nil {
		return errors.Wrap(err, "faild to get corresponding node")
	}

	criSocket, ok := node.ObjectMeta.Annotations[constants.AnnotationKubeadmCRISocket]
	if !ok {
		return errors.Errorf("node %s doesn't have %s annotation", nodeName, constants.AnnotationKubeadmCRISocket)
	}

	// returns the nodeRegistration attributes
	nodeRegistration.Name = nodeName
	nodeRegistration.CRISocket = criSocket
	nodeRegistration.Taints = node.Spec.Taints
	// NB. currently nodeRegistration.KubeletExtraArgs isn't stored at node level but only in the kubeadm-flags.env
	//     that isn't modified during upgrades
	//     in future we might reconsider this thus enabling changes to the kubeadm-flags.env during upgrades as well
	return nil
}

// getNodeNameFromConfig gets the node name from a kubelet config file
// TODO: in future we want to switch to a more canonical way for doing this e.g. by having this
//       information in the local kubelet config.yaml
func getNodeNameFromKubeletConfig(kubeconfigDir string) (string, error) {
	// loads the kubelet.conf file
	fileName := filepath.Join(kubeconfigDir, constants.KubeletKubeConfigFileName)
	config, err := clientcmd.LoadFromFile(fileName)
	if err != nil {
		return "", err
	}

	// gets the info about the current user
	authInfo := config.AuthInfos[config.Contexts[config.CurrentContext].AuthInfo]

	// gets the X509 certificate with current user credentials
	var certs []*x509.Certificate
	if len(authInfo.ClientCertificateData) > 0 {
		// if the config file uses an embedded x509 certificate (e.g. kubelet.conf created by kubeadm), parse it
		if certs, err = certutil.ParseCertsPEM(authInfo.ClientCertificateData); err != nil {
			return "", err
		}
	} else if len(authInfo.ClientCertificate) > 0 {
		// if the config file links an external x509 certificate (e.g. kubelet.conf created by TLS bootstrap), load it
		if certs, err = certutil.CertsFromFile(authInfo.ClientCertificate); err != nil {
			return "", err
		}
	} else {
		return "", errors.New("invalid kubelet.conf. X509 certificate expected")
	}

	// We are only putting one certificate in the certificate pem file, so it's safe to just pick the first one
	// TODO: Support multiple certs here in order to be able to rotate certs
	cert := certs[0]

	// gets the node name from the certificate common name
	return strings.TrimPrefix(cert.Subject.CommonName, constants.NodesUserPrefix), nil
}

// getAPIEndpoint returns the APIEndpoint for the current node
func getAPIEndpoint(data map[string]string, nodeName string, apiEndpoint *kubeadmapi.APIEndpoint) error {
	// gets the ClusterStatus from kubeadm-config
	clusterStatus, err := UnmarshalClusterStatus(data)
	if err != nil {
		return err
	}

	// gets the APIEndpoint for the current machine from the ClusterStatus
	e, ok := clusterStatus.APIEndpoints[nodeName]
	if !ok {
		return errors.New("failed to get APIEndpoint information for this node")
	}

	apiEndpoint.AdvertiseAddress = e.AdvertiseAddress
	apiEndpoint.BindPort = e.BindPort
	return nil
}

// getComponentConfigs gets the component configs from the corresponding config maps
func getComponentConfigs(client clientset.Interface, clusterConfiguration *kubeadmapi.ClusterConfiguration) error {
	// some config maps is versioned, so we need the KubernetesVersion for getting the right config map
	k8sVersion := version.MustParseGeneric(clusterConfiguration.KubernetesVersion)
	for kind, registration := range componentconfigs.Known {
		obj, err := registration.GetFromConfigMap(client, k8sVersion)
		if err != nil {
			return err
		}

		if ok := registration.SetToInternalConfig(obj, clusterConfiguration); !ok {
			return errors.Errorf("couldn't save componentconfig value for kind %q", string(kind))
		}
	}
	return nil
}

// GetClusterStatus returns the kubeadm cluster status read from the kubeadm-config ConfigMap
func GetClusterStatus(client clientset.Interface) (*kubeadmapi.ClusterStatus, error) {
	configMap, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(constants.KubeadmConfigConfigMap, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		return &kubeadmapi.ClusterStatus{}, nil
	}
	if err != nil {
		return nil, err
	}

	clusterStatus, err := UnmarshalClusterStatus(configMap.Data)
	if err != nil {
		return nil, err
	}

	return clusterStatus, nil
}

// UnmarshalClusterStatus takes raw ConfigMap.Data and converts it to a ClusterStatus object
func UnmarshalClusterStatus(data map[string]string) (*kubeadmapi.ClusterStatus, error) {
	clusterStatusData, ok := data[constants.ClusterStatusConfigMapKey]
	if !ok {
		return nil, errors.Errorf("unexpected error when reading kubeadm-config ConfigMap: %s key value pair missing", constants.ClusterStatusConfigMapKey)
	}
	clusterStatus := &kubeadmapi.ClusterStatus{}
	if err := runtime.DecodeInto(kubeadmscheme.Codecs.UniversalDecoder(), []byte(clusterStatusData), clusterStatus); err != nil {
		return nil, err
	}

	return clusterStatus, nil
}
