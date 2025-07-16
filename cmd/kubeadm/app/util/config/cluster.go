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
	"context"
	"crypto/x509"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"

	"github.com/pkg/errors"

	authv1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	errorsutil "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	certutil "k8s.io/client-go/util/cert"
	nodeutil "k8s.io/component-helpers/node/util"
	"k8s.io/klog/v2"
	kubeletconfig "k8s.io/kubelet/config/v1beta1"
	"sigs.k8s.io/yaml"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/config/strict"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/output"
)

// FetchInitConfigurationFromCluster fetches configuration from a ConfigMap in the cluster
func FetchInitConfigurationFromCluster(client clientset.Interface, printer output.Printer, logPrefix string, newControlPlane, skipComponentConfigs bool) (*kubeadmapi.InitConfiguration, error) {
	if printer == nil {
		printer = &output.TextPrinter{}
	}
	_, _ = printer.Printf("[%s] Reading configuration from the %q ConfigMap in namespace %q...\n",
		logPrefix, constants.KubeadmConfigConfigMap, metav1.NamespaceSystem)
	_, _ = printer.Printf("[%s] Use 'kubeadm init phase upload-config --config your-config-file' to re-upload it.\n", logPrefix)

	// Fetch the actual config from cluster
	cfg, err := getInitConfigurationFromCluster(constants.KubernetesDir, client, newControlPlane, skipComponentConfigs)
	if err != nil {
		return nil, err
	}

	// Apply dynamic defaults
	// NB. skip CRI detection here because it won't be used at all and will be overridden later
	if err := SetInitDynamicDefaults(cfg, true); err != nil {
		return nil, err
	}

	return cfg, nil
}

// getInitConfigurationFromCluster is separate only for testing purposes, don't call it directly, use FetchInitConfigurationFromCluster instead
func getInitConfigurationFromCluster(kubeconfigDir string, client clientset.Interface, newControlPlane, skipComponentConfigs bool) (*kubeadmapi.InitConfiguration, error) {
	// Also, the config map really should be KubeadmConfigConfigMap...
	configMap, err := apiclient.GetConfigMapWithShortRetry(client, metav1.NamespaceSystem, constants.KubeadmConfigConfigMap)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get config map")
	}

	// Take an empty versioned InitConfiguration, statically default it and convert it to the internal type
	versionedInitcfg := &kubeadmapiv1.InitConfiguration{}
	kubeadmscheme.Scheme.Default(versionedInitcfg)
	initcfg := &kubeadmapi.InitConfiguration{}
	if err := kubeadmscheme.Scheme.Convert(versionedInitcfg, initcfg, nil); err != nil {
		return nil, errors.Wrap(err, "could not prepare a defaulted InitConfiguration")
	}

	// gets ClusterConfiguration from kubeadm-config
	clusterConfigurationData, ok := configMap.Data[constants.ClusterConfigurationConfigMapKey]
	if !ok {
		return nil, errors.Errorf("unexpected error when reading kubeadm-config ConfigMap: %s key value pair missing", constants.ClusterConfigurationConfigMapKey)
	}
	// If ClusterConfiguration was patched by something other than kubeadm, it may have errors. Warn about them.
	if err := strict.VerifyUnmarshalStrict([]*runtime.Scheme{kubeadmscheme.Scheme},
		kubeadmapiv1.SchemeGroupVersion.WithKind(constants.ClusterConfigurationKind),
		[]byte(clusterConfigurationData)); err != nil {
		klog.Warning(err.Error())
	}
	if err := runtime.DecodeInto(kubeadmscheme.Codecs.UniversalDecoder(), []byte(clusterConfigurationData), &initcfg.ClusterConfiguration); err != nil {
		return nil, errors.Wrap(err, "failed to decode cluster configuration data")
	}

	if !skipComponentConfigs {
		// get the component configs from the corresponding config maps
		if err := componentconfigs.FetchFromCluster(&initcfg.ClusterConfiguration, client); err != nil {
			return nil, errors.Wrap(err, "failed to get component configs")
		}
	}

	// if this isn't a new controlplane instance (e.g. in case of kubeadm upgrades)
	// get nodes specific information as well
	if !newControlPlane {
		// gets the nodeRegistration for the current from the node object
		kubeconfigFile := filepath.Join(kubeconfigDir, constants.KubeletKubeConfigFileName)
		if err := GetNodeRegistration(kubeconfigFile, client, &initcfg.NodeRegistration, &initcfg.ClusterConfiguration); err != nil {
			return nil, errors.Wrap(err, "failed to get node registration")
		}
		// gets the APIEndpoint for the current node
		if err := getAPIEndpoint(client, initcfg.NodeRegistration.Name, &initcfg.LocalAPIEndpoint); err != nil {
			return nil, errors.Wrap(err, "failed to getAPIEndpoint")
		}
	}
	return initcfg, nil
}

// GetNodeName uses 3 different approaches for getting the node name.
// First it attempts to construct a client from the given kubeconfig file
// and get the SelfSubjectReview review for it - i.e. like "kubectl auth whoami".
// If that fails it attempt to parse the kubeconfig client certificate subject.
// Finally, it falls back to using the host name, which might not always be correct
// due to node name overrides.
func GetNodeName(kubeconfigFile string) (string, error) {
	var (
		nodeName string
		err      error
	)
	if kubeconfigFile != "" {
		client, err := kubeconfig.ClientSetFromFile(kubeconfigFile)
		if err == nil {
			nodeName, err = getNodeNameFromSSR(client)
			if err == nil {
				return nodeName, nil
			}
		}
		nodeName, err = getNodeNameFromKubeletConfig(kubeconfigFile)
		if err == nil {
			return nodeName, nil
		}
	}
	nodeName, err = nodeutil.GetHostname("")
	if err != nil {
		return "", errors.Wrapf(err, "could not get node name")
	}
	return nodeName, nil
}

// GetNodeRegistration returns the nodeRegistration for the current node
func GetNodeRegistration(kubeconfigFile string, client clientset.Interface, nodeRegistration *kubeadmapi.NodeRegistrationOptions, clusterCfg *kubeadmapi.ClusterConfiguration) error {
	// gets the name of the current node
	nodeName, err := GetNodeName(kubeconfigFile)
	if err != nil {
		return err
	}

	// gets the corresponding node and retrieves attributes stored there.
	node, err := client.CoreV1().Nodes().Get(context.Background(), nodeName, metav1.GetOptions{})
	if err != nil {
		return errors.Wrap(err, "failed to get corresponding node")
	}

	var (
		criSocket              string
		ok                     bool
		missingAnnotationError = errors.Errorf("node %s doesn't have %s annotation", nodeName, constants.AnnotationKubeadmCRISocket)
	)
	if features.Enabled(clusterCfg.FeatureGates, features.NodeLocalCRISocket) {
		_, err = os.Stat(filepath.Join(constants.KubeletRunDirectory, constants.KubeletInstanceConfigurationFileName))
		if os.IsNotExist(err) {
			criSocket, ok = node.ObjectMeta.Annotations[constants.AnnotationKubeadmCRISocket]
			if !ok {
				return missingAnnotationError
			}
		} else {
			kubeletConfig, err := readKubeletConfig(constants.KubeletRunDirectory, constants.KubeletInstanceConfigurationFileName)
			if err != nil {
				return errors.Wrapf(err, "node %q does not have a kubelet instance configuration", nodeName)
			}
			criSocket = kubeletConfig.ContainerRuntimeEndpoint
		}
	} else {
		criSocket, ok = node.ObjectMeta.Annotations[constants.AnnotationKubeadmCRISocket]
		if !ok {
			return missingAnnotationError
		}
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

// getNodeNameFromKubeletConfig gets the node name from a kubelet config file
// TODO: in future we want to switch to a more canonical way for doing this e.g. by having this
// information in the local kubelet config configuration file.
func getNodeNameFromKubeletConfig(fileName string) (string, error) {
	// loads the kubelet.conf file
	config, err := clientcmd.LoadFromFile(fileName)
	if err != nil {
		return "", err
	}

	// gets the info about the current user
	currentContext, exists := config.Contexts[config.CurrentContext]
	if !exists {
		return "", errors.Errorf("invalid kubeconfig file %s: missing context %s", fileName, config.CurrentContext)
	}
	authInfo, exists := config.AuthInfos[currentContext.AuthInfo]
	if !exists {
		return "", errors.Errorf("invalid kubeconfig file %s: missing AuthInfo %s", fileName, currentContext.AuthInfo)
	}

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
		return "", errors.Errorf("invalid kubeconfig file %s. x509 certificate expected", fileName)
	}

	// Safely pick the first one because the sender's certificate must come first in the list.
	// For details, see: https://www.rfc-editor.org/rfc/rfc4346#section-7.4.2
	cert := certs[0]

	// gets the node name from the certificate common name
	return strings.TrimPrefix(cert.Subject.CommonName, constants.NodesUserPrefix), nil
}

// getNodeNameFromSSR reads the node name from the SelfSubjectReview for a given client.
// If the kubelet.conf is passed as fileName it can be used to retrieve the node name.
func getNodeNameFromSSR(client clientset.Interface) (string, error) {
	ssr, err := client.AuthenticationV1().SelfSubjectReviews().
		Create(context.Background(), &authv1.SelfSubjectReview{}, metav1.CreateOptions{})
	if err != nil {
		return "", err
	}
	user := ssr.Status.UserInfo.Username
	if !strings.HasPrefix(user, constants.NodesUserPrefix) {
		return "", errors.Errorf("%q is not a node client, must have %q prefix in the name",
			user, constants.NodesUserPrefix)
	}
	return strings.TrimPrefix(user, constants.NodesUserPrefix), nil
}

func getAPIEndpoint(client clientset.Interface, nodeName string, apiEndpoint *kubeadmapi.APIEndpoint) error {
	return getAPIEndpointWithRetry(client, nodeName, apiEndpoint,
		constants.KubernetesAPICallRetryInterval, kubeadmapi.GetActiveTimeouts().KubernetesAPICall.Duration)
}

func getAPIEndpointWithRetry(client clientset.Interface, nodeName string, apiEndpoint *kubeadmapi.APIEndpoint,
	interval, timeout time.Duration) error {
	var err error
	var errs []error

	if err = getAPIEndpointFromPodAnnotation(client, nodeName, apiEndpoint, interval, timeout); err == nil {
		return nil
	}
	errs = append(errs, errors.WithMessagef(err, "could not retrieve API endpoints for node %q using pod annotations", nodeName))
	return errorsutil.NewAggregate(errs)
}

func getAPIEndpointFromPodAnnotation(client clientset.Interface, nodeName string, apiEndpoint *kubeadmapi.APIEndpoint,
	interval, timeout time.Duration) error {
	var rawAPIEndpoint string
	var lastErr error
	// Let's tolerate some unexpected transient failures from the API server or load balancers. Also, if
	// static pods were not yet mirrored into the API server we want to wait for this propagation.
	err := wait.PollUntilContextTimeout(context.Background(), interval, timeout, true,
		func(ctx context.Context) (bool, error) {
			rawAPIEndpoint, lastErr = getRawAPIEndpointFromPodAnnotationWithoutRetry(ctx, client, nodeName)
			return lastErr == nil, nil
		})
	if err != nil {
		return err
	}
	parsedAPIEndpoint, err := kubeadmapi.APIEndpointFromString(rawAPIEndpoint)
	if err != nil {
		return errors.Wrapf(err, "could not parse API endpoint for node %q", nodeName)
	}
	*apiEndpoint = parsedAPIEndpoint
	return nil
}

func getRawAPIEndpointFromPodAnnotationWithoutRetry(ctx context.Context, client clientset.Interface, nodeName string) (string, error) {
	podList, err := client.CoreV1().Pods(metav1.NamespaceSystem).List(
		ctx,
		metav1.ListOptions{
			FieldSelector: fmt.Sprintf("spec.nodeName=%s", nodeName),
			LabelSelector: fmt.Sprintf("component=%s,tier=%s", constants.KubeAPIServer, constants.ControlPlaneTier),
		},
	)
	if err != nil {
		return "", errors.Wrap(err, "could not retrieve list of pods to determine api server endpoints")
	}
	if len(podList.Items) != 1 {
		return "", errors.Errorf("API server pod for node name %q has %d entries, only one was expected", nodeName, len(podList.Items))
	}
	if apiServerEndpoint, ok := podList.Items[0].Annotations[constants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey]; ok {
		return apiServerEndpoint, nil
	}
	return "", errors.Errorf("API server pod for node name %q hasn't got a %q annotation, cannot retrieve API endpoint", nodeName, constants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey)
}

// readKubeletConfig reads a KubeletConfiguration from the specified file.
func readKubeletConfig(kubeletDir, fileName string) (*kubeletconfig.KubeletConfiguration, error) {
	kubeletFile := path.Join(kubeletDir, fileName)

	data, err := os.ReadFile(kubeletFile)
	if err != nil {
		return nil, errors.Wrapf(err, "could not read kubelet configuration file %q", kubeletFile)
	}

	var config kubeletconfig.KubeletConfiguration
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, errors.Wrapf(err, "could not parse kubelet configuration file %q", kubeletFile)
	}

	return &config, nil
}
