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

package controlplane

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	staticpodutil "k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
	authzmodes "k8s.io/kubernetes/pkg/kubeapiserver/authorizer/modes"
	"k8s.io/kubernetes/pkg/util/version"
)

// Static pod definitions in golang form are included below so that `kubeadm init` can get going.
const (
	DefaultCloudConfigPath = "/etc/kubernetes/cloud-config"

	defaultv17AdmissionControl = "Initializers,NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,ResourceQuota"
)

// CreateInitStaticPodManifestFiles will write all static pod manifest files needed to bring up the control plane.
func CreateInitStaticPodManifestFiles(manifestDir string, cfg *kubeadmapi.MasterConfiguration) error {
	return createStaticPodFiles(manifestDir, cfg, kubeadmconstants.KubeAPIServer, kubeadmconstants.KubeControllerManager, kubeadmconstants.KubeScheduler)
}

// CreateAPIServerStaticPodManifestFile will write APIserver static pod manifest file.
func CreateAPIServerStaticPodManifestFile(manifestDir string, cfg *kubeadmapi.MasterConfiguration) error {
	return createStaticPodFiles(manifestDir, cfg, kubeadmconstants.KubeAPIServer)
}

// CreateControllerManagerStaticPodManifestFile will write  controller manager static pod manifest file.
func CreateControllerManagerStaticPodManifestFile(manifestDir string, cfg *kubeadmapi.MasterConfiguration) error {
	return createStaticPodFiles(manifestDir, cfg, kubeadmconstants.KubeControllerManager)
}

// CreateSchedulerStaticPodManifestFile will write scheduler static pod manifest file.
func CreateSchedulerStaticPodManifestFile(manifestDir string, cfg *kubeadmapi.MasterConfiguration) error {
	return createStaticPodFiles(manifestDir, cfg, kubeadmconstants.KubeScheduler)
}

// GetStaticPodSpecs returns all staticPodSpecs actualized to the context of the current MasterConfiguration
// NB. this methods holds the information about how kubeadm creates static pod mainfests.
func GetStaticPodSpecs(cfg *kubeadmapi.MasterConfiguration, k8sVersion *version.Version) map[string]v1.Pod {

	// Get the required hostpath mounts
	mounts := getHostPathVolumesForTheControlPlane(cfg)

	// Prepare static pod specs
	staticPodSpecs := map[string]v1.Pod{
		kubeadmconstants.KubeAPIServer: staticpodutil.ComponentPod(v1.Container{
			Name:          kubeadmconstants.KubeAPIServer,
			Image:         images.GetCoreImage(kubeadmconstants.KubeAPIServer, cfg.ImageRepository, cfg.KubernetesVersion, cfg.UnifiedControlPlaneImage),
			Command:       getAPIServerCommand(cfg, k8sVersion),
			VolumeMounts:  mounts.GetVolumeMounts(kubeadmconstants.KubeAPIServer),
			LivenessProbe: staticpodutil.ComponentProbe(int(cfg.API.BindPort), "/healthz", v1.URISchemeHTTPS),
			Resources:     staticpodutil.ComponentResources("250m"),
			Env:           getProxyEnvVars(),
		}, mounts.GetVolumes(kubeadmconstants.KubeAPIServer)),
		kubeadmconstants.KubeControllerManager: staticpodutil.ComponentPod(v1.Container{
			Name:          kubeadmconstants.KubeControllerManager,
			Image:         images.GetCoreImage(kubeadmconstants.KubeControllerManager, cfg.ImageRepository, cfg.KubernetesVersion, cfg.UnifiedControlPlaneImage),
			Command:       getControllerManagerCommand(cfg, k8sVersion),
			VolumeMounts:  mounts.GetVolumeMounts(kubeadmconstants.KubeControllerManager),
			LivenessProbe: staticpodutil.ComponentProbe(10252, "/healthz", v1.URISchemeHTTP),
			Resources:     staticpodutil.ComponentResources("200m"),
			Env:           getProxyEnvVars(),
		}, mounts.GetVolumes(kubeadmconstants.KubeControllerManager)),
		kubeadmconstants.KubeScheduler: staticpodutil.ComponentPod(v1.Container{
			Name:          kubeadmconstants.KubeScheduler,
			Image:         images.GetCoreImage(kubeadmconstants.KubeScheduler, cfg.ImageRepository, cfg.KubernetesVersion, cfg.UnifiedControlPlaneImage),
			Command:       getSchedulerCommand(cfg),
			VolumeMounts:  mounts.GetVolumeMounts(kubeadmconstants.KubeScheduler),
			LivenessProbe: staticpodutil.ComponentProbe(10251, "/healthz", v1.URISchemeHTTP),
			Resources:     staticpodutil.ComponentResources("100m"),
			Env:           getProxyEnvVars(),
		}, mounts.GetVolumes(kubeadmconstants.KubeScheduler)),
	}

	return staticPodSpecs
}

// createStaticPodFiles creates all the requested static pod files.
func createStaticPodFiles(manifestDir string, cfg *kubeadmapi.MasterConfiguration, componentNames ...string) error {

	// TODO: Move the "pkg/util/version".Version object into the internal API instead of always parsing the string
	k8sVersion, err := version.ParseSemantic(cfg.KubernetesVersion)
	if err != nil {
		return err
	}

	// gets the StaticPodSpecs, actualized for the current MasterConfiguration
	specs := GetStaticPodSpecs(cfg, k8sVersion)

	// creates required static pod specs
	for _, componentName := range componentNames {
		// retrives the StaticPodSpec for given component
		spec, exists := specs[componentName]
		if !exists {
			return fmt.Errorf("couldn't retrive StaticPodSpec for %s", componentName)
		}

		// writes the StaticPodSpec to disk
		if err := staticpodutil.WriteStaticPodToDisk(componentName, manifestDir, spec); err != nil {
			return fmt.Errorf("failed to create static pod manifest file for %q: %v", componentName, err)
		}
	}

	return nil
}

// getAPIServerCommand builds the right API server command from the given config object and version
func getAPIServerCommand(cfg *kubeadmapi.MasterConfiguration, k8sVersion *version.Version) []string {
	defaultArguments := map[string]string{
		"advertise-address":                 cfg.API.AdvertiseAddress,
		"insecure-port":                     "0",
		"admission-control":                 defaultv17AdmissionControl,
		"service-cluster-ip-range":          cfg.Networking.ServiceSubnet,
		"service-account-key-file":          filepath.Join(cfg.CertificatesDir, kubeadmconstants.ServiceAccountPublicKeyName),
		"client-ca-file":                    filepath.Join(cfg.CertificatesDir, kubeadmconstants.CACertName),
		"tls-cert-file":                     filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerCertName),
		"tls-private-key-file":              filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerKeyName),
		"kubelet-client-certificate":        filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerKubeletClientCertName),
		"kubelet-client-key":                filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerKubeletClientKeyName),
		"secure-port":                       fmt.Sprintf("%d", cfg.API.BindPort),
		"allow-privileged":                  "true",
		"experimental-bootstrap-token-auth": "true",
		"kubelet-preferred-address-types":   "InternalIP,ExternalIP,Hostname",
		// add options to configure the front proxy.  Without the generated client cert, this will never be useable
		// so add it unconditionally with recommended values
		"requestheader-username-headers":     "X-Remote-User",
		"requestheader-group-headers":        "X-Remote-Group",
		"requestheader-extra-headers-prefix": "X-Remote-Extra-",
		"requestheader-client-ca-file":       filepath.Join(cfg.CertificatesDir, kubeadmconstants.FrontProxyCACertName),
		"requestheader-allowed-names":        "front-proxy-client",
		"proxy-client-cert-file":             filepath.Join(cfg.CertificatesDir, kubeadmconstants.FrontProxyClientCertName),
		"proxy-client-key-file":              filepath.Join(cfg.CertificatesDir, kubeadmconstants.FrontProxyClientKeyName),
	}

	command := []string{"kube-apiserver"}
	command = append(command, staticpodutil.GetExtraParameters(cfg.APIServerExtraArgs, defaultArguments)...)
	command = append(command, getAuthzParameters(cfg.AuthorizationModes)...)

	// Check if the user decided to use an external etcd cluster
	if len(cfg.Etcd.Endpoints) > 0 {
		command = append(command, fmt.Sprintf("--etcd-servers=%s", strings.Join(cfg.Etcd.Endpoints, ",")))
	} else {
		command = append(command, "--etcd-servers=http://127.0.0.1:2379")
	}

	// Is etcd secured?
	if cfg.Etcd.CAFile != "" {
		command = append(command, fmt.Sprintf("--etcd-cafile=%s", cfg.Etcd.CAFile))
	}
	if cfg.Etcd.CertFile != "" && cfg.Etcd.KeyFile != "" {
		etcdClientFileArg := fmt.Sprintf("--etcd-certfile=%s", cfg.Etcd.CertFile)
		etcdKeyFileArg := fmt.Sprintf("--etcd-keyfile=%s", cfg.Etcd.KeyFile)
		command = append(command, etcdClientFileArg, etcdKeyFileArg)
	}

	if cfg.CloudProvider != "" {
		command = append(command, "--cloud-provider="+cfg.CloudProvider)

		// Only append the --cloud-config option if there's a such file
		if _, err := os.Stat(DefaultCloudConfigPath); err == nil {
			command = append(command, "--cloud-config="+DefaultCloudConfigPath)
		}
	}

	return command
}

// getControllerManagerCommand builds the right controller manager command from the given config object and version
func getControllerManagerCommand(cfg *kubeadmapi.MasterConfiguration, k8sVersion *version.Version) []string {
	defaultArguments := map[string]string{
		"address":                          "127.0.0.1",
		"leader-elect":                     "true",
		"kubeconfig":                       filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.ControllerManagerKubeConfigFileName),
		"root-ca-file":                     filepath.Join(cfg.CertificatesDir, kubeadmconstants.CACertName),
		"service-account-private-key-file": filepath.Join(cfg.CertificatesDir, kubeadmconstants.ServiceAccountPrivateKeyName),
		"cluster-signing-cert-file":        filepath.Join(cfg.CertificatesDir, kubeadmconstants.CACertName),
		"cluster-signing-key-file":         filepath.Join(cfg.CertificatesDir, kubeadmconstants.CAKeyName),
		"use-service-account-credentials":  "true",
		"controllers":                      "*,bootstrapsigner,tokencleaner",
	}

	command := []string{"kube-controller-manager"}
	command = append(command, staticpodutil.GetExtraParameters(cfg.ControllerManagerExtraArgs, defaultArguments)...)

	if cfg.CloudProvider != "" {
		command = append(command, "--cloud-provider="+cfg.CloudProvider)

		// Only append the --cloud-config option if there's a such file
		if _, err := os.Stat(DefaultCloudConfigPath); err == nil {
			command = append(command, "--cloud-config="+DefaultCloudConfigPath)
		}
	}

	// Let the controller-manager allocate Node CIDRs for the Pod network.
	// Each node will get a subspace of the address CIDR provided with --pod-network-cidr.
	if cfg.Networking.PodSubnet != "" {
		command = append(command, "--allocate-node-cidrs=true", "--cluster-cidr="+cfg.Networking.PodSubnet)
	}
	return command
}

// getSchedulerCommand builds the right scheduler command from the given config object and version
func getSchedulerCommand(cfg *kubeadmapi.MasterConfiguration) []string {
	defaultArguments := map[string]string{
		"address":      "127.0.0.1",
		"leader-elect": "true",
		"kubeconfig":   filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.SchedulerKubeConfigFileName),
	}

	command := []string{"kube-scheduler"}
	command = append(command, staticpodutil.GetExtraParameters(cfg.SchedulerExtraArgs, defaultArguments)...)
	return command
}

// getProxyEnvVars builds a list of environment variables to use in the control plane containers in order to use the right proxy
func getProxyEnvVars() []v1.EnvVar {
	envs := []v1.EnvVar{}
	for _, env := range os.Environ() {
		pos := strings.Index(env, "=")
		if pos == -1 {
			// malformed environment variable, skip it.
			continue
		}
		name := env[:pos]
		value := env[pos+1:]
		if strings.HasSuffix(strings.ToLower(name), "_proxy") && value != "" {
			envVar := v1.EnvVar{Name: name, Value: value}
			envs = append(envs, envVar)
		}
	}
	return envs
}

// getAuthzParameters gets the authorization-related parameters to the api server
// At this point, we can assume the list of authorization modes is valid (due to that it has been validated in the API machinery code already)
// If the list is empty; it's defaulted (mostly for unit testing)
func getAuthzParameters(modes []string) []string {
	command := []string{}
	strset := sets.NewString(modes...)

	if len(modes) == 0 {
		return []string{fmt.Sprintf("--authorization-mode=%s", kubeadmapiext.DefaultAuthorizationModes)}
	}

	if strset.Has(authzmodes.ModeABAC) {
		command = append(command, "--authorization-policy-file="+kubeadmconstants.AuthorizationPolicyPath)
	}
	if strset.Has(authzmodes.ModeWebhook) {
		command = append(command, "--authorization-webhook-config-file="+kubeadmconstants.AuthorizationWebhookConfigPath)
	}

	command = append(command, "--authorization-mode="+strings.Join(modes, ","))
	return command
}
