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
	"net"
	"os"
	"path/filepath"
	"strings"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	certphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	staticpodutil "k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
	authzmodes "k8s.io/kubernetes/pkg/kubeapiserver/authorizer/modes"
	"k8s.io/kubernetes/pkg/master/reconcilers"
	"k8s.io/kubernetes/pkg/util/version"
)

// Static pod definitions in golang form are included below so that `kubeadm init` can get going.
const (
	DefaultCloudConfigPath = "/etc/kubernetes/cloud-config"

	defaultV18AdmissionControl    = "Initializers,NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,ResourceQuota"
	deprecatedV19AdmissionControl = "Initializers,NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,ResourceQuota"
	defaultV19AdmissionControl    = "Initializers,NamespaceLifecycle,LimitRanger,ServiceAccount,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,ResourceQuota"
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
			Image:         images.GetCoreImage(kubeadmconstants.KubeAPIServer, cfg.GetControlPlaneImageRepository(), cfg.KubernetesVersion, cfg.UnifiedControlPlaneImage),
			Command:       getAPIServerCommand(cfg, k8sVersion),
			VolumeMounts:  staticpodutil.VolumeMountMapToSlice(mounts.GetVolumeMounts(kubeadmconstants.KubeAPIServer)),
			LivenessProbe: staticpodutil.ComponentProbe(cfg, kubeadmconstants.KubeAPIServer, int(cfg.API.BindPort), "/healthz", v1.URISchemeHTTPS),
			Resources:     staticpodutil.ComponentResources("250m"),
			Env:           getProxyEnvVars(),
		}, mounts.GetVolumes(kubeadmconstants.KubeAPIServer)),
		kubeadmconstants.KubeControllerManager: staticpodutil.ComponentPod(v1.Container{
			Name:          kubeadmconstants.KubeControllerManager,
			Image:         images.GetCoreImage(kubeadmconstants.KubeControllerManager, cfg.GetControlPlaneImageRepository(), cfg.KubernetesVersion, cfg.UnifiedControlPlaneImage),
			Command:       getControllerManagerCommand(cfg, k8sVersion),
			VolumeMounts:  staticpodutil.VolumeMountMapToSlice(mounts.GetVolumeMounts(kubeadmconstants.KubeControllerManager)),
			LivenessProbe: staticpodutil.ComponentProbe(cfg, kubeadmconstants.KubeControllerManager, 10252, "/healthz", v1.URISchemeHTTP),
			Resources:     staticpodutil.ComponentResources("200m"),
			Env:           getProxyEnvVars(),
		}, mounts.GetVolumes(kubeadmconstants.KubeControllerManager)),
		kubeadmconstants.KubeScheduler: staticpodutil.ComponentPod(v1.Container{
			Name:          kubeadmconstants.KubeScheduler,
			Image:         images.GetCoreImage(kubeadmconstants.KubeScheduler, cfg.GetControlPlaneImageRepository(), cfg.KubernetesVersion, cfg.UnifiedControlPlaneImage),
			Command:       getSchedulerCommand(cfg),
			VolumeMounts:  staticpodutil.VolumeMountMapToSlice(mounts.GetVolumeMounts(kubeadmconstants.KubeScheduler)),
			LivenessProbe: staticpodutil.ComponentProbe(cfg, kubeadmconstants.KubeScheduler, 10251, "/healthz", v1.URISchemeHTTP),
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

		fmt.Printf("[controlplane] Wrote Static Pod manifest for component %s to %q\n", componentName, kubeadmconstants.GetStaticPodFilepath(componentName, manifestDir))
	}

	return nil
}

// getAPIServerCommand builds the right API server command from the given config object and version
func getAPIServerCommand(cfg *kubeadmapi.MasterConfiguration, k8sVersion *version.Version) []string {
	defaultArguments := map[string]string{
		"advertise-address":               cfg.API.AdvertiseAddress,
		"insecure-port":                   "0",
		"admission-control":               defaultV19AdmissionControl,
		"service-cluster-ip-range":        cfg.Networking.ServiceSubnet,
		"service-account-key-file":        filepath.Join(cfg.CertificatesDir, kubeadmconstants.ServiceAccountPublicKeyName),
		"client-ca-file":                  filepath.Join(cfg.CertificatesDir, kubeadmconstants.CACertName),
		"tls-cert-file":                   filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerCertName),
		"tls-private-key-file":            filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerKeyName),
		"kubelet-client-certificate":      filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerKubeletClientCertName),
		"kubelet-client-key":              filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerKubeletClientKeyName),
		"enable-bootstrap-token-auth":     "true",
		"secure-port":                     fmt.Sprintf("%d", cfg.API.BindPort),
		"allow-privileged":                "true",
		"kubelet-preferred-address-types": "InternalIP,ExternalIP,Hostname",
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

	if k8sVersion.Minor() == 8 {
		defaultArguments["admission-control"] = defaultV18AdmissionControl
	}

	if cfg.CloudProvider == "aws" || cfg.CloudProvider == "gce" {
		defaultArguments["admission-control"] = deprecatedV19AdmissionControl
	}

	command = append(command, kubeadmutil.BuildArgumentListFromMap(defaultArguments, cfg.APIServerExtraArgs)...)
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

	if features.Enabled(cfg.FeatureGates, features.HighAvailability) {
		command = append(command, "--endpoint-reconciler-type="+reconcilers.LeaseEndpointReconcilerType)
	}

	if features.Enabled(cfg.FeatureGates, features.DynamicKubeletConfig) {
		command = append(command, "--feature-gates=DynamicKubeletConfig=true")
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

	// If using external CA, pass empty string to controller manager instead of ca.key/ca.crt path,
	// so that the csrsigning controller fails to start
	if res, _ := certphase.UsingExternalCA(cfg); res {
		defaultArguments["cluster-signing-key-file"] = ""
		defaultArguments["cluster-signing-cert-file"] = ""
	}

	command := []string{"kube-controller-manager"}
	command = append(command, kubeadmutil.BuildArgumentListFromMap(defaultArguments, cfg.ControllerManagerExtraArgs)...)

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
		maskSize := "24"
		if ip, _, err := net.ParseCIDR(cfg.Networking.PodSubnet); err == nil {
			if ip.To4() == nil {
				maskSize = "64"
			}
		}
		command = append(command, "--allocate-node-cidrs=true", "--cluster-cidr="+cfg.Networking.PodSubnet,
			"--node-cidr-mask-size="+maskSize)
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
	command = append(command, kubeadmutil.BuildArgumentListFromMap(defaultArguments, cfg.SchedulerExtraArgs)...)
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
