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
	"strconv"
	"strings"

	"github.com/pkg/errors"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	utilsnet "k8s.io/utils/net"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	certphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	staticpodutil "k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/users"
)

// CreateInitStaticPodManifestFiles will write all static pod manifest files needed to bring up the control plane.
func CreateInitStaticPodManifestFiles(manifestDir, patchesDir string, cfg *kubeadmapi.InitConfiguration, isDryRun bool) error {
	klog.V(1).Infoln("[control-plane] creating static Pod files")
	return CreateStaticPodFiles(manifestDir, patchesDir, &cfg.ClusterConfiguration, &cfg.LocalAPIEndpoint, isDryRun, kubeadmconstants.KubeAPIServer, kubeadmconstants.KubeControllerManager, kubeadmconstants.KubeScheduler)
}

// GetStaticPodSpecs returns all staticPodSpecs actualized to the context of the current configuration
// NB. this method holds the information about how kubeadm creates static pod manifests.
func GetStaticPodSpecs(cfg *kubeadmapi.ClusterConfiguration, endpoint *kubeadmapi.APIEndpoint, proxyEnvs []kubeadmapi.EnvVar) map[string]v1.Pod {
	// Get the required hostpath mounts
	mounts := getHostPathVolumesForTheControlPlane(cfg)
	if proxyEnvs == nil {
		proxyEnvs = kubeadmutil.GetProxyEnvVars(nil)
	}
	componentHealthCheckTimeout := kubeadmapi.GetActiveTimeouts().ControlPlaneComponentHealthCheck

	// Prepare static pod specs
	staticPodSpecs := map[string]v1.Pod{
		kubeadmconstants.KubeAPIServer: staticpodutil.ComponentPod(v1.Container{
			Name:            kubeadmconstants.KubeAPIServer,
			Image:           images.GetKubernetesImage(kubeadmconstants.KubeAPIServer, cfg),
			ImagePullPolicy: v1.PullIfNotPresent,
			Command:         getAPIServerCommand(cfg, endpoint),
			VolumeMounts:    staticpodutil.VolumeMountMapToSlice(mounts.GetVolumeMounts(kubeadmconstants.KubeAPIServer)),
			LivenessProbe:   staticpodutil.LivenessProbe(staticpodutil.GetAPIServerProbeAddress(endpoint), "/livez", endpoint.BindPort, v1.URISchemeHTTPS),
			ReadinessProbe:  staticpodutil.ReadinessProbe(staticpodutil.GetAPIServerProbeAddress(endpoint), "/readyz", endpoint.BindPort, v1.URISchemeHTTPS),
			StartupProbe:    staticpodutil.StartupProbe(staticpodutil.GetAPIServerProbeAddress(endpoint), "/livez", endpoint.BindPort, v1.URISchemeHTTPS, componentHealthCheckTimeout),
			Resources:       staticpodutil.ComponentResources("250m"),
			Env:             kubeadmutil.MergeKubeadmEnvVars(proxyEnvs, cfg.APIServer.ExtraEnvs),
		}, mounts.GetVolumes(kubeadmconstants.KubeAPIServer),
			map[string]string{kubeadmconstants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey: endpoint.String()}),
		kubeadmconstants.KubeControllerManager: staticpodutil.ComponentPod(v1.Container{
			Name:            kubeadmconstants.KubeControllerManager,
			Image:           images.GetKubernetesImage(kubeadmconstants.KubeControllerManager, cfg),
			ImagePullPolicy: v1.PullIfNotPresent,
			Command:         getControllerManagerCommand(cfg),
			VolumeMounts:    staticpodutil.VolumeMountMapToSlice(mounts.GetVolumeMounts(kubeadmconstants.KubeControllerManager)),
			LivenessProbe:   staticpodutil.LivenessProbe(staticpodutil.GetControllerManagerProbeAddress(cfg), "/healthz", kubeadmconstants.KubeControllerManagerPort, v1.URISchemeHTTPS),
			StartupProbe:    staticpodutil.StartupProbe(staticpodutil.GetControllerManagerProbeAddress(cfg), "/healthz", kubeadmconstants.KubeControllerManagerPort, v1.URISchemeHTTPS, componentHealthCheckTimeout),
			Resources:       staticpodutil.ComponentResources("200m"),
			Env:             kubeadmutil.MergeKubeadmEnvVars(proxyEnvs, cfg.ControllerManager.ExtraEnvs),
		}, mounts.GetVolumes(kubeadmconstants.KubeControllerManager), nil),
		kubeadmconstants.KubeScheduler: staticpodutil.ComponentPod(v1.Container{
			Name:            kubeadmconstants.KubeScheduler,
			Image:           images.GetKubernetesImage(kubeadmconstants.KubeScheduler, cfg),
			ImagePullPolicy: v1.PullIfNotPresent,
			Command:         getSchedulerCommand(cfg),
			VolumeMounts:    staticpodutil.VolumeMountMapToSlice(mounts.GetVolumeMounts(kubeadmconstants.KubeScheduler)),
			LivenessProbe:   staticpodutil.LivenessProbe(staticpodutil.GetSchedulerProbeAddress(cfg), "/healthz", kubeadmconstants.KubeSchedulerPort, v1.URISchemeHTTPS),
			StartupProbe:    staticpodutil.StartupProbe(staticpodutil.GetSchedulerProbeAddress(cfg), "/healthz", kubeadmconstants.KubeSchedulerPort, v1.URISchemeHTTPS, componentHealthCheckTimeout),
			Resources:       staticpodutil.ComponentResources("100m"),
			Env:             kubeadmutil.MergeKubeadmEnvVars(proxyEnvs, cfg.Scheduler.ExtraEnvs),
		}, mounts.GetVolumes(kubeadmconstants.KubeScheduler), nil),
	}
	return staticPodSpecs
}

// CreateStaticPodFiles creates all the requested static pod files.
func CreateStaticPodFiles(manifestDir, patchesDir string, cfg *kubeadmapi.ClusterConfiguration, endpoint *kubeadmapi.APIEndpoint, isDryRun bool, componentNames ...string) error {
	// gets the StaticPodSpecs, actualized for the current ClusterConfiguration
	klog.V(1).Infoln("[control-plane] getting StaticPodSpecs")
	specs := GetStaticPodSpecs(cfg, endpoint, nil)

	var usersAndGroups *users.UsersAndGroups
	var err error
	if features.Enabled(cfg.FeatureGates, features.RootlessControlPlane) {
		if isDryRun {
			fmt.Printf("[control-plane] Would create users and groups for %+v to run as non-root\n", componentNames)
		} else {
			usersAndGroups, err = staticpodutil.GetUsersAndGroups()
			if err != nil {
				return errors.Wrap(err, "failed to create users and groups")
			}
		}
	}

	// creates required static pod specs
	for _, componentName := range componentNames {
		// retrieves the StaticPodSpec for given component
		spec, exists := specs[componentName]
		if !exists {
			return errors.Errorf("couldn't retrieve StaticPodSpec for %q", componentName)
		}

		// print all volumes that are mounted
		for _, v := range spec.Spec.Volumes {
			klog.V(2).Infof("[control-plane] adding volume %q for component %q", v.Name, componentName)
		}

		if features.Enabled(cfg.FeatureGates, features.RootlessControlPlane) {
			if isDryRun {
				fmt.Printf("[control-plane] Would update static pod manifest for %q to run run as non-root\n", componentName)
			} else {
				if usersAndGroups != nil {
					if err := staticpodutil.RunComponentAsNonRoot(componentName, &spec, usersAndGroups, cfg); err != nil {
						return errors.Wrapf(err, "failed to run component %q as non-root", componentName)
					}
				}
			}
		}

		// if patchesDir is defined, patch the static Pod manifest
		if patchesDir != "" {
			patchedSpec, err := staticpodutil.PatchStaticPod(&spec, patchesDir, os.Stdout)
			if err != nil {
				return errors.Wrapf(err, "failed to patch static Pod manifest file for %q", componentName)
			}
			spec = *patchedSpec
		}

		// writes the StaticPodSpec to disk
		if err := staticpodutil.WriteStaticPodToDisk(componentName, manifestDir, spec); err != nil {
			return errors.Wrapf(err, "failed to create static pod manifest file for %q", componentName)
		}

		klog.V(1).Infof("[control-plane] wrote static Pod manifest for component %q to %q\n", componentName, kubeadmconstants.GetStaticPodFilepath(componentName, manifestDir))
	}

	return nil
}

// getAPIServerCommand builds the right API server command from the given config object and version
func getAPIServerCommand(cfg *kubeadmapi.ClusterConfiguration, localAPIEndpoint *kubeadmapi.APIEndpoint) []string {
	defaultArguments := []kubeadmapi.Arg{
		{Name: "advertise-address", Value: localAPIEndpoint.AdvertiseAddress},
		{Name: "enable-admission-plugins", Value: "NodeRestriction"},
		{Name: "service-cluster-ip-range", Value: cfg.Networking.ServiceSubnet},
		{Name: "service-account-key-file", Value: filepath.Join(cfg.CertificatesDir, kubeadmconstants.ServiceAccountPublicKeyName)},
		{Name: "service-account-signing-key-file", Value: filepath.Join(cfg.CertificatesDir, kubeadmconstants.ServiceAccountPrivateKeyName)},
		{Name: "service-account-issuer", Value: fmt.Sprintf("https://kubernetes.default.svc.%s", cfg.Networking.DNSDomain)},
		{Name: "client-ca-file", Value: filepath.Join(cfg.CertificatesDir, kubeadmconstants.CACertName)},
		{Name: "tls-cert-file", Value: filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerCertName)},
		{Name: "tls-private-key-file", Value: filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerKeyName)},
		{Name: "kubelet-client-certificate", Value: filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerKubeletClientCertName)},
		{Name: "kubelet-client-key", Value: filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerKubeletClientKeyName)},
		{Name: "enable-bootstrap-token-auth", Value: "true"},
		{Name: "secure-port", Value: fmt.Sprintf("%d", localAPIEndpoint.BindPort)},
		{Name: "allow-privileged", Value: "true"},
		{Name: "kubelet-preferred-address-types", Value: "InternalIP,ExternalIP,Hostname"},
		// add options to configure the front proxy.  Without the generated client cert, this will never be usable
		// so add it unconditionally with recommended values
		{Name: "requestheader-username-headers", Value: "X-Remote-User"},
		{Name: "requestheader-group-headers", Value: "X-Remote-Group"},
		{Name: "requestheader-extra-headers-prefix", Value: "X-Remote-Extra-"},
		{Name: "requestheader-client-ca-file", Value: filepath.Join(cfg.CertificatesDir, kubeadmconstants.FrontProxyCACertName)},
		{Name: "requestheader-allowed-names", Value: "front-proxy-client"},
		{Name: "proxy-client-cert-file", Value: filepath.Join(cfg.CertificatesDir, kubeadmconstants.FrontProxyClientCertName)},
		{Name: "proxy-client-key-file", Value: filepath.Join(cfg.CertificatesDir, kubeadmconstants.FrontProxyClientKeyName)},
	}

	command := []string{"kube-apiserver"}

	// If the user set endpoints for an external etcd cluster
	if cfg.Etcd.External != nil {
		defaultArguments = kubeadmapi.SetArgValues(defaultArguments, "etcd-servers", strings.Join(cfg.Etcd.External.Endpoints, ","), 1)

		// Use any user supplied etcd certificates
		if cfg.Etcd.External.CAFile != "" {
			defaultArguments = kubeadmapi.SetArgValues(defaultArguments, "etcd-cafile", cfg.Etcd.External.CAFile, 1)
		}
		if cfg.Etcd.External.CertFile != "" && cfg.Etcd.External.KeyFile != "" {
			defaultArguments = kubeadmapi.SetArgValues(defaultArguments, "etcd-certfile", cfg.Etcd.External.CertFile, 1)
			defaultArguments = kubeadmapi.SetArgValues(defaultArguments, "etcd-keyfile", cfg.Etcd.External.KeyFile, 1)

		}
	} else {
		// Default to etcd static pod on localhost
		// localhost IP family should be the same that the AdvertiseAddress
		etcdLocalhostAddress := "127.0.0.1"
		if utilsnet.IsIPv6String(localAPIEndpoint.AdvertiseAddress) {
			etcdLocalhostAddress = "::1"
		}
		defaultArguments = kubeadmapi.SetArgValues(defaultArguments, "etcd-servers", fmt.Sprintf("https://%s", net.JoinHostPort(etcdLocalhostAddress, strconv.Itoa(kubeadmconstants.EtcdListenClientPort))), 1)
		defaultArguments = kubeadmapi.SetArgValues(defaultArguments, "etcd-cafile", filepath.Join(cfg.CertificatesDir, kubeadmconstants.EtcdCACertName), 1)
		defaultArguments = kubeadmapi.SetArgValues(defaultArguments, "etcd-certfile", filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerEtcdClientCertName), 1)
		defaultArguments = kubeadmapi.SetArgValues(defaultArguments, "etcd-keyfile", filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerEtcdClientKeyName), 1)

		// Apply user configurations for local etcd
		if cfg.Etcd.Local != nil {
			if value, idx := kubeadmapi.GetArgValue(cfg.Etcd.Local.ExtraArgs, "advertise-client-urls", -1); idx > -1 {
				defaultArguments = kubeadmapi.SetArgValues(defaultArguments, "etcd-servers", value, 1)
			}
		}
	}

	if cfg.APIServer.ExtraArgs == nil {
		cfg.APIServer.ExtraArgs = []kubeadmapi.Arg{}
	}
	authzVal, _ := kubeadmapi.GetArgValue(cfg.APIServer.ExtraArgs, "authorization-mode", -1)
	_, hasStructuredAuthzVal := kubeadmapi.GetArgValue(cfg.APIServer.ExtraArgs, "authorization-config", -1)
	if hasStructuredAuthzVal == -1 {
		defaultArguments = kubeadmapi.SetArgValues(defaultArguments, "authorization-mode", getAuthzModes(authzVal), 1)
	}
	command = append(command, kubeadmutil.ArgumentsToCommand(defaultArguments, cfg.APIServer.ExtraArgs)...)

	return command
}

// getAuthzModes gets the authorization-related parameters to the api server
// Node,RBAC is the default mode if nothing is passed to kubeadm. User provided modes override
// the default.
func getAuthzModes(authzModeExtraArgs string) string {
	defaultMode := []string{
		kubeadmconstants.ModeNode,
		kubeadmconstants.ModeRBAC,
	}

	if len(authzModeExtraArgs) > 0 {
		mode := []string{}
		for _, requested := range strings.Split(authzModeExtraArgs, ",") {
			if isValidAuthzMode(requested) {
				mode = append(mode, requested)
			} else {
				klog.Warningf("ignoring unknown kube-apiserver authorization-mode %q", requested)
			}
		}

		// only return the user provided mode if at least one was valid
		if len(mode) > 0 {
			if !compareAuthzModes(defaultMode, mode) {
				klog.Warningf("the default kube-apiserver authorization-mode is %q; using %q",
					strings.Join(defaultMode, ","),
					strings.Join(mode, ","),
				)
			}
			return strings.Join(mode, ",")
		}
	}
	return strings.Join(defaultMode, ",")
}

// compareAuthzModes compares two given authz modes and returns false if they do not match
func compareAuthzModes(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i, m := range a {
		if m != b[i] {
			return false
		}
	}
	return true
}

func isValidAuthzMode(authzMode string) bool {
	allModes := []string{
		kubeadmconstants.ModeNode,
		kubeadmconstants.ModeRBAC,
		kubeadmconstants.ModeWebhook,
		kubeadmconstants.ModeABAC,
		kubeadmconstants.ModeAlwaysAllow,
		kubeadmconstants.ModeAlwaysDeny,
	}

	for _, mode := range allModes {
		if authzMode == mode {
			return true
		}
	}
	return false
}

// getControllerManagerCommand builds the right controller manager command from the given config object and version
func getControllerManagerCommand(cfg *kubeadmapi.ClusterConfiguration) []string {

	kubeconfigFile := filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.ControllerManagerKubeConfigFileName)
	caFile := filepath.Join(cfg.CertificatesDir, kubeadmconstants.CACertName)

	defaultArguments := []kubeadmapi.Arg{
		{Name: "bind-address", Value: "127.0.0.1"},
		{Name: "leader-elect", Value: "true"},
		{Name: "kubeconfig", Value: kubeconfigFile},
		{Name: "authentication-kubeconfig", Value: kubeconfigFile},
		{Name: "authorization-kubeconfig", Value: kubeconfigFile},
		{Name: "client-ca-file", Value: caFile},
		{Name: "requestheader-client-ca-file", Value: filepath.Join(cfg.CertificatesDir, kubeadmconstants.FrontProxyCACertName)},
		{Name: "root-ca-file", Value: caFile},
		{Name: "service-account-private-key-file", Value: filepath.Join(cfg.CertificatesDir, kubeadmconstants.ServiceAccountPrivateKeyName)},
		{Name: "cluster-signing-cert-file", Value: caFile},
		{Name: "cluster-signing-key-file", Value: filepath.Join(cfg.CertificatesDir, kubeadmconstants.CAKeyName)},
		{Name: "use-service-account-credentials", Value: "true"},
		{Name: "controllers", Value: "*,bootstrapsigner,tokencleaner"},
	}

	// If using external CA, pass empty string to controller manager instead of ca.key/ca.crt path,
	// so that the csrsigning controller fails to start
	if res, _ := certphase.UsingExternalCA(cfg); res {
		defaultArguments = kubeadmapi.SetArgValues(defaultArguments, "cluster-signing-key-file", "", 1)
		defaultArguments = kubeadmapi.SetArgValues(defaultArguments, "cluster-signing-cert-file", "", 1)
	}

	// Let the controller-manager allocate Node CIDRs for the Pod network.
	// Each node will get a subspace of the address CIDR provided with --pod-network-cidr.
	if cfg.Networking.PodSubnet != "" {
		defaultArguments = kubeadmapi.SetArgValues(defaultArguments, "allocate-node-cidrs", "true", 1)
		defaultArguments = kubeadmapi.SetArgValues(defaultArguments, "cluster-cidr", cfg.Networking.PodSubnet, 1)
		if cfg.Networking.ServiceSubnet != "" {
			defaultArguments = kubeadmapi.SetArgValues(defaultArguments, "service-cluster-ip-range", cfg.Networking.ServiceSubnet, 1)
		}
	}

	// Set cluster name
	if cfg.ClusterName != "" {
		defaultArguments = kubeadmapi.SetArgValues(defaultArguments, "cluster-name", cfg.ClusterName, 1)
	}

	command := []string{"kube-controller-manager"}
	command = append(command, kubeadmutil.ArgumentsToCommand(defaultArguments, cfg.ControllerManager.ExtraArgs)...)

	return command
}

// getSchedulerCommand builds the right scheduler command from the given config object and version
func getSchedulerCommand(cfg *kubeadmapi.ClusterConfiguration) []string {
	kubeconfigFile := filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.SchedulerKubeConfigFileName)
	defaultArguments := []kubeadmapi.Arg{
		{Name: "bind-address", Value: "127.0.0.1"},
		{Name: "leader-elect", Value: "true"},
		{Name: "kubeconfig", Value: kubeconfigFile},
		{Name: "authentication-kubeconfig", Value: kubeconfigFile},
		{Name: "authorization-kubeconfig", Value: kubeconfigFile},
	}

	command := []string{"kube-scheduler"}
	command = append(command, kubeadmutil.ArgumentsToCommand(defaultArguments, cfg.Scheduler.ExtraArgs)...)
	return command
}
