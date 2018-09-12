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

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1alpha3 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha3"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	certphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	staticpodutil "k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
	authzmodes "k8s.io/kubernetes/pkg/kubeapiserver/authorizer/modes"
	"k8s.io/kubernetes/pkg/util/version"
)

// CreateInitStaticPodManifestFiles will write all static pod manifest files needed to bring up the control plane.
func CreateInitStaticPodManifestFiles(manifestDir string, cfg *kubeadmapi.InitConfiguration) error {
	glog.V(1).Infoln("[controlplane] creating static pod files")
	return createStaticPodFiles(manifestDir, cfg, kubeadmconstants.KubeAPIServer, kubeadmconstants.KubeControllerManager, kubeadmconstants.KubeScheduler)
}

// CreateAPIServerStaticPodManifestFile will write APIserver static pod manifest file.
func CreateAPIServerStaticPodManifestFile(manifestDir string, cfg *kubeadmapi.InitConfiguration) error {
	glog.V(1).Infoln("creating APIserver static pod files")
	return createStaticPodFiles(manifestDir, cfg, kubeadmconstants.KubeAPIServer)
}

// CreateControllerManagerStaticPodManifestFile will write  controller manager static pod manifest file.
func CreateControllerManagerStaticPodManifestFile(manifestDir string, cfg *kubeadmapi.InitConfiguration) error {
	glog.V(1).Infoln("creating controller manager static pod files")
	return createStaticPodFiles(manifestDir, cfg, kubeadmconstants.KubeControllerManager)
}

// CreateSchedulerStaticPodManifestFile will write scheduler static pod manifest file.
func CreateSchedulerStaticPodManifestFile(manifestDir string, cfg *kubeadmapi.InitConfiguration) error {
	glog.V(1).Infoln("creating scheduler static pod files")
	return createStaticPodFiles(manifestDir, cfg, kubeadmconstants.KubeScheduler)
}

// GetStaticPodSpecs returns all staticPodSpecs actualized to the context of the current InitConfiguration
// NB. this methods holds the information about how kubeadm creates static pod manifests.
func GetStaticPodSpecs(cfg *kubeadmapi.InitConfiguration, k8sVersion *version.Version) map[string]v1.Pod {
	// Get the required hostpath mounts
	mounts := getHostPathVolumesForTheControlPlane(cfg)

	// Prepare static pod specs
	staticPodSpecs := map[string]v1.Pod{
		kubeadmconstants.KubeAPIServer: staticpodutil.ComponentPod(v1.Container{
			Name:            kubeadmconstants.KubeAPIServer,
			Image:           images.GetKubeControlPlaneImage(kubeadmconstants.KubeAPIServer, &cfg.ClusterConfiguration),
			ImagePullPolicy: v1.PullIfNotPresent,
			Command:         getAPIServerCommand(cfg),
			VolumeMounts:    staticpodutil.VolumeMountMapToSlice(mounts.GetVolumeMounts(kubeadmconstants.KubeAPIServer)),
			LivenessProbe:   staticpodutil.ComponentProbe(cfg, kubeadmconstants.KubeAPIServer, int(cfg.APIEndpoint.BindPort), "/healthz", v1.URISchemeHTTPS),
			Resources:       staticpodutil.ComponentResources("250m"),
			Env:             getProxyEnvVars(),
		}, mounts.GetVolumes(kubeadmconstants.KubeAPIServer)),
		kubeadmconstants.KubeControllerManager: staticpodutil.ComponentPod(v1.Container{
			Name:            kubeadmconstants.KubeControllerManager,
			Image:           images.GetKubeControlPlaneImage(kubeadmconstants.KubeControllerManager, &cfg.ClusterConfiguration),
			ImagePullPolicy: v1.PullIfNotPresent,
			Command:         getControllerManagerCommand(cfg, k8sVersion),
			VolumeMounts:    staticpodutil.VolumeMountMapToSlice(mounts.GetVolumeMounts(kubeadmconstants.KubeControllerManager)),
			LivenessProbe:   staticpodutil.ComponentProbe(cfg, kubeadmconstants.KubeControllerManager, 10252, "/healthz", v1.URISchemeHTTP),
			Resources:       staticpodutil.ComponentResources("200m"),
			Env:             getProxyEnvVars(),
		}, mounts.GetVolumes(kubeadmconstants.KubeControllerManager)),
		kubeadmconstants.KubeScheduler: staticpodutil.ComponentPod(v1.Container{
			Name:            kubeadmconstants.KubeScheduler,
			Image:           images.GetKubeControlPlaneImage(kubeadmconstants.KubeScheduler, &cfg.ClusterConfiguration),
			ImagePullPolicy: v1.PullIfNotPresent,
			Command:         getSchedulerCommand(cfg),
			VolumeMounts:    staticpodutil.VolumeMountMapToSlice(mounts.GetVolumeMounts(kubeadmconstants.KubeScheduler)),
			LivenessProbe:   staticpodutil.ComponentProbe(cfg, kubeadmconstants.KubeScheduler, 10251, "/healthz", v1.URISchemeHTTP),
			Resources:       staticpodutil.ComponentResources("100m"),
			Env:             getProxyEnvVars(),
		}, mounts.GetVolumes(kubeadmconstants.KubeScheduler)),
	}
	return staticPodSpecs
}

// createStaticPodFiles creates all the requested static pod files.
func createStaticPodFiles(manifestDir string, cfg *kubeadmapi.InitConfiguration, componentNames ...string) error {
	// TODO: Move the "pkg/util/version".Version object into the internal API instead of always parsing the string
	k8sVersion, err := version.ParseSemantic(cfg.KubernetesVersion)
	if err != nil {
		return err
	}

	// gets the StaticPodSpecs, actualized for the current InitConfiguration
	glog.V(1).Infoln("[controlplane] getting StaticPodSpecs")
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

		fmt.Printf("[controlplane] wrote Static Pod manifest for component %s to %q\n", componentName, kubeadmconstants.GetStaticPodFilepath(componentName, manifestDir))
	}

	return nil
}

// getAPIServerCommand builds the right API server command from the given config object and version
func getAPIServerCommand(cfg *kubeadmapi.InitConfiguration) []string {
	defaultArguments := map[string]string{
		"advertise-address":               cfg.APIEndpoint.AdvertiseAddress,
		"insecure-port":                   "0",
		"enable-admission-plugins":        "NodeRestriction",
		"service-cluster-ip-range":        cfg.Networking.ServiceSubnet,
		"service-account-key-file":        filepath.Join(cfg.CertificatesDir, kubeadmconstants.ServiceAccountPublicKeyName),
		"client-ca-file":                  filepath.Join(cfg.CertificatesDir, kubeadmconstants.CACertName),
		"tls-cert-file":                   filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerCertName),
		"tls-private-key-file":            filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerKeyName),
		"kubelet-client-certificate":      filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerKubeletClientCertName),
		"kubelet-client-key":              filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerKubeletClientKeyName),
		"enable-bootstrap-token-auth":     "true",
		"secure-port":                     fmt.Sprintf("%d", cfg.APIEndpoint.BindPort),
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

	// If the user set endpoints for an external etcd cluster
	if cfg.Etcd.External != nil {
		defaultArguments["etcd-servers"] = strings.Join(cfg.Etcd.External.Endpoints, ",")

		// Use any user supplied etcd certificates
		if cfg.Etcd.External.CAFile != "" {
			defaultArguments["etcd-cafile"] = cfg.Etcd.External.CAFile
		}
		if cfg.Etcd.External.CertFile != "" && cfg.Etcd.External.KeyFile != "" {
			defaultArguments["etcd-certfile"] = cfg.Etcd.External.CertFile
			defaultArguments["etcd-keyfile"] = cfg.Etcd.External.KeyFile
		}
	} else {
		// Default to etcd static pod on localhost
		defaultArguments["etcd-servers"] = "https://127.0.0.1:2379"
		defaultArguments["etcd-cafile"] = filepath.Join(cfg.CertificatesDir, kubeadmconstants.EtcdCACertName)
		defaultArguments["etcd-certfile"] = filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerEtcdClientCertName)
		defaultArguments["etcd-keyfile"] = filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerEtcdClientKeyName)

		// Apply user configurations for local etcd
		if cfg.Etcd.Local != nil {
			if value, ok := cfg.Etcd.Local.ExtraArgs["listen-client-urls"]; ok {
				defaultArguments["etcd-servers"] = value
			}
		}
	}

	if features.Enabled(cfg.FeatureGates, features.HighAvailability) {
		defaultArguments["endpoint-reconciler-type"] = kubeadmconstants.LeaseEndpointReconcilerType
	}

	if features.Enabled(cfg.FeatureGates, features.DynamicKubeletConfig) {
		defaultArguments["feature-gates"] = "DynamicKubeletConfig=true"
	}

	if features.Enabled(cfg.FeatureGates, features.Auditing) {
		defaultArguments["audit-policy-file"] = kubeadmconstants.GetStaticPodAuditPolicyFile()
		defaultArguments["audit-log-path"] = filepath.Join(kubeadmconstants.StaticPodAuditPolicyLogDir, kubeadmconstants.AuditPolicyLogFile)
		if cfg.AuditPolicyConfiguration.LogMaxAge == nil {
			defaultArguments["audit-log-maxage"] = fmt.Sprintf("%d", kubeadmapiv1alpha3.DefaultAuditPolicyLogMaxAge)
		} else {
			defaultArguments["audit-log-maxage"] = fmt.Sprintf("%d", *cfg.AuditPolicyConfiguration.LogMaxAge)
		}
	}
	if cfg.APIServerExtraArgs == nil {
		cfg.APIServerExtraArgs = map[string]string{}
	}
	cfg.APIServerExtraArgs["authorization-mode"] = getAuthzModes(cfg.APIServerExtraArgs["authorization-mode"])
	command = append(command, kubeadmutil.BuildArgumentListFromMap(defaultArguments, cfg.APIServerExtraArgs)...)

	return command
}

// getAuthzModes gets the authorization-related parameters to the api server
// Node,RBAC should be fixed in this order at the beginning
// AlwaysAllow and AlwaysDeny is ignored as they are only for testing
func getAuthzModes(authzModeExtraArgs string) string {
	modes := []string{
		authzmodes.ModeNode,
		authzmodes.ModeRBAC,
	}
	if strings.Contains(authzModeExtraArgs, authzmodes.ModeABAC) {
		modes = append(modes, authzmodes.ModeABAC)
	}
	if strings.Contains(authzModeExtraArgs, authzmodes.ModeWebhook) {
		modes = append(modes, authzmodes.ModeWebhook)
	}
	return strings.Join(modes, ",")
}

// calcNodeCidrSize determines the size of the subnets used on each node, based
// on the pod subnet provided.  For IPv4, we assume that the pod subnet will
// be /16 and use /24. If the pod subnet cannot be parsed, the IPv4 value will
// be used (/24).
//
// For IPv6, the algorithm will do two three. First, the node CIDR will be set
// to a multiple of 8, using the available bits for easier readability by user.
// Second, the number of nodes will be 512 to 64K to attempt to maximize the
// number of nodes (see NOTE below). Third, pod networks of /113 and larger will
// be rejected, as the amount of bits available is too small.
//
// A special case is when the pod network size is /112, where /120 will be used,
// only allowing 256 nodes and 256 pods.
//
// If the pod network size is /113 or larger, the node CIDR will be set to the same
// size and this will be rejected later in validation.
//
// NOTE: Currently, the design allows a maximum of 64K nodes. This algorithm splits
// the available bits to maximize the number used for nodes, but still have the node
// CIDR be a multiple of eight.
//
func calcNodeCidrSize(podSubnet string) string {
	maskSize := "24"
	if ip, podCidr, err := net.ParseCIDR(podSubnet); err == nil {
		if ip.To4() == nil {
			var nodeCidrSize int
			podNetSize, totalBits := podCidr.Mask.Size()
			switch {
			case podNetSize == 112:
				// Special case, allows 256 nodes, 256 pods/node
				nodeCidrSize = 120
			case podNetSize < 112:
				// Use multiple of 8 for node CIDR, with 512 to 64K nodes
				nodeCidrSize = totalBits - ((totalBits-podNetSize-1)/8-1)*8
			default:
				// Not enough bits, will fail later, when validate
				nodeCidrSize = podNetSize
			}
			maskSize = strconv.Itoa(nodeCidrSize)
		}
	}
	return maskSize
}

// getControllerManagerCommand builds the right controller manager command from the given config object and version
func getControllerManagerCommand(cfg *kubeadmapi.InitConfiguration, k8sVersion *version.Version) []string {
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

	//add the extra arguments for v1.12+
	if k8sVersion.Major() >= 1 && k8sVersion.Minor() >= 12 {
		defaultArguments["authentication-kubeconfig"] = filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.ControllerManagerKubeConfigFileName)
		defaultArguments["authorization-kubeconfig"] = filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.ControllerManagerKubeConfigFileName)
		defaultArguments["client-ca-file"] = filepath.Join(cfg.CertificatesDir, kubeadmconstants.CACertName)
		defaultArguments["requestheader-client-ca-file"] = filepath.Join(cfg.CertificatesDir, kubeadmconstants.FrontProxyCACertName)
	}

	// If using external CA, pass empty string to controller manager instead of ca.key/ca.crt path,
	// so that the csrsigning controller fails to start
	if res, _ := certphase.UsingExternalCA(cfg); res {
		defaultArguments["cluster-signing-key-file"] = ""
		defaultArguments["cluster-signing-cert-file"] = ""
	}

	// Let the controller-manager allocate Node CIDRs for the Pod network.
	// Each node will get a subspace of the address CIDR provided with --pod-network-cidr.
	if cfg.Networking.PodSubnet != "" {
		maskSize := calcNodeCidrSize(cfg.Networking.PodSubnet)
		defaultArguments["allocate-node-cidrs"] = "true"
		defaultArguments["cluster-cidr"] = cfg.Networking.PodSubnet
		defaultArguments["node-cidr-mask-size"] = maskSize
	}

	command := []string{"kube-controller-manager"}
	command = append(command, kubeadmutil.BuildArgumentListFromMap(defaultArguments, cfg.ControllerManagerExtraArgs)...)

	return command
}

// getSchedulerCommand builds the right scheduler command from the given config object and version
func getSchedulerCommand(cfg *kubeadmapi.InitConfiguration) []string {
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
