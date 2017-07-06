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
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ghodss/yaml"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	authzmodes "k8s.io/kubernetes/pkg/kubeapiserver/authorizer/modes"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/util/version"
)

// Static pod definitions in golang form are included below so that `kubeadm init` can get going.
const (
	DefaultCloudConfigPath = "/etc/kubernetes/cloud-config"

	defaultv17AdmissionControl = "Initializers,NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,ResourceQuota"

	etcd                  = "etcd"
	kubeAPIServer         = "kube-apiserver"
	kubeControllerManager = "kube-controller-manager"
	kubeScheduler         = "kube-scheduler"
)

// WriteStaticPodManifests builds manifest objects based on user provided configuration and then dumps it to disk
// where kubelet will pick and schedule them.
func WriteStaticPodManifests(cfg *kubeadmapi.MasterConfiguration) error {
	volumes := []v1.Volume{k8sVolume()}
	volumeMounts := []v1.VolumeMount{k8sVolumeMount()}

	if isCertsVolumeMountNeeded() {
		volumes = append(volumes, certsVolume(cfg))
		volumeMounts = append(volumeMounts, certsVolumeMount())
	}

	if isPkiVolumeMountNeeded() {
		volumes = append(volumes, pkiVolume())
		volumeMounts = append(volumeMounts, pkiVolumeMount())
	}

	if !strings.HasPrefix(cfg.CertificatesDir, kubeadmapiext.DefaultCertificatesDir) {
		volumes = append(volumes, newVolume("certdir", cfg.CertificatesDir))
		volumeMounts = append(volumeMounts, newVolumeMount("certdir", cfg.CertificatesDir))
	}

	k8sVersion, err := version.ParseSemantic(cfg.KubernetesVersion)
	if err != nil {
		return err
	}

	// Prepare static pod specs
	staticPodSpecs := map[string]v1.Pod{
		kubeAPIServer: componentPod(v1.Container{
			Name:          kubeAPIServer,
			Image:         images.GetCoreImage(images.KubeAPIServerImage, cfg, kubeadmapi.GlobalEnvParams.HyperkubeImage),
			Command:       getAPIServerCommand(cfg, false, k8sVersion),
			VolumeMounts:  volumeMounts,
			LivenessProbe: componentProbe(int(cfg.API.BindPort), "/healthz", v1.URISchemeHTTPS),
			Resources:     componentResources("250m"),
			Env:           getProxyEnvVars(),
		}, volumes...),
		kubeControllerManager: componentPod(v1.Container{
			Name:          kubeControllerManager,
			Image:         images.GetCoreImage(images.KubeControllerManagerImage, cfg, kubeadmapi.GlobalEnvParams.HyperkubeImage),
			Command:       getControllerManagerCommand(cfg, false, k8sVersion),
			VolumeMounts:  volumeMounts,
			LivenessProbe: componentProbe(10252, "/healthz", v1.URISchemeHTTP),
			Resources:     componentResources("200m"),
			Env:           getProxyEnvVars(),
		}, volumes...),
		kubeScheduler: componentPod(v1.Container{
			Name:          kubeScheduler,
			Image:         images.GetCoreImage(images.KubeSchedulerImage, cfg, kubeadmapi.GlobalEnvParams.HyperkubeImage),
			Command:       getSchedulerCommand(cfg, false),
			VolumeMounts:  []v1.VolumeMount{k8sVolumeMount()},
			LivenessProbe: componentProbe(10251, "/healthz", v1.URISchemeHTTP),
			Resources:     componentResources("100m"),
			Env:           getProxyEnvVars(),
		}, k8sVolume()),
	}

	// Add etcd static pod spec only if external etcd is not configured
	if len(cfg.Etcd.Endpoints) == 0 {
		etcdPod := componentPod(v1.Container{
			Name:          etcd,
			Command:       getEtcdCommand(cfg),
			VolumeMounts:  []v1.VolumeMount{certsVolumeMount(), etcdVolumeMount(cfg.Etcd.DataDir), k8sVolumeMount()},
			Image:         images.GetCoreImage(images.KubeEtcdImage, cfg, kubeadmapi.GlobalEnvParams.EtcdImage),
			LivenessProbe: componentProbe(2379, "/health", v1.URISchemeHTTP),
		}, certsVolume(cfg), etcdVolume(cfg), k8sVolume())

		etcdPod.Spec.SecurityContext = &v1.PodSecurityContext{
			SELinuxOptions: &v1.SELinuxOptions{
				// Unconfine the etcd container so it can write to the data dir with SELinux enforcing:
				Type: "spc_t",
			},
		}

		staticPodSpecs[etcd] = etcdPod
	}

	manifestsPath := filepath.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, kubeadmconstants.ManifestsSubDirName)
	if err := os.MkdirAll(manifestsPath, 0700); err != nil {
		return fmt.Errorf("failed to create directory %q [%v]", manifestsPath, err)
	}
	for name, spec := range staticPodSpecs {
		filename := filepath.Join(manifestsPath, name+".yaml")
		serialized, err := yaml.Marshal(spec)
		if err != nil {
			return fmt.Errorf("failed to marshal manifest for %q to YAML [%v]", name, err)
		}
		if err := cmdutil.DumpReaderToFile(bytes.NewReader(serialized), filename); err != nil {
			return fmt.Errorf("failed to create static pod manifest file for %q (%q) [%v]", name, filename, err)
		}
	}
	return nil
}

func newVolume(name, path string) v1.Volume {
	return v1.Volume{
		Name: name,
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{Path: path},
		},
	}
}

func newVolumeMount(name, path string) v1.VolumeMount {
	return v1.VolumeMount{
		Name:      name,
		MountPath: path,
	}
}

// etcdVolume exposes a path on the host in order to guarantee data survival during reboot.
func etcdVolume(cfg *kubeadmapi.MasterConfiguration) v1.Volume {
	return v1.Volume{
		Name: "etcd",
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{Path: cfg.Etcd.DataDir},
		},
	}
}

func etcdVolumeMount(dataDir string) v1.VolumeMount {
	return v1.VolumeMount{
		Name:      "etcd",
		MountPath: dataDir,
	}
}

func isCertsVolumeMountNeeded() bool {
	// Always return true for now. We may add conditional logic here for images which do not require host mounting /etc/ssl
	// hyperkube for example already has valid ca-certificates installed
	return true
}

// certsVolume exposes host SSL certificates to pod containers.
func certsVolume(cfg *kubeadmapi.MasterConfiguration) v1.Volume {
	return v1.Volume{
		Name: "certs",
		VolumeSource: v1.VolumeSource{
			// TODO(phase1+) make path configurable
			HostPath: &v1.HostPathVolumeSource{Path: "/etc/ssl/certs"},
		},
	}
}

func certsVolumeMount() v1.VolumeMount {
	return v1.VolumeMount{
		Name:      "certs",
		MountPath: "/etc/ssl/certs",
	}
}

func isPkiVolumeMountNeeded() bool {
	// On some systems were we host-mount /etc/ssl/certs, it is also required to mount /etc/pki. This is needed
	// due to symlinks pointing from files in /etc/ssl/certs into /etc/pki/
	if _, err := os.Stat("/etc/pki"); err == nil {
		return true
	}
	return false
}

func pkiVolume() v1.Volume {
	return v1.Volume{
		Name: "pki",
		VolumeSource: v1.VolumeSource{
			// TODO(phase1+) make path configurable
			HostPath: &v1.HostPathVolumeSource{Path: "/etc/pki"},
		},
	}
}

func pkiVolumeMount() v1.VolumeMount {
	return v1.VolumeMount{
		Name:      "pki",
		MountPath: "/etc/pki",
	}
}

func k8sVolume() v1.Volume {
	return v1.Volume{
		Name: "k8s",
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{Path: kubeadmapi.GlobalEnvParams.KubernetesDir},
		},
	}
}

func k8sVolumeMount() v1.VolumeMount {
	return v1.VolumeMount{
		Name:      "k8s",
		MountPath: kubeadmapi.GlobalEnvParams.KubernetesDir,
		ReadOnly:  true,
	}
}

func componentResources(cpu string) v1.ResourceRequirements {
	return v1.ResourceRequirements{
		Requests: v1.ResourceList{
			v1.ResourceName(v1.ResourceCPU): resource.MustParse(cpu),
		},
	}
}

func componentProbe(port int, path string, scheme v1.URIScheme) *v1.Probe {
	return &v1.Probe{
		Handler: v1.Handler{
			HTTPGet: &v1.HTTPGetAction{
				Host:   "127.0.0.1",
				Path:   path,
				Port:   intstr.FromInt(port),
				Scheme: scheme,
			},
		},
		InitialDelaySeconds: 15,
		TimeoutSeconds:      15,
		FailureThreshold:    8,
	}
}

func componentPod(container v1.Container, volumes ...v1.Volume) v1.Pod {
	return v1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Pod",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:        container.Name,
			Namespace:   "kube-system",
			Annotations: map[string]string{kubetypes.CriticalPodAnnotationKey: ""},
			Labels:      map[string]string{"component": container.Name, "tier": "control-plane"},
		},
		Spec: v1.PodSpec{
			Containers:  []v1.Container{container},
			HostNetwork: true,
			Volumes:     volumes,
		},
	}
}

func getAPIServerCommand(cfg *kubeadmapi.MasterConfiguration, selfHosted bool, k8sVersion *version.Version) []string {
	defaultArguments := map[string]string{
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
	command = append(command, getExtraParameters(cfg.APIServerExtraArgs, defaultArguments)...)
	command = append(command, getAuthzParameters(cfg.AuthorizationModes)...)

	if selfHosted {
		command = append(command, "--advertise-address=$(POD_IP)")
	} else {
		command = append(command, fmt.Sprintf("--advertise-address=%s", cfg.API.AdvertiseAddress))
	}

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

func getEtcdCommand(cfg *kubeadmapi.MasterConfiguration) []string {
	defaultArguments := map[string]string{
		"listen-client-urls":    "http://127.0.0.1:2379",
		"advertise-client-urls": "http://127.0.0.1:2379",
		"data-dir":              cfg.Etcd.DataDir,
	}

	command := []string{"etcd"}
	command = append(command, getExtraParameters(cfg.Etcd.ExtraArgs, defaultArguments)...)
	return command
}

func getControllerManagerCommand(cfg *kubeadmapi.MasterConfiguration, selfHosted bool, k8sVersion *version.Version) []string {
	defaultArguments := map[string]string{
		"address":                          "127.0.0.1",
		"leader-elect":                     "true",
		"kubeconfig":                       filepath.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, kubeadmconstants.ControllerManagerKubeConfigFileName),
		"root-ca-file":                     filepath.Join(cfg.CertificatesDir, kubeadmconstants.CACertName),
		"service-account-private-key-file": filepath.Join(cfg.CertificatesDir, kubeadmconstants.ServiceAccountPrivateKeyName),
		"cluster-signing-cert-file":        filepath.Join(cfg.CertificatesDir, kubeadmconstants.CACertName),
		"cluster-signing-key-file":         filepath.Join(cfg.CertificatesDir, kubeadmconstants.CAKeyName),
		"use-service-account-credentials":  "true",
		"controllers":                      "*,bootstrapsigner,tokencleaner",
	}

	command := []string{"kube-controller-manager"}
	command = append(command, getExtraParameters(cfg.ControllerManagerExtraArgs, defaultArguments)...)

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

func getSchedulerCommand(cfg *kubeadmapi.MasterConfiguration, selfHosted bool) []string {
	defaultArguments := map[string]string{
		"address":      "127.0.0.1",
		"leader-elect": "true",
		"kubeconfig":   filepath.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, kubeadmconstants.SchedulerKubeConfigFileName),
	}

	command := []string{"kube-scheduler"}
	command = append(command, getExtraParameters(cfg.SchedulerExtraArgs, defaultArguments)...)
	return command
}

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

func getExtraParameters(overrides map[string]string, defaults map[string]string) []string {
	var command []string
	for k, v := range overrides {
		if len(v) > 0 {
			command = append(command, fmt.Sprintf("--%s=%s", k, v))
		}
	}
	for k, v := range defaults {
		if _, overrideExists := overrides[k]; !overrideExists {
			command = append(command, fmt.Sprintf("--%s=%s", k, v))
		}
	}
	return command
}
