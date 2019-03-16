/*
Copyright 2017 The Kubernetes Authors.

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

package image

import (
	"fmt"
	"io/ioutil"
	"os"

	yaml "gopkg.in/yaml.v2"
)

// RegistryList holds public and private image registries
type RegistryList struct {
	DockerLibraryRegistry string `yaml:"dockerLibraryRegistry"`
	E2eRegistry           string `yaml:"e2eRegistry"`
	EtcdRegistry          string `yaml:"etcdRegistry"`
	GcRegistry            string `yaml:"gcRegistry"`
	PrivateRegistry       string `yaml:"privateRegistry"`
	SampleRegistry        string `yaml:"sampleRegistry"`
}

// Config holds an images registry, name, and version
type Config struct {
	registry string
	name     string
	version  string
}

// SetRegistry sets an image registry in a Config struct
func (i *Config) SetRegistry(registry string) {
	i.registry = registry
}

// SetName sets an image name in a Config struct
func (i *Config) SetName(name string) {
	i.name = name
}

// SetVersion sets an image version in a Config struct
func (i *Config) SetVersion(version string) {
	i.version = version
}

func initReg() RegistryList {
	registry := RegistryList{
		DockerLibraryRegistry: "docker.io/library",
		E2eRegistry:           "gcr.io/kubernetes-e2e-test-images",
		EtcdRegistry:          "quay.io/coreos",
		GcRegistry:            "k8s.gcr.io",
		PrivateRegistry:       "gcr.io/k8s-authenticated-test",
		SampleRegistry:        "gcr.io/google-samples",
	}
	repoList := os.Getenv("KUBE_TEST_REPO_LIST")
	if repoList == "" {
		return registry
	}

	fileContent, err := ioutil.ReadFile(repoList)
	if err != nil {
		panic(fmt.Errorf("Error reading '%v' file contents: %v", repoList, err))
	}

	err = yaml.Unmarshal(fileContent, &registry)
	if err != nil {
		panic(fmt.Errorf("Error unmarshalling '%v' YAML file: %v", repoList, err))
	}
	return registry
}

var (
	registry              = initReg()
	dockerLibraryRegistry = registry.DockerLibraryRegistry
	e2eRegistry           = registry.E2eRegistry
	etcdRegistry          = registry.EtcdRegistry
	gcRegistry            = registry.GcRegistry
	// PrivateRegistry is an image repository that requires authentication
	PrivateRegistry = registry.PrivateRegistry
	sampleRegistry  = registry.SampleRegistry

	// Preconfigured image configs
	imageConfigs = initImageConfigs()
)

const (
	// CRDConversionWebhook image
	CRDConversionWebhook = iota
	// AdmissionWebhook image
	AdmissionWebhook
	// APIServer image
	APIServer
	// AppArmorLoader image
	AppArmorLoader
	// AuditProxy image
	AuditProxy
	// BusyBox image
	BusyBox
	// CheckMetadataConcealment image
	CheckMetadataConcealment
	// CudaVectorAdd image
	CudaVectorAdd
	// CudaVectorAdd2 image
	CudaVectorAdd2
	// Dnsutils image
	Dnsutils
	// EchoServer image
	EchoServer
	// EntrypointTester image
	EntrypointTester
	// Etcd image
	Etcd
	// Fakegitserver image
	Fakegitserver
	// GBFrontend image
	GBFrontend
	// GBRedisSlave image
	GBRedisSlave
	// Hostexec image
	Hostexec
	// IpcUtils image
	IpcUtils
	// Iperf image
	Iperf
	// JessieDnsutils image
	JessieDnsutils
	// Kitten image
	Kitten
	// Liveness image
	Liveness
	// LogsGenerator image
	LogsGenerator
	// Mounttest image
	Mounttest
	// MounttestUser image
	MounttestUser
	// Nautilus image
	Nautilus
	// Net image
	Net
	// Netexec image
	Netexec
	// Nettest image
	Nettest
	// Nginx image
	Nginx
	// NginxNew image
	NginxNew
	// Nonewprivs image
	Nonewprivs
	// NoSnatTest image
	NoSnatTest
	// NoSnatTestProxy image
	NoSnatTestProxy
	// Pause - when these values are updated, also update cmd/kubelet/app/options/container_runtime.go
	// Pause image
	Pause
	// Porter image
	Porter
	// PortForwardTester image
	PortForwardTester
	// Redis image
	Redis
	// ResourceConsumer image
	ResourceConsumer
	// ResourceController image
	ResourceController
	// ServeHostname image
	ServeHostname
	// TestWebserver image
	TestWebserver
	// VolumeNFSServer image
	VolumeNFSServer
	// VolumeISCSIServer image
	VolumeISCSIServer
	// VolumeGlusterServer image
	VolumeGlusterServer
	// VolumeRBDServer image
	VolumeRBDServer
)

func initImageConfigs() map[int]Config {
	configs := map[int]Config{}
	configs[CRDConversionWebhook] = Config{e2eRegistry, "crd-conversion-webhook", "1.13rev2"}
	configs[AdmissionWebhook] = Config{e2eRegistry, "webhook", "1.14v1"}
	configs[APIServer] = Config{e2eRegistry, "sample-apiserver", "1.10"}
	configs[AppArmorLoader] = Config{e2eRegistry, "apparmor-loader", "1.0"}
	configs[AuditProxy] = Config{e2eRegistry, "audit-proxy", "1.0"}
	configs[BusyBox] = Config{dockerLibraryRegistry, "busybox", "1.29"}
	configs[CheckMetadataConcealment] = Config{e2eRegistry, "metadata-concealment", "1.2"}
	configs[CudaVectorAdd] = Config{e2eRegistry, "cuda-vector-add", "1.0"}
	configs[CudaVectorAdd2] = Config{e2eRegistry, "cuda-vector-add", "2.0"}
	configs[Dnsutils] = Config{e2eRegistry, "dnsutils", "1.1"}
	configs[EchoServer] = Config{e2eRegistry, "echoserver", "2.2"}
	configs[EntrypointTester] = Config{e2eRegistry, "entrypoint-tester", "1.0"}
	configs[Etcd] = Config{etcdRegistry, "etcd", "v3.3.10"}
	configs[Fakegitserver] = Config{e2eRegistry, "fakegitserver", "1.0"}
	configs[GBFrontend] = Config{sampleRegistry, "gb-frontend", "v6"}
	configs[GBRedisSlave] = Config{sampleRegistry, "gb-redisslave", "v3"}
	configs[Hostexec] = Config{e2eRegistry, "hostexec", "1.1"}
	configs[IpcUtils] = Config{e2eRegistry, "ipc-utils", "1.0"}
	configs[Iperf] = Config{e2eRegistry, "iperf", "1.0"}
	configs[JessieDnsutils] = Config{e2eRegistry, "jessie-dnsutils", "1.0"}
	configs[Kitten] = Config{e2eRegistry, "kitten", "1.0"}
	configs[Liveness] = Config{e2eRegistry, "liveness", "1.1"}
	configs[LogsGenerator] = Config{e2eRegistry, "logs-generator", "1.0"}
	configs[Mounttest] = Config{e2eRegistry, "mounttest", "1.0"}
	configs[MounttestUser] = Config{e2eRegistry, "mounttest-user", "1.0"}
	configs[Nautilus] = Config{e2eRegistry, "nautilus", "1.0"}
	configs[Net] = Config{e2eRegistry, "net", "1.0"}
	configs[Netexec] = Config{e2eRegistry, "netexec", "1.1"}
	configs[Nettest] = Config{e2eRegistry, "nettest", "1.0"}
	configs[Nginx] = Config{dockerLibraryRegistry, "nginx", "1.14-alpine"}
	configs[NginxNew] = Config{dockerLibraryRegistry, "nginx", "1.15-alpine"}
	configs[Nonewprivs] = Config{e2eRegistry, "nonewprivs", "1.0"}
	configs[NoSnatTest] = Config{e2eRegistry, "no-snat-test", "1.0"}
	configs[NoSnatTestProxy] = Config{e2eRegistry, "no-snat-test-proxy", "1.0"}
	// Pause - when these values are updated, also update cmd/kubelet/app/options/container_runtime.go
	configs[Pause] = Config{gcRegistry, "pause", "3.1"}
	configs[Porter] = Config{e2eRegistry, "porter", "1.0"}
	configs[PortForwardTester] = Config{e2eRegistry, "port-forward-tester", "1.0"}
	configs[Redis] = Config{e2eRegistry, "redis", "1.0"}
	configs[ResourceConsumer] = Config{e2eRegistry, "resource-consumer", "1.5"}
	configs[ResourceController] = Config{e2eRegistry, "resource-consumer/controller", "1.0"}
	configs[ServeHostname] = Config{e2eRegistry, "serve-hostname", "1.1"}
	configs[TestWebserver] = Config{e2eRegistry, "test-webserver", "1.0"}
	configs[VolumeNFSServer] = Config{e2eRegistry, "volume/nfs", "1.0"}
	configs[VolumeISCSIServer] = Config{e2eRegistry, "volume/iscsi", "1.0"}
	configs[VolumeGlusterServer] = Config{e2eRegistry, "volume/gluster", "1.0"}
	configs[VolumeRBDServer] = Config{e2eRegistry, "volume/rbd", "1.0.1"}
	return configs
}

// GetImageConfigs returns the map of imageConfigs
func GetImageConfigs() map[int]Config {
	return imageConfigs
}

// GetConfig returns the Config object for an image
func GetConfig(image int) Config {
	return imageConfigs[image]
}

// GetE2EImage returns the fully qualified URI to an image (including version)
func GetE2EImage(image int) string {
	return fmt.Sprintf("%s/%s:%s", imageConfigs[image].registry, imageConfigs[image].name, imageConfigs[image].version)
}

// GetE2EImage returns the fully qualified URI to an image (including version)
func (i *Config) GetE2EImage() string {
	return fmt.Sprintf("%s/%s:%s", i.registry, i.name, i.version)
}

// GetPauseImageName returns the pause image name with proper version
func GetPauseImageName() string {
	return GetE2EImage(Pause)
}
