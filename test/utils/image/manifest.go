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
	"strings"

	yaml "gopkg.in/yaml.v2"
)

// RegistryList holds public and private image registries
type RegistryList struct {
	GcAuthenticatedRegistry string `yaml:"gcAuthenticatedRegistry"`
	DockerLibraryRegistry   string `yaml:"dockerLibraryRegistry"`
	E2eRegistry             string `yaml:"e2eRegistry"`
	InvalidRegistry         string `yaml:"invalidRegistry"`
	GcRegistry              string `yaml:"gcRegistry"`
	GcrReleaseRegistry      string `yaml:"gcrReleaseRegistry"`
	GoogleContainerRegistry string `yaml:"googleContainerRegistry"`
	PrivateRegistry         string `yaml:"privateRegistry"`
	SampleRegistry          string `yaml:"sampleRegistry"`
	QuayK8sCSI              string `yaml:"quayK8sCSI"`
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
		GcAuthenticatedRegistry: "gcr.io/authenticated-image-pulling",
		DockerLibraryRegistry:   "docker.io/library",
		E2eRegistry:             "gcr.io/kubernetes-e2e-test-images",
		InvalidRegistry:         "invalid.com/invalid",
		GcRegistry:              "k8s.gcr.io",
		GcrReleaseRegistry:      "gcr.io/gke-release",
		GoogleContainerRegistry: "gcr.io/google-containers",
		PrivateRegistry:         "gcr.io/k8s-authenticated-test",
		SampleRegistry:          "gcr.io/google-samples",
		QuayK8sCSI:              "quay.io/k8scsi",
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
	registry                = initReg()
	dockerLibraryRegistry   = registry.DockerLibraryRegistry
	e2eRegistry             = registry.E2eRegistry
	e2eGcRegistry           = "gcr.io/kubernetes-e2e-test-images"
	gcAuthenticatedRegistry = registry.GcAuthenticatedRegistry
	gcRegistry              = registry.GcRegistry
	gcrReleaseRegistry      = registry.GcrReleaseRegistry
	googleContainerRegistry = registry.GoogleContainerRegistry
	invalidRegistry         = registry.InvalidRegistry
	quayK8sCSI              = registry.QuayK8sCSI
	// PrivateRegistry is an image repository that requires authentication
	PrivateRegistry = registry.PrivateRegistry
	sampleRegistry  = registry.SampleRegistry

	// Preconfigured image configs
	imageConfigs = initImageConfigs()
)

const (
	// Agnhost image
	Agnhost = iota
	// Alpine image
	Alpine
	// APIServer image
	APIServer
	// AppArmorLoader image
	AppArmorLoader
	// AuthenticatedAlpine image
	AuthenticatedAlpine
	// AuthenticatedWindowsNanoServer image
	AuthenticatedWindowsNanoServer
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
	// DebianBase image
	DebianBase
	// EchoServer image
	EchoServer
	// Etcd image
	Etcd
	// GBFrontend image
	GBFrontend
	// Httpd image
	Httpd
	// HttpdNew image
	HttpdNew
	// Invalid image
	Invalid
	// InvalidRegistryImage image
	InvalidRegistryImage
	// IpcUtils image
	IpcUtils
	// JessieDnsutils image
	JessieDnsutils
	// Kitten image
	Kitten
	// Mounttest image
	Mounttest
	// MounttestUser image
	MounttestUser
	// Nautilus image
	Nautilus
	// Nginx image
	Nginx
	// NginxNew image
	NginxNew
	// Nonewprivs image
	Nonewprivs
	// NonRoot runs with a default user of 1234
	NonRoot
	// Pause - when these values are updated, also update cmd/kubelet/app/options/container_runtime.go
	// Pause image
	Pause
	// Perl image
	Perl
	// PrometheusDummyExporter image
	PrometheusDummyExporter
	// PrometheusToSd image
	PrometheusToSd
	// Redis image
	Redis
	// ResourceConsumer image
	ResourceConsumer
	// ResourceController image
	ResourceController
	// SdDummyExporter image
	SdDummyExporter
	// StartupScript image
	StartupScript
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
	// WindowsNanoServer image
	WindowsNanoServer
)

func initImageConfigs() map[int]Config {
	configs := map[int]Config{}
	configs[Agnhost] = Config{e2eRegistry, "agnhost", "2.4"}
	configs[Alpine] = Config{dockerLibraryRegistry, "alpine", "3.7"}
	configs[AuthenticatedAlpine] = Config{gcAuthenticatedRegistry, "alpine", "3.7"}
	configs[AuthenticatedWindowsNanoServer] = Config{gcAuthenticatedRegistry, "windows-nanoserver", "v1"}
	configs[APIServer] = Config{e2eRegistry, "sample-apiserver", "1.10"}
	configs[AppArmorLoader] = Config{e2eRegistry, "apparmor-loader", "1.0"}
	configs[BusyBox] = Config{dockerLibraryRegistry, "busybox", "1.29"}
	configs[CheckMetadataConcealment] = Config{e2eRegistry, "metadata-concealment", "1.2"}
	configs[CudaVectorAdd] = Config{e2eRegistry, "cuda-vector-add", "1.0"}
	configs[CudaVectorAdd2] = Config{e2eRegistry, "cuda-vector-add", "2.0"}
	configs[Dnsutils] = Config{e2eRegistry, "dnsutils", "1.1"}
	configs[DebianBase] = Config{googleContainerRegistry, "debian-base", "0.4.1"}
	configs[EchoServer] = Config{e2eRegistry, "echoserver", "2.2"}
	configs[Etcd] = Config{gcRegistry, "etcd", "3.3.10"}
	configs[GBFrontend] = Config{sampleRegistry, "gb-frontend", "v6"}
	configs[Httpd] = Config{dockerLibraryRegistry, "httpd", "2.4.38-alpine"}
	configs[HttpdNew] = Config{dockerLibraryRegistry, "httpd", "2.4.39-alpine"}
	configs[Invalid] = Config{gcRegistry, "invalid-image", "invalid-tag"}
	configs[InvalidRegistryImage] = Config{invalidRegistry, "alpine", "3.1"}
	configs[IpcUtils] = Config{e2eRegistry, "ipc-utils", "1.0"}
	configs[JessieDnsutils] = Config{e2eRegistry, "jessie-dnsutils", "1.0"}
	configs[Kitten] = Config{e2eRegistry, "kitten", "1.0"}
	configs[Mounttest] = Config{e2eRegistry, "mounttest", "1.0"}
	configs[MounttestUser] = Config{e2eRegistry, "mounttest-user", "1.0"}
	configs[Nautilus] = Config{e2eRegistry, "nautilus", "1.0"}
	configs[Nginx] = Config{dockerLibraryRegistry, "nginx", "1.14-alpine"}
	configs[NginxNew] = Config{dockerLibraryRegistry, "nginx", "1.15-alpine"}
	configs[Nonewprivs] = Config{e2eRegistry, "nonewprivs", "1.0"}
	configs[NonRoot] = Config{e2eRegistry, "nonroot", "1.0"}
	// Pause - when these values are updated, also update cmd/kubelet/app/options/container_runtime.go
	configs[Pause] = Config{gcRegistry, "pause", "3.1"}
	configs[Perl] = Config{dockerLibraryRegistry, "perl", "5.26"}
	configs[PrometheusDummyExporter] = Config{gcRegistry, "prometheus-dummy-exporter", "v0.1.0"}
	configs[PrometheusToSd] = Config{gcRegistry, "prometheus-to-sd", "v0.5.0"}
	configs[Redis] = Config{dockerLibraryRegistry, "redis", "5.0.5-alpine"}
	configs[ResourceConsumer] = Config{e2eRegistry, "resource-consumer", "1.5"}
	configs[ResourceController] = Config{e2eRegistry, "resource-consumer-controller", "1.0"}
	configs[SdDummyExporter] = Config{gcRegistry, "sd-dummy-exporter", "v0.2.0"}
	configs[StartupScript] = Config{googleContainerRegistry, "startup-script", "v1"}
	configs[TestWebserver] = Config{e2eRegistry, "test-webserver", "1.0"}
	configs[VolumeNFSServer] = Config{e2eRegistry, "volume/nfs", "1.0"}
	configs[VolumeISCSIServer] = Config{e2eRegistry, "volume/iscsi", "2.0"}
	configs[VolumeGlusterServer] = Config{e2eRegistry, "volume/gluster", "1.0"}
	configs[VolumeRBDServer] = Config{e2eRegistry, "volume/rbd", "1.0.1"}
	configs[WindowsNanoServer] = Config{e2eGcRegistry, "windows-nanoserver", "v1"}
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

// ReplaceRegistryInImageURL replaces the registry in the image URL with a custom one
func ReplaceRegistryInImageURL(imageURL string) (string, error) {
	parts := strings.Split(imageURL, "/")
	countParts := len(parts)
	registryAndUser := strings.Join(parts[:countParts-1], "/")

	switch registryAndUser {
	case "gcr.io/kubernetes-e2e-test-images":
		registryAndUser = e2eRegistry
	case "k8s.gcr.io":
		registryAndUser = gcRegistry
	case "gcr.io/k8s-authenticated-test":
		registryAndUser = PrivateRegistry
	case "gcr.io/google-samples":
		registryAndUser = sampleRegistry
	case "gcr.io/gke-release":
		registryAndUser = gcrReleaseRegistry
	case "docker.io/library":
		registryAndUser = dockerLibraryRegistry
	case "quay.io/k8scsi":
		registryAndUser = quayK8sCSI
	default:
		if countParts == 1 {
			// We assume we found an image from docker hub library
			// e.g. openjdk -> docker.io/library/openjdk
			registryAndUser = dockerLibraryRegistry
			break
		}

		return "", fmt.Errorf("Registry: %s is missing in test/utils/image/manifest.go, please add the registry, otherwise the test will fail on air-gapped clusters", registryAndUser)
	}

	return fmt.Sprintf("%s/%s", registryAndUser, parts[countParts-1]), nil
}
