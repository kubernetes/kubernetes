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
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"io/ioutil"
	"os"
	"regexp"
	"strings"

	yaml "gopkg.in/yaml.v2"
)

// RegistryList holds public and private image registries
type RegistryList struct {
	GcAuthenticatedRegistry string `yaml:"gcAuthenticatedRegistry"`
	DockerLibraryRegistry   string `yaml:"dockerLibraryRegistry"`
	DockerGluster           string `yaml:"dockerGluster"`
	E2eRegistry             string `yaml:"e2eRegistry"`
	E2eVolumeRegistry       string `yaml:"e2eVolumeRegistry"`
	PromoterE2eRegistry     string `yaml:"promoterE2eRegistry"`
	BuildImageRegistry      string `yaml:"buildImageRegistry"`
	InvalidRegistry         string `yaml:"invalidRegistry"`
	GcRegistry              string `yaml:"gcRegistry"`
	SigStorageRegistry      string `yaml:"sigStorageRegistry"`
	GcrReleaseRegistry      string `yaml:"gcrReleaseRegistry"`
	PrivateRegistry         string `yaml:"privateRegistry"`
	SampleRegistry          string `yaml:"sampleRegistry"`
	MicrosoftRegistry       string `yaml:"microsoftRegistry"`
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
		DockerGluster:           "docker.io/gluster",
		E2eRegistry:             "gcr.io/kubernetes-e2e-test-images",
		E2eVolumeRegistry:       "gcr.io/kubernetes-e2e-test-images/volume",
		PromoterE2eRegistry:     "k8s.gcr.io/e2e-test-images",
		BuildImageRegistry:      "k8s.gcr.io/build-image",
		InvalidRegistry:         "invalid.com/invalid",
		GcRegistry:              "k8s.gcr.io",
		SigStorageRegistry:      "k8s.gcr.io/sig-storage",
		GcrReleaseRegistry:      "gcr.io/gke-release",
		PrivateRegistry:         "gcr.io/k8s-authenticated-test",
		SampleRegistry:          "gcr.io/google-samples",
		MicrosoftRegistry:       "mcr.microsoft.com",
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
	registry = initReg()

	// PrivateRegistry is an image repository that requires authentication
	PrivateRegistry = registry.PrivateRegistry

	// Preconfigured image configs
	dockerLibraryRegistry   = registry.DockerLibraryRegistry
	dockerGluster           = registry.DockerGluster
	e2eRegistry             = registry.E2eRegistry
	e2eVolumeRegistry       = registry.E2eVolumeRegistry
	promoterE2eRegistry     = registry.PromoterE2eRegistry
	buildImageRegistry      = registry.BuildImageRegistry
	gcAuthenticatedRegistry = registry.GcAuthenticatedRegistry
	gcRegistry              = registry.GcRegistry
	sigStorageRegistry      = registry.SigStorageRegistry
	gcrReleaseRegistry      = registry.GcrReleaseRegistry
	invalidRegistry         = registry.InvalidRegistry
	sampleRegistry          = registry.SampleRegistry
	microsoftRegistry       = registry.MicrosoftRegistry

	imageConfigs, originalImageConfigs = initImageConfigs()
)

const (
	// None is to be used for unset/default images
	None = iota
	// Agnhost image
	Agnhost
	// AgnhostPrivate image
	AgnhostPrivate
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
	// DebianIptables Image
	DebianIptables
	// EchoServer image
	EchoServer
	// Etcd image
	Etcd
	// GlusterDynamicProvisioner image
	GlusterDynamicProvisioner
	// Httpd image
	Httpd
	// HttpdNew image
	HttpdNew
	// InvalidRegistryImage image
	InvalidRegistryImage
	// IpcUtils image
	IpcUtils
	// JessieDnsutils image
	JessieDnsutils
	// Kitten image
	Kitten
	// Nautilus image
	Nautilus
	// NFSProvisioner image
	NFSProvisioner
	// Nginx image
	Nginx
	// NginxNew image
	NginxNew
	// NodePerfNpbEp image
	NodePerfNpbEp
	// NodePerfNpbIs image
	NodePerfNpbIs
	// NodePerfTfWideDeep image
	NodePerfTfWideDeep
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
	// RegressionIssue74839 image
	RegressionIssue74839
	// ResourceConsumer image
	ResourceConsumer
	// SdDummyExporter image
	SdDummyExporter
	// VolumeNFSServer image
	VolumeNFSServer
	// VolumeISCSIServer image
	VolumeISCSIServer
	// VolumeGlusterServer image
	VolumeGlusterServer
	// VolumeRBDServer image
	VolumeRBDServer
	// WindowsServer image
	WindowsServer
)

func initImageConfigs() (map[int]Config, map[int]Config) {
	configs := map[int]Config{}
	configs[Agnhost] = Config{promoterE2eRegistry, "agnhost", "2.26"}
	configs[AgnhostPrivate] = Config{PrivateRegistry, "agnhost", "2.6"}
	configs[AuthenticatedAlpine] = Config{gcAuthenticatedRegistry, "alpine", "3.7"}
	configs[AuthenticatedWindowsNanoServer] = Config{gcAuthenticatedRegistry, "windows-nanoserver", "v1"}
	configs[APIServer] = Config{promoterE2eRegistry, "sample-apiserver", "1.17.4"}
	configs[AppArmorLoader] = Config{promoterE2eRegistry, "apparmor-loader", "1.3"}
	configs[BusyBox] = Config{dockerLibraryRegistry, "busybox", "1.29"}
	configs[CheckMetadataConcealment] = Config{promoterE2eRegistry, "metadata-concealment", "1.6"}
	configs[CudaVectorAdd] = Config{e2eRegistry, "cuda-vector-add", "1.0"}
	configs[CudaVectorAdd2] = Config{promoterE2eRegistry, "cuda-vector-add", "2.2"}
	configs[DebianIptables] = Config{buildImageRegistry, "debian-iptables", "buster-v1.5.0"}
	configs[EchoServer] = Config{promoterE2eRegistry, "echoserver", "2.3"}
	configs[Etcd] = Config{gcRegistry, "etcd", "3.4.13-0"}
	configs[GlusterDynamicProvisioner] = Config{dockerGluster, "glusterdynamic-provisioner", "v1.0"}
	configs[Httpd] = Config{dockerLibraryRegistry, "httpd", "2.4.38-alpine"}
	configs[HttpdNew] = Config{dockerLibraryRegistry, "httpd", "2.4.39-alpine"}
	configs[InvalidRegistryImage] = Config{invalidRegistry, "alpine", "3.1"}
	configs[IpcUtils] = Config{promoterE2eRegistry, "ipc-utils", "1.2"}
	configs[JessieDnsutils] = Config{promoterE2eRegistry, "jessie-dnsutils", "1.4"}
	configs[Kitten] = Config{promoterE2eRegistry, "kitten", "1.4"}
	configs[Nautilus] = Config{promoterE2eRegistry, "nautilus", "1.4"}
	configs[NFSProvisioner] = Config{sigStorageRegistry, "nfs-provisioner", "v2.2.2"}
	configs[Nginx] = Config{dockerLibraryRegistry, "nginx", "1.14-alpine"}
	configs[NginxNew] = Config{dockerLibraryRegistry, "nginx", "1.15-alpine"}
	configs[NodePerfNpbEp] = Config{promoterE2eRegistry, "node-perf/npb-ep", "1.1"}
	configs[NodePerfNpbIs] = Config{promoterE2eRegistry, "node-perf/npb-is", "1.1"}
	configs[NodePerfTfWideDeep] = Config{promoterE2eRegistry, "node-perf/tf-wide-deep", "1.1"}
	configs[Nonewprivs] = Config{promoterE2eRegistry, "nonewprivs", "1.3"}
	configs[NonRoot] = Config{promoterE2eRegistry, "nonroot", "1.1"}
	// Pause - when these values are updated, also update cmd/kubelet/app/options/container_runtime.go
	configs[Pause] = Config{gcRegistry, "pause", "3.4.1"}
	configs[Perl] = Config{dockerLibraryRegistry, "perl", "5.26"}
	configs[PrometheusDummyExporter] = Config{gcRegistry, "prometheus-dummy-exporter", "v0.1.0"}
	configs[PrometheusToSd] = Config{gcRegistry, "prometheus-to-sd", "v0.5.0"}
	configs[Redis] = Config{promoterE2eRegistry, "redis", "5.0.5-alpine"}
	configs[RegressionIssue74839] = Config{promoterE2eRegistry, "regression-issue-74839", "1.2"}
	configs[ResourceConsumer] = Config{e2eRegistry, "resource-consumer", "1.5"}
	configs[SdDummyExporter] = Config{gcRegistry, "sd-dummy-exporter", "v0.2.0"}
	configs[VolumeNFSServer] = Config{promoterE2eRegistry, "volume/nfs", "1.2"}
	configs[VolumeISCSIServer] = Config{promoterE2eRegistry, "volume/iscsi", "2.2"}
	configs[VolumeGlusterServer] = Config{promoterE2eRegistry, "volume/gluster", "1.2"}
	configs[VolumeRBDServer] = Config{promoterE2eRegistry, "volume/rbd", "1.0.3"}
	configs[WindowsServer] = Config{microsoftRegistry, "windows", "1809"}

	// if requested, map all the SHAs into a known format based on the input
	originalImageConfigs := configs
	if repo := os.Getenv("KUBE_TEST_REPO"); len(repo) > 0 {
		configs = GetMappedImageConfigs(originalImageConfigs, repo)
	}

	return configs, originalImageConfigs
}

// GetMappedImageConfigs returns the images if they were mapped to the provided
// image repository.
func GetMappedImageConfigs(originalImageConfigs map[int]Config, repo string) map[int]Config {
	configs := make(map[int]Config)
	for i, config := range originalImageConfigs {
		switch i {
		case InvalidRegistryImage, AuthenticatedAlpine,
			AuthenticatedWindowsNanoServer, AgnhostPrivate:
			// These images are special and can't be run out of the cloud - some because they
			// are authenticated, and others because they are not real images. Tests that depend
			// on these images can't be run without access to the public internet.
			configs[i] = config
			continue
		}

		// Build a new tag with a the index, a hash of the image spec (to be unique) and
		// shorten and make the pull spec "safe" so it will fit in the tag
		configs[i] = getRepositoryMappedConfig(i, config, repo)
	}
	return configs
}

var (
	reCharSafe = regexp.MustCompile(`[^\w]`)
	reDashes   = regexp.MustCompile(`-+`)
)

// getRepositoryMappedConfig maps an existing image to the provided repo, generating a
// tag that is unique with the input config. The tag will contain the index, a hash of
// the image spec (to be unique) and shorten and make the pull spec "safe" so it will
// fit in the tag to allow a human to recognize the value. If index is -1, then no
// index will be added to the tag.
func getRepositoryMappedConfig(index int, config Config, repo string) Config {
	parts := strings.SplitN(repo, "/", 2)
	registry, name := parts[0], parts[1]

	pullSpec := config.GetE2EImage()

	h := sha256.New()
	h.Write([]byte(pullSpec))
	hash := base64.RawURLEncoding.EncodeToString(h.Sum(nil)[:16])

	shortName := reCharSafe.ReplaceAllLiteralString(pullSpec, "-")
	shortName = reDashes.ReplaceAllLiteralString(shortName, "-")
	maxLength := 127 - 16 - 6 - 10
	if len(shortName) > maxLength {
		shortName = shortName[len(shortName)-maxLength:]
	}
	var version string
	if index == -1 {
		version = fmt.Sprintf("e2e-%s-%s", shortName, hash)
	} else {
		version = fmt.Sprintf("e2e-%d-%s-%s", index, shortName, hash)
	}

	return Config{
		registry: registry,
		name:     name,
		version:  version,
	}
}

// GetOriginalImageConfigs returns the configuration before any mapping rules.
func GetOriginalImageConfigs() map[int]Config {
	return originalImageConfigs
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

	if repo := os.Getenv("KUBE_TEST_REPO"); len(repo) > 0 {
		index := -1
		for i, v := range originalImageConfigs {
			if v.GetE2EImage() == imageURL {
				index = i
				break
			}
		}
		last := strings.SplitN(parts[countParts-1], ":", 2)
		config := getRepositoryMappedConfig(index, Config{
			registry: parts[0],
			name:     strings.Join([]string{strings.Join(parts[1:countParts-1], "/"), last[0]}, "/"),
			version:  last[1],
		}, repo)
		return config.GetE2EImage(), nil
	}

	switch registryAndUser {
	case "gcr.io/kubernetes-e2e-test-images":
		registryAndUser = e2eRegistry
	case "gcr.io/kubernetes-e2e-test-images/volume":
		registryAndUser = e2eVolumeRegistry
	case "k8s.gcr.io":
		registryAndUser = gcRegistry
	case "k8s.gcr.io/sig-storage":
		registryAndUser = sigStorageRegistry
	case "gcr.io/k8s-authenticated-test":
		registryAndUser = PrivateRegistry
	case "gcr.io/google-samples":
		registryAndUser = sampleRegistry
	case "gcr.io/gke-release":
		registryAndUser = gcrReleaseRegistry
	case "docker.io/library":
		registryAndUser = dockerLibraryRegistry
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
