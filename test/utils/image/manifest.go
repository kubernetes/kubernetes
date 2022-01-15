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
	"embed"
	"encoding/base64"
	"fmt"
	"io/ioutil"
	"os"
	"regexp"
	"strings"

	"gopkg.in/yaml.v2"
)

// RegistryList holds public and private image registries
type RegistryList struct {
	GcAuthenticatedRegistry  string `yaml:"gcAuthenticatedRegistry"`
	PromoterE2eRegistry      string `yaml:"promoterE2eRegistry"`
	BuildImageRegistry       string `yaml:"buildImageRegistry"`
	InvalidRegistry          string `yaml:"invalidRegistry"`
	GcEtcdRegistry           string `yaml:"gcEtcdRegistry"`
	GcRegistry               string `yaml:"gcRegistry"`
	SigStorageRegistry       string `yaml:"sigStorageRegistry"`
	PrivateRegistry          string `yaml:"privateRegistry"`
	MicrosoftRegistry        string `yaml:"microsoftRegistry"`
	DockerLibraryRegistry    string `yaml:"dockerLibraryRegistry"`
	CloudProviderGcpRegistry string `yaml:"cloudProviderGcpRegistry"`
}

// Config holds an images registry, name, and version
type Config struct {
	Registry string `yaml:"registry"`
	Name     string `yaml:"name"`
	Version  string `yaml:"version"`
}

type Configs struct {
	Configs []Config `yaml:"configs"`
}

//go:embed registry.yaml
var registryList embed.FS

//go:embed images.yaml
var images embed.FS

// SetRegistry sets an image registry in a Config struct
func (i *Config) SetRegistry(registry string) {
	i.Registry = registry
}

// SetName sets an image name in a Config struct
func (i *Config) SetName(name string) {
	i.Name = name
}

// SetVersion sets an image version in a Config struct
func (i *Config) SetVersion(version string) {
	i.Version = version
}

func initReg() RegistryList {
	data, _ := registryList.ReadFile("registry.yaml")
	var registry RegistryList
	yaml.Unmarshal(data, &registry)

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

	// Preconfigured image configs
	imageConfigs, originalImageConfigs = initImageConfigs(registry)
)

const (
	// None is to be used for unset/default images
	None = "None"
	// Agnhost image
	Agnhost = "Agnhost"
	// AgnhostPrivate image
	AgnhostPrivate = "AgnhostPrivate"
	// APIServer image
	APIServer = "APIServer"
	// AppArmorLoader image
	AppArmorLoader = "AppArmorLoader"
	// AuthenticatedAlpine image
	AuthenticatedAlpine = "AuthenticatedAlpine"
	// AuthenticatedWindowsNanoServer image
	AuthenticatedWindowsNanoServer = "AuthenticatedWindowsNanoServer"
	// BusyBox image
	BusyBox = "BusyBox"
	// CheckMetadataConcealment image
	CheckMetadataConcealment = "CheckMetadataConcealment"
	// CudaVectorAdd image
	CudaVectorAdd = "CudaVectorAdd"
	// CudaVectorAdd2 image
	CudaVectorAdd2 = "CudaVectorAdd2"
	// DebianIptables Image
	DebianIptables = "DebianIptables"
	// EchoServer image
	EchoServer = "EchoServer"
	// Etcd image
	Etcd = "Etcd"
	// GlusterDynamicProvisioner image
	GlusterDynamicProvisioner = "GlusterDynamicProvisioner"
	// Httpd image
	Httpd = "Httpd"
	// HttpdNew image
	HttpdNew = "HttpdNew"
	// InvalidRegistryImage image
	InvalidRegistryImage = "InvalidRegistryImage"
	// IpcUtils image
	IpcUtils = "IpcUtils"
	// JessieDnsutils image
	JessieDnsutils = "JessieDnsutils"
	// Kitten image
	Kitten = "Kitten"
	// Nautilus image
	Nautilus = "Nautilus"
	// NFSProvisioner image
	NFSProvisioner = "NFSProvisioner"
	// Nginx image
	Nginx = "Nginx"
	// NginxNew image
	NginxNew = "NginxNew"
	// NodePerfNpbEp image
	NodePerfNpbEp = "NodePerfNpbEp"
	// NodePerfNpbIs image
	NodePerfNpbIs = "NodePerfNpbIs"
	// NodePerfTfWideDeep image
	NodePerfTfWideDeep = "NodePerfTfWideDeep"
	// Nonewprivs image
	Nonewprivs = "Nonewprivs"
	// NonRoot runs with a default user of 1234
	NonRoot = "NonRoot"
	// Pause - when these values are updated, also update cmd/kubelet/app/options/container_runtime.go
	// Pause image
	Pause = "Pause"
	// Perl image
	Perl = "Perl"
	// PrometheusDummyExporter image
	PrometheusDummyExporter = "PrometheusDummyExporter"
	// PrometheusToSd image
	PrometheusToSd = "PrometheusToSd"
	// Redis image
	Redis = "Redis"
	// RegressionIssue74839 image
	RegressionIssue74839 = "RegressionIssue74839"
	// ResourceConsumer image
	ResourceConsumer = "ResourceConsumer"
	// SdDummyExporter image
	SdDummyExporter = "SdDummyExporter"
	// VolumeNFSServer image
	VolumeNFSServer = "VolumeNFSServer"
	// VolumeISCSIServer image
	VolumeISCSIServer = "VolumeISCSIServer"
	// VolumeGlusterServer image
	VolumeGlusterServer = "VolumeISCSIServer"
	// VolumeRBDServer image
	VolumeRBDServer = "VolumeISCSIServer"
	// WindowsServer image
	WindowsServer = "WindowsServer"
)

func initImageConfigs(list RegistryList) (Configs, Configs) {
	data, _ := images.ReadFile("images.yaml")
	var configs Configs
	yaml.Unmarshal(data, &configs)

	// if requested, map all the SHAs into a known format based on the input
	originalImageConfigs := configs
	if repo := os.Getenv("KUBE_TEST_REPO"); len(repo) > 0 {
		configs = GetMappedImageConfigs(list, originalImageConfigs, repo)
	}

	return configs, originalImageConfigs
}

// GetMappedImageConfigs returns the images if they were mapped to the provided
// image repository.
func GetMappedImageConfigs(list RegistryList, originalImageConfigs Configs, repo string) Configs {
	configs := make([]Config, 0)
	for _, config := range originalImageConfigs.Configs {
		switch config.Registry {
		case "InvalidRegistryImage", "AuthenticatedAlpine",
			"AuthenticatedWindowsNanoServer", "AgnhostPrivate":
			// These images are special and can't be run out of the cloud - some because they
			// are authenticated, and others because they are not real images. Tests that depend
			// on these images can't be run without access to the public internet.
			configs = append(configs, config)
			continue
		}

		// Build a new tag with a the index, a hash of the image spec (to be unique) and
		// shorten and make the pull spec "safe" so it will fit in the tag
		configs = append(configs, getRepositoryMappedConfig(0, config, repo))
	}
	return Configs{Configs: configs}
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
	hash := base64.RawURLEncoding.EncodeToString(h.Sum(nil))[:16]

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
		Registry: registry,
		Name:     name,
		Version:  version,
	}
}

// GetOriginalImageConfigs returns the configuration before any mapping rules.
func GetOriginalImageConfigs() []Config {
	return originalImageConfigs.Configs
}

// GetImageConfigs returns the map of imageConfigs
func GetImageConfigs() []Config {
	return imageConfigs.Configs
}

// GetConfig returns the Config object for an image
func GetConfig(image string) Config {
	for _, c := range imageConfigs.Configs {
		if c.Name == image {
			return c
		}
	}
	return Config{}
}

// GetE2EImage returns the fully qualified URI to an image (including version)
func GetE2EImage(image string) string {
	for _, c := range imageConfigs.Configs {
		if c.Name == image {
			return fmt.Sprintf("%s/%s:%s", c.Registry, c.Name, c.Version)
		}
	}
	return ""
}

// GetE2EImage returns the fully qualified URI to an image (including version)
func (i *Config) GetE2EImage() string {
	return fmt.Sprintf("%s/%s:%s", i.Registry, i.Name, i.Version)
}

// GetPauseImageName returns the pause image name with proper version
func GetPauseImageName() string {
	return GetE2EImage("Pause")
}

// ReplaceRegistryInImageURL replaces the registry in the image URL with a custom one based
// on the configured registries.
func ReplaceRegistryInImageURL(imageURL string) (string, error) {
	return replaceRegistryInImageURLWithList(imageURL, registry)
}

// replaceRegistryInImageURLWithList replaces the registry in the image URL with a custom one based
// on the given registry list.
func replaceRegistryInImageURLWithList(imageURL string, reg RegistryList) (string, error) {
	parts := strings.Split(imageURL, "/")
	countParts := len(parts)
	registryAndUser := strings.Join(parts[:countParts-1], "/")

	if repo := os.Getenv("KUBE_TEST_REPO"); len(repo) > 0 {
		index := -1
		for i, v := range originalImageConfigs.Configs {
			if v.GetE2EImage() == imageURL {
				index = i
				break
			}
		}
		last := strings.SplitN(parts[countParts-1], ":", 2)
		if len(last) == 1 {
			return "", fmt.Errorf("image %q is required to be in an image:tag format", imageURL)
		}
		config := getRepositoryMappedConfig(index, Config{
			Registry: parts[0],
			Name:     strings.Join([]string{strings.Join(parts[1:countParts-1], "/"), last[0]}, "/"),
			Version:  last[1],
		}, repo)
		return config.GetE2EImage(), nil
	}

	switch registryAndUser {
	case registry.GcRegistry:
		registryAndUser = reg.GcRegistry
	case registry.SigStorageRegistry:
		registryAndUser = reg.SigStorageRegistry
	case registry.PrivateRegistry:
		registryAndUser = reg.PrivateRegistry
	case registry.InvalidRegistry:
		registryAndUser = reg.InvalidRegistry
	case registry.MicrosoftRegistry:
		registryAndUser = reg.MicrosoftRegistry
	case registry.PromoterE2eRegistry:
		registryAndUser = reg.PromoterE2eRegistry
	case registry.BuildImageRegistry:
		registryAndUser = reg.BuildImageRegistry
	case registry.GcAuthenticatedRegistry:
		registryAndUser = reg.GcAuthenticatedRegistry
	case registry.DockerLibraryRegistry:
		registryAndUser = reg.DockerLibraryRegistry
	case registry.CloudProviderGcpRegistry:
		registryAndUser = reg.CloudProviderGcpRegistry
	default:
		if countParts == 1 {
			// We assume we found an image from docker hub library
			// e.g. openjdk -> docker.io/library/openjdk
			registryAndUser = reg.DockerLibraryRegistry
			break
		}

		return "", fmt.Errorf("Registry: %s is missing in test/utils/image/manifest.go, please add the registry, otherwise the test will fail on air-gapped clusters", registryAndUser)
	}

	return fmt.Sprintf("%s/%s", registryAndUser, parts[countParts-1]), nil
}
