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
	gcRegistry            = registry.GcRegistry
	// PrivateRegistry is an image repository that requires authentication
	PrivateRegistry = registry.PrivateRegistry
	sampleRegistry  = registry.SampleRegistry
)

// Preconfigured image configs
var (
	CRDConversionWebhook     = Config{e2eRegistry, "crd-conversion-webhook", "1.13rev2"}
	AdmissionWebhook         = Config{e2eRegistry, "webhook", "1.13v1"}
	APIServer                = Config{e2eRegistry, "sample-apiserver", "1.10"}
	AppArmorLoader           = Config{e2eRegistry, "apparmor-loader", "1.0"}
	BusyBox                  = Config{dockerLibraryRegistry, "busybox", "1.29"}
	CheckMetadataConcealment = Config{e2eRegistry, "metadata-concealment", "1.1.1"}
	CudaVectorAdd            = Config{e2eRegistry, "cuda-vector-add", "1.0"}
	Dnsutils                 = Config{e2eRegistry, "dnsutils", "1.1"}
	EchoServer               = Config{e2eRegistry, "echoserver", "2.2"}
	EntrypointTester         = Config{e2eRegistry, "entrypoint-tester", "1.0"}
	Fakegitserver            = Config{e2eRegistry, "fakegitserver", "1.0"}
	GBFrontend               = Config{sampleRegistry, "gb-frontend", "v6"}
	GBRedisSlave             = Config{sampleRegistry, "gb-redisslave", "v3"}
	Hostexec                 = Config{e2eRegistry, "hostexec", "1.1"}
	IpcUtils                 = Config{e2eRegistry, "ipc-utils", "1.0"}
	Iperf                    = Config{e2eRegistry, "iperf", "1.0"}
	JessieDnsutils           = Config{e2eRegistry, "jessie-dnsutils", "1.0"}
	Kitten                   = Config{e2eRegistry, "kitten", "1.0"}
	Liveness                 = Config{e2eRegistry, "liveness", "1.0"}
	LogsGenerator            = Config{e2eRegistry, "logs-generator", "1.0"}
	Mounttest                = Config{e2eRegistry, "mounttest", "1.0"}
	MounttestUser            = Config{e2eRegistry, "mounttest-user", "1.0"}
	Nautilus                 = Config{e2eRegistry, "nautilus", "1.0"}
	Net                      = Config{e2eRegistry, "net", "1.0"}
	Netexec                  = Config{e2eRegistry, "netexec", "1.1"}
	Nettest                  = Config{e2eRegistry, "nettest", "1.0"}
	Nginx                    = Config{dockerLibraryRegistry, "nginx", "1.14-alpine"}
	NginxNew                 = Config{dockerLibraryRegistry, "nginx", "1.15-alpine"}
	Nonewprivs               = Config{e2eRegistry, "nonewprivs", "1.0"}
	NoSnatTest               = Config{e2eRegistry, "no-snat-test", "1.0"}
	NoSnatTestProxy          = Config{e2eRegistry, "no-snat-test-proxy", "1.0"}
	// Pause - when these values are updated, also update cmd/kubelet/app/options/container_runtime.go
	Pause               = Config{gcRegistry, "pause", "3.1"}
	Porter              = Config{e2eRegistry, "porter", "1.0"}
	PortForwardTester   = Config{e2eRegistry, "port-forward-tester", "1.0"}
	Redis               = Config{e2eRegistry, "redis", "1.0"}
	ResourceConsumer    = Config{e2eRegistry, "resource-consumer", "1.4"}
	ResourceController  = Config{e2eRegistry, "resource-consumer/controller", "1.0"}
	ServeHostname       = Config{e2eRegistry, "serve-hostname", "1.1"}
	TestWebserver       = Config{e2eRegistry, "test-webserver", "1.0"}
	VolumeNFSServer     = Config{e2eRegistry, "volume/nfs", "1.0"}
	VolumeISCSIServer   = Config{e2eRegistry, "volume/iscsi", "1.0"}
	VolumeGlusterServer = Config{e2eRegistry, "volume/gluster", "1.0"}
	VolumeRBDServer     = Config{e2eRegistry, "volume/rbd", "1.0.1"}

	// Use a newer version of the metadata concealment check on 1.14+ servers. See http://issue.k8s.io/71094
	CheckMetadataConcealment1_2 = Config{e2eRegistry, "metadata-concealment", "1.2"}
)

// GetE2EImage returns the fully qualified URI to an image (including version)
func GetE2EImage(image Config) string {
	return fmt.Sprintf("%s/%s:%s", image.registry, image.name, image.version)
}

// GetPauseImageName returns the pause image name with proper version
func GetPauseImageName() string {
	return GetE2EImage(Pause)
}
