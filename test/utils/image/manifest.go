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

type RegistryList struct {
	DockerLibraryRegistry string `yaml:"dockerLibraryRegistry"`
	E2eRegistry           string `yaml:"e2eRegistry"`
	GcRegistry            string `yaml:"gcRegistry"`
	PrivateRegistry       string `yaml:"privateRegistry"`
	SampleRegistry        string `yaml:"sampleRegistry"`
}
type ImageConfig struct {
	registry string
	name     string
	version  string
}

func (i *ImageConfig) SetRegistry(registry string) {
	i.registry = registry
}

func (i *ImageConfig) SetName(name string) {
	i.name = name
}

func (i *ImageConfig) SetVersion(version string) {
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
	PrivateRegistry       = registry.PrivateRegistry
	sampleRegistry        = registry.SampleRegistry

	AdmissionWebhook         = ImageConfig{e2eRegistry, "webhook", "1.13v1"}
	APIServer                = ImageConfig{e2eRegistry, "sample-apiserver", "1.0"}
	AppArmorLoader           = ImageConfig{e2eRegistry, "apparmor-loader", "1.0"}
	BusyBox                  = ImageConfig{dockerLibraryRegistry, "busybox", "1.29"}
	CheckMetadataConcealment = ImageConfig{e2eRegistry, "metadata-concealment", "1.0"}
	CudaVectorAdd            = ImageConfig{e2eRegistry, "cuda-vector-add", "1.0"}
	Dnsutils                 = ImageConfig{e2eRegistry, "dnsutils", "1.1"}
	EchoServer               = ImageConfig{e2eRegistry, "echoserver", "2.2"}
	EntrypointTester         = ImageConfig{e2eRegistry, "entrypoint-tester", "1.0"}
	Fakegitserver            = ImageConfig{e2eRegistry, "fakegitserver", "1.0"}
	GBFrontend               = ImageConfig{sampleRegistry, "gb-frontend", "v6"}
	GBRedisSlave             = ImageConfig{sampleRegistry, "gb-redisslave", "v3"}
	Hostexec                 = ImageConfig{e2eRegistry, "hostexec", "1.1"}
	IpcUtils                 = ImageConfig{e2eRegistry, "ipc-utils", "1.0"}
	Iperf                    = ImageConfig{e2eRegistry, "iperf", "1.0"}
	JessieDnsutils           = ImageConfig{e2eRegistry, "jessie-dnsutils", "1.0"}
	Kitten                   = ImageConfig{e2eRegistry, "kitten", "1.0"}
	Liveness                 = ImageConfig{e2eRegistry, "liveness", "1.0"}
	LogsGenerator            = ImageConfig{e2eRegistry, "logs-generator", "1.0"}
	Mounttest                = ImageConfig{e2eRegistry, "mounttest", "1.0"}
	MounttestUser            = ImageConfig{e2eRegistry, "mounttest-user", "1.0"}
	Nautilus                 = ImageConfig{e2eRegistry, "nautilus", "1.0"}
	Net                      = ImageConfig{e2eRegistry, "net", "1.0"}
	Netexec                  = ImageConfig{e2eRegistry, "netexec", "1.0"}
	Nettest                  = ImageConfig{e2eRegistry, "nettest", "1.0"}
	Nginx                    = ImageConfig{dockerLibraryRegistry, "nginx", "1.14-alpine"}
	NginxNew                 = ImageConfig{dockerLibraryRegistry, "nginx", "1.15-alpine"}
	Nonewprivs               = ImageConfig{e2eRegistry, "nonewprivs", "1.0"}
	NoSnatTest               = ImageConfig{e2eRegistry, "no-snat-test", "1.0"}
	NoSnatTestProxy          = ImageConfig{e2eRegistry, "no-snat-test-proxy", "1.0"}
	// When these values are updated, also update cmd/kubelet/app/options/container_runtime.go
	Pause               = ImageConfig{gcRegistry, "pause", "3.1"}
	Porter              = ImageConfig{e2eRegistry, "porter", "1.0"}
	PortForwardTester   = ImageConfig{e2eRegistry, "port-forward-tester", "1.0"}
	Redis               = ImageConfig{e2eRegistry, "redis", "1.0"}
	ResourceConsumer    = ImageConfig{e2eRegistry, "resource-consumer", "1.3"}
	ResourceController  = ImageConfig{e2eRegistry, "resource-consumer/controller", "1.0"}
	ServeHostname       = ImageConfig{e2eRegistry, "serve-hostname", "1.1"}
	TestWebserver       = ImageConfig{e2eRegistry, "test-webserver", "1.0"}
	VolumeNFSServer     = ImageConfig{e2eRegistry, "volume/nfs", "1.0"}
	VolumeISCSIServer   = ImageConfig{e2eRegistry, "volume/iscsi", "1.0"}
	VolumeGlusterServer = ImageConfig{e2eRegistry, "volume/gluster", "1.0"}
	VolumeRBDServer     = ImageConfig{e2eRegistry, "volume/rbd", "1.0.1"}
)

func GetE2EImage(image ImageConfig) string {
	return fmt.Sprintf("%s/%s:%s", image.registry, image.name, image.version)
}

// GetPauseImageName returns the pause image name with proper version
func GetPauseImageName() string {
	return GetE2EImage(Pause)
}
