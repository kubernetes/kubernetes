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
	"runtime"
)

const (
	dockerHubRegistry = "docker.io"
	e2eRegistry       = "gcr.io/kubernetes-e2e-test-images"
	gcRegistry        = "k8s.gcr.io"
	PrivateRegistry   = "gcr.io/k8s-authenticated-test"
	sampleRegistry    = "gcr.io/google-samples"
)

type ImageConfig struct {
	registry string
	name     string
	version  string
	hasArch  bool
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

var (
	AdmissionWebhook         = ImageConfig{e2eRegistry, "webhook", "1.12v1", false}
	APIServer                = ImageConfig{e2eRegistry, "sample-apiserver", "1.0", false}
	AppArmorLoader           = ImageConfig{gcRegistry, "apparmor-loader", "0.1", false}
	BusyBox                  = ImageConfig{gcRegistry, "busybox", "1.24", false}
	CheckMetadataConcealment = ImageConfig{gcRegistry, "check-metadata-concealment", "v0.0.3", false}
	CudaVectorAdd            = ImageConfig{e2eRegistry, "cuda-vector-add", "1.0", false}
	Dnsutils                 = ImageConfig{e2eRegistry, "dnsutils", "1.1", false}
	EchoServer               = ImageConfig{gcRegistry, "echoserver", "1.10", false}
	EntrypointTester         = ImageConfig{e2eRegistry, "entrypoint-tester", "1.0", false}
	Fakegitserver            = ImageConfig{e2eRegistry, "fakegitserver", "1.0", false}
	GBFrontend               = ImageConfig{sampleRegistry, "gb-frontend", "v6", false}
	GBRedisSlave             = ImageConfig{sampleRegistry, "gb-redisslave", "v3", false}
	Hostexec                 = ImageConfig{e2eRegistry, "hostexec", "1.1", false}
	IpcUtils                 = ImageConfig{e2eRegistry, "ipc-utils", "1.0", false}
	Iperf                    = ImageConfig{e2eRegistry, "iperf", "1.0", false}
	JessieDnsutils           = ImageConfig{e2eRegistry, "jessie-dnsutils", "1.0", false}
	Kitten                   = ImageConfig{e2eRegistry, "kitten", "1.0", false}
	Liveness                 = ImageConfig{e2eRegistry, "liveness", "1.0", false}
	LogsGenerator            = ImageConfig{e2eRegistry, "logs-generator", "1.0", false}
	Mounttest                = ImageConfig{e2eRegistry, "mounttest", "1.0", false}
	MounttestUser            = ImageConfig{e2eRegistry, "mounttest-user", "1.0", false}
	Nautilus                 = ImageConfig{e2eRegistry, "nautilus", "1.0", false}
	Net                      = ImageConfig{e2eRegistry, "net", "1.0", false}
	Netexec                  = ImageConfig{e2eRegistry, "netexec", "1.0", false}
	Nettest                  = ImageConfig{e2eRegistry, "nettest", "1.0", false}
	Nginx                    = ImageConfig{dockerHubRegistry, "nginx", "1.14-alpine", false}
	NginxNew                 = ImageConfig{dockerHubRegistry, "nginx", "1.15-alpine", false}
	Nonewprivs               = ImageConfig{e2eRegistry, "nonewprivs", "1.0", false}
	NoSnatTest               = ImageConfig{e2eRegistry, "no-snat-test", "1.0", false}
	NoSnatTestProxy          = ImageConfig{e2eRegistry, "no-snat-test-proxy", "1.0", false}
	// When these values are updated, also update cmd/kubelet/app/options/container_runtime.go
	Pause               = ImageConfig{gcRegistry, "pause", "3.1", false}
	Porter              = ImageConfig{e2eRegistry, "porter", "1.0", false}
	PortForwardTester   = ImageConfig{e2eRegistry, "port-forward-tester", "1.0", false}
	Redis               = ImageConfig{e2eRegistry, "redis", "1.0", false}
	ResourceConsumer    = ImageConfig{e2eRegistry, "resource-consumer", "1.3", false}
	ResourceController  = ImageConfig{e2eRegistry, "resource-consumer/controller", "1.0", false}
	ServeHostname       = ImageConfig{e2eRegistry, "serve-hostname", "1.1", false}
	TestWebserver       = ImageConfig{e2eRegistry, "test-webserver", "1.0", false}
	VolumeNFSServer     = ImageConfig{e2eRegistry, "volume-nfs", "0.8", false}
	VolumeISCSIServer   = ImageConfig{e2eRegistry, "volume-iscsi", "0.2", false}
	VolumeGlusterServer = ImageConfig{e2eRegistry, "volume-gluster", "0.5", false}
	VolumeRBDServer     = ImageConfig{e2eRegistry, "volume-rbd", "0.2", false}
)

func GetE2EImage(image ImageConfig) string {
	return GetE2EImageWithArch(image, runtime.GOARCH)
}

func GetE2EImageWithArch(image ImageConfig, arch string) string {
	if image.hasArch {
		return fmt.Sprintf("%s/%s-%s:%s", image.registry, image.name, arch, image.version)
	} else {
		return fmt.Sprintf("%s/%s:%s", image.registry, image.name, image.version)
	}
}

// GetPauseImageName returns the pause image name with proper version
func GetPauseImageName() string {
	return GetE2EImage(Pause)
}
