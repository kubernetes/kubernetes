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
	e2eRegistry     = "gcr.io/kubernetes-e2e-test-images"
	gcRegistry      = "k8s.gcr.io"
	PrivateRegistry = "gcr.io/k8s-authenticated-test"
	sampleRegistry  = "gcr.io/google-samples"
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
	AdmissionWebhook         = ImageConfig{e2eRegistry, "k8s-sample-admission-webhook", "1.10v2", true}
	APIServer                = ImageConfig{e2eRegistry, "k8s-aggregator-sample-apiserver", "1.7v2", true}
	AppArmorLoader           = ImageConfig{gcRegistry, "apparmor-loader", "0.1", false}
	BusyBox                  = ImageConfig{gcRegistry, "busybox", "1.24", false}
	CheckMetadataConcealment = ImageConfig{gcRegistry, "check-metadata-concealment", "v0.0.3", false}
	ClusterTester            = ImageConfig{e2eRegistry, "clusterapi-tester", "1.0", true}
	CudaVectorAdd            = ImageConfig{e2eRegistry, "cuda-vector-add", "1.0", true}
	Dnsutils                 = ImageConfig{e2eRegistry, "dnsutils", "1.0", true}
	DNSMasq                  = ImageConfig{gcRegistry, "k8s-dns-dnsmasq", "1.14.5", true}
	EchoServer               = ImageConfig{gcRegistry, "echoserver", "1.6", false}
	EntrypointTester         = ImageConfig{e2eRegistry, "entrypoint-tester", "1.0", true}
	E2ENet                   = ImageConfig{gcRegistry, "e2e-net", "1.0", true}
	Fakegitserver            = ImageConfig{e2eRegistry, "fakegitserver", "1.0", true}
	GBFrontend               = ImageConfig{sampleRegistry, "gb-frontend", "v5", true}
	GBRedisSlave             = ImageConfig{sampleRegistry, "gb-redisslave", "v2", true}
	Goproxy                  = ImageConfig{e2eRegistry, "goproxy", "1.0", true}
	Hostexec                 = ImageConfig{e2eRegistry, "hostexec", "1.1", true}
	IpcUtils                 = ImageConfig{e2eRegistry, "ipc-utils", "1.0", true}
	Iperf                    = ImageConfig{e2eRegistry, "iperf", "1.0", true}
	JessieDnsutils           = ImageConfig{e2eRegistry, "jessie-dnsutils", "1.0", true}
	Kitten                   = ImageConfig{e2eRegistry, "kitten", "1.0", true}
	Liveness                 = ImageConfig{e2eRegistry, "liveness", "1.0", true}
	LogsGenerator            = ImageConfig{e2eRegistry, "logs-generator", "1.0", true}
	Mounttest                = ImageConfig{e2eRegistry, "mounttest", "1.0", true}
	MounttestUser            = ImageConfig{e2eRegistry, "mounttest-user", "1.0", true}
	Nautilus                 = ImageConfig{e2eRegistry, "nautilus", "1.0", true}
	Net                      = ImageConfig{e2eRegistry, "net", "1.0", true}
	Netexec                  = ImageConfig{e2eRegistry, "netexec", "1.0", true}
	Nettest                  = ImageConfig{e2eRegistry, "nettest", "1.0", true}
	NginxSlim                = ImageConfig{gcRegistry, "nginx-slim", "0.20", true}
	NginxSlimNew             = ImageConfig{gcRegistry, "nginx-slim", "0.21", true}
	Nonewprivs               = ImageConfig{e2eRegistry, "nonewprivs", "1.0", true}
	NoSnatTest               = ImageConfig{e2eRegistry, "no-snat-test", "1.0", true}
	NoSnatTestProxy          = ImageConfig{e2eRegistry, "no-snat-test-proxy", "1.0", true}
	NWayHTTP                 = ImageConfig{e2eRegistry, "n-way-http", "1.0", true}
	// When these values are updated, also update cmd/kubelet/app/options/container_runtime.go
	Pause               = ImageConfig{gcRegistry, "pause", "3.1", true}
	Porter              = ImageConfig{e2eRegistry, "porter", "1.0", true}
	PortForwardTester   = ImageConfig{e2eRegistry, "port-forward-tester", "1.0", true}
	Redis               = ImageConfig{e2eRegistry, "redis", "1.0", true}
	ResourceConsumer    = ImageConfig{e2eRegistry, "resource-consumer", "1.3", true}
	ResourceController  = ImageConfig{e2eRegistry, "resource-consumer/controller", "1.0", true}
	SDDummyExporter     = ImageConfig{gcRegistry, "sd-dummy-exporter", "v0.1.0", false}
	ServeHostname       = ImageConfig{e2eRegistry, "serve-hostname", "1.0", true}
	TestWebserver       = ImageConfig{e2eRegistry, "test-webserver", "1.0", true}
	VolumeNFSServer     = ImageConfig{gcRegistry, "volume-nfs", "0.8", false}
	VolumeISCSIServer   = ImageConfig{gcRegistry, "volume-icsci", "0.1", false}
	VolumeGlusterServer = ImageConfig{gcRegistry, "volume-gluster", "0.2", false}
	VolumeCephServer    = ImageConfig{gcRegistry, "volume-ceph", "0.1", false}
	VolumeRBDServer     = ImageConfig{gcRegistry, "volume-rbd", "0.1", false}
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

// GetPauseImageNameForHostArch fetches the pause image name for the same architecture the test is running on.
func GetPauseImageNameForHostArch() string {
	return GetE2EImage(Pause)
}
