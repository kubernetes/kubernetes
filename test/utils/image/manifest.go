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
	gcRegistry      = "gcr.io/google-containers"
	PrivateRegistry = "gcr.io/k8s-authenticated-test"
	sampleRegistry  = "gcr.io/google-samples"
)

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

var (
	ClusterTester      = ImageConfig{e2eRegistry, "clusterapi-tester", "1.0"}
	CudaVectorAdd      = ImageConfig{e2eRegistry, "cuda-vector-add", "1.0"}
	Dnsutils           = ImageConfig{e2eRegistry, "dnsutils", "1.0"}
	EntrypointTester   = ImageConfig{e2eRegistry, "entrypoint-tester", "1.0"}
	Fakegitserver      = ImageConfig{e2eRegistry, "fakegitserver", "1.0"}
	GBFrontend         = ImageConfig{sampleRegistry, "gb-frontend", "v5"}
	GBRedisSlave       = ImageConfig{sampleRegistry, "gb-redisslave", "v2"}
	Goproxy            = ImageConfig{e2eRegistry, "goproxy", "1.0"}
	Hostexec           = ImageConfig{e2eRegistry, "hostexec", "1.0"}
	Iperf              = ImageConfig{e2eRegistry, "iperf", "1.0"}
	JessieDnsutils     = ImageConfig{e2eRegistry, "jessie-dnsutils", "1.0"}
	Kitten             = ImageConfig{e2eRegistry, "kitten", "1.0"}
	Liveness           = ImageConfig{e2eRegistry, "liveness", "1.0"}
	LogsGenerator      = ImageConfig{e2eRegistry, "logs-generator", "1.0"}
	Mounttest          = ImageConfig{e2eRegistry, "mounttest", "1.0"}
	MounttestUser      = ImageConfig{e2eRegistry, "mounttest-user", "1.0"}
	Nautilus           = ImageConfig{e2eRegistry, "nautilus", "1.0"}
	Net                = ImageConfig{e2eRegistry, "net", "1.0"}
	Netexec            = ImageConfig{e2eRegistry, "netexec", "1.0"}
	Nettest            = ImageConfig{e2eRegistry, "nettest", "1.0"}
	NginxSlim          = ImageConfig{gcRegistry, "nginx-slim", "0.20"}
	NginxSlimNew       = ImageConfig{gcRegistry, "nginx-slim", "0.21"}
	Nonewprivs         = ImageConfig{e2eRegistry, "nonewprivs", "1.0"}
	NoSnatTest         = ImageConfig{e2eRegistry, "no-snat-test", "1.0"}
	NoSnatTestProxy    = ImageConfig{e2eRegistry, "no-snat-test-proxy", "1.0"}
	NWayHTTP           = ImageConfig{e2eRegistry, "n-way-http", "1.0"}
	Pause              = ImageConfig{gcRegistry, "pause", "3.0"}
	Porter             = ImageConfig{e2eRegistry, "porter", "1.0"}
	PortForwardTester  = ImageConfig{e2eRegistry, "port-forward-tester", "1.0"}
	Redis              = ImageConfig{e2eRegistry, "redis", "1.0"}
	ResourceConsumer   = ImageConfig{e2eRegistry, "resource-consumer", "1.1"}
	ResourceController = ImageConfig{e2eRegistry, "resource-consumer/controller", "1.0"}
	ServeHostname      = ImageConfig{e2eRegistry, "serve-hostname", "1.0"}
	TestWebserver      = ImageConfig{e2eRegistry, "test-webserver", "1.0"}
)

func GetE2EImage(image ImageConfig) string {
	return fmt.Sprintf("%s/%s-%s:%s", image.registry, image.name, runtime.GOARCH, image.version)
}
