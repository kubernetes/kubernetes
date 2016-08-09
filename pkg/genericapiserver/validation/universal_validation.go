/*
Copyright 2014 The Kubernetes Authors.

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

package validation

import (
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/genericapiserver/options"
)

// TODO: Longer term we should read this from some config store, rather than a flag.
func verifyClusterIPFlags(options *options.ServerRunOptions) {
	if options.ServiceClusterIPRange.IP == nil {
		glog.Fatal("No --service-cluster-ip-range specified")
	}
	var ones, bits = options.ServiceClusterIPRange.Mask.Size()
	if bits-ones > 20 {
		glog.Fatal("Specified --service-cluster-ip-range is too large")
	}
}

func verifyServiceNodePort(options *options.ServerRunOptions) {
	if options.KubernetesServiceNodePort < 0 || options.KubernetesServiceNodePort > 65535 {
		glog.Fatalf("--kubernetes-service-node-port %v must be between 0 and 65535, inclusive. If 0, the Kubernetes master service will be of type ClusterIP.", options.KubernetesServiceNodePort)
	}

	if options.KubernetesServiceNodePort > 0 && !options.ServiceNodePortRange.Contains(options.KubernetesServiceNodePort) {
		glog.Fatalf("Kubernetes service port range %v doesn't contain %v", options.ServiceNodePortRange, (options.KubernetesServiceNodePort))
	}
}

func verifySecureAndInsecurePort(options *options.ServerRunOptions) {
	if options.SecurePort < 0 || options.SecurePort > 65535 {
		glog.Fatalf("--secure-port %v must be between 0 and 65535, inclusive. 0 for turning off secure port.", options.SecurePort)
	}

	// TODO: Allow 0 to turn off insecure port.
	if options.InsecurePort < 1 || options.InsecurePort > 65535 {
		glog.Fatalf("--insecure-port %v must be between 1 and 65535, inclusive.", options.InsecurePort)
	}

	if options.SecurePort == options.InsecurePort {
		glog.Fatalf("--secure-port and --insecure-port cannot use the same port.")
	}
}

func ValidateRunOptions(options *options.ServerRunOptions) {
	verifyClusterIPFlags(options)
	verifyServiceNodePort(options)
	verifySecureAndInsecurePort(options)
}
