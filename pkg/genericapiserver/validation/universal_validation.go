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
	"fmt"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/genericapiserver/options"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
)

// TODO: Longer term we should read this from some config store, rather than a flag.
func verifyClusterIPFlags(options *options.ServerRunOptions) []error {
	errors := []error{}
	if options.ServiceClusterIPRange.IP == nil {
		errors = append(errors, fmt.Errorf("No --service-cluster-ip-range specified"))
	}
	var ones, bits = options.ServiceClusterIPRange.Mask.Size()
	if bits-ones > 20 {
		errors = append(errors, fmt.Errorf("Specified --service-cluster-ip-range is too large"))
	}
	return errors
}

func verifyServiceNodePort(options *options.ServerRunOptions) []error {
	errors := []error{}
	if options.KubernetesServiceNodePort < 0 || options.KubernetesServiceNodePort > 65535 {
		errors = append(errors, fmt.Errorf("--kubernetes-service-node-port %v must be between 0 and 65535, inclusive. If 0, the Kubernetes master service will be of type ClusterIP.", options.KubernetesServiceNodePort))
	}

	if options.KubernetesServiceNodePort > 0 && !options.ServiceNodePortRange.Contains(options.KubernetesServiceNodePort) {
		errors = append(errors, fmt.Errorf("Kubernetes service port range %v doesn't contain %v", options.ServiceNodePortRange, (options.KubernetesServiceNodePort)))
	}
	return errors
}

func verifySecureAndInsecurePort(options *options.ServerRunOptions) []error {
	errors := []error{}
	if options.SecurePort < 0 || options.SecurePort > 65535 {
		errors = append(errors, fmt.Errorf("--secure-port %v must be between 0 and 65535, inclusive. 0 for turning off secure port.", options.SecurePort))
	}

	if options.InsecurePort < 0 || options.InsecurePort > 65535 {
		errors = append(errors, fmt.Errorf("--insecure-port %v must be between 0 and 65535, inclusive. 0 for turning off insecure port.", options.InsecurePort))
	}

	if options.SecurePort == 0 && options.InsecurePort == 0 {
		glog.Fatalf("--secure-port and --insecure-port cannot be turned off at the same time.")
	}

	if options.SecurePort == options.InsecurePort {
		errors = append(errors, fmt.Errorf("--secure-port and --insecure-port cannot use the same port."))
	}
	return errors
}

func ValidateRunOptions(options *options.ServerRunOptions) {
	errors := []error{}
	if errs := verifyClusterIPFlags(options); len(errs) > 0 {
		errors = append(errors, errs...)
	}
	if errs := verifyServiceNodePort(options); len(errs) > 0 {
		errors = append(errors, errs...)
	}
	if errs := verifySecureAndInsecurePort(options); len(errs) > 0 {
		errors = append(errors, errs...)
	}
	if err := utilerrors.NewAggregate(errors); err != nil {
		glog.Fatalf("Validate server run options failed: %v", err)
	}
}
