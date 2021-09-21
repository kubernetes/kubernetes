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

// Package options contains flags and options for initializing kube-apiserver
package options

import (
	genericoptions "k8s.io/apiserver/pkg/server/options"
	netutils "k8s.io/utils/net"
)

// NewSecureServingOptions gives default values for the kube-apiserver which are not the options wanted by
// "normal" API servers running on the platform
func NewSecureServingOptions() *genericoptions.SecureServingOptionsWithLoopback {
	o := genericoptions.SecureServingOptions{
		BindAddress: netutils.ParseIPSloppy("0.0.0.0"),
		BindPort:    6443,
		Required:    true,
		ServerCert: genericoptions.GeneratableKeyCert{
			PairName:      "apiserver",
			CertDirectory: "/var/run/kubernetes",
		},
	}
	return o.WithLoopback()
}
