/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package client

import (
	"time"
)

// FlagSet abstracts the flag interface for compatibility with both Golang "flag"
// and cobra pflags (Posix style).
type FlagSet interface {
	StringVar(p *string, name, value, usage string)
	BoolVar(p *bool, name string, value bool, usage string)
	UintVar(p *uint, name string, value uint, usage string)
	DurationVar(p *time.Duration, name string, value time.Duration, usage string)
	Float32Var(p *float32, name string, value float32, usage string)
	IntVar(p *int, name string, value int, usage string)
}

// BindClientConfigFlags registers a standard set of CLI flags for connecting to a Kubernetes API server.
// TODO this method is superseded by pkg/client/clientcmd/client_builder.go
func BindClientConfigFlags(flags FlagSet, config *Config) {
	flags.StringVar(&config.Host, "master", config.Host, "The address of the Kubernetes API server")
	flags.StringVar(&config.Version, "api_version", config.Version, "The API version to use when talking to the server")
	flags.BoolVar(&config.Insecure, "insecure_skip_tls_verify", config.Insecure, "If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.")
	flags.StringVar(&config.CertFile, "client_certificate", config.CertFile, "Path to a client key file for TLS.")
	flags.StringVar(&config.KeyFile, "client_key", config.KeyFile, "Path to a client key file for TLS.")
	flags.StringVar(&config.CAFile, "certificate_authority", config.CAFile, "Path to a cert. file for the certificate authority.")
	flags.Float32Var(&config.QPS, "max_outgoing_qps", config.QPS, "Maximum number of queries per second that could be issued by this client.")
	flags.IntVar(&config.Burst, "max_outgoing_burst", config.Burst, "Maximum throttled burst")
}

func BindKubeletClientConfigFlags(flags FlagSet, config *KubeletConfig) {
	flags.BoolVar(&config.EnableHttps, "kubelet_https", config.EnableHttps, "Use https for kubelet connections")
	flags.UintVar(&config.Port, "kubelet_port", config.Port, "Kubelet port")
	flags.DurationVar(&config.HTTPTimeout, "kubelet_timeout", config.HTTPTimeout, "Timeout for kubelet operations")
	flags.StringVar(&config.CertFile, "kubelet_client_certificate", config.CertFile, "Path to a client key file for TLS.")
	flags.StringVar(&config.KeyFile, "kubelet_client_key", config.KeyFile, "Path to a client key file for TLS.")
	flags.StringVar(&config.CAFile, "kubelet_certificate_authority", config.CAFile, "Path to a cert. file for the certificate authority.")
}
