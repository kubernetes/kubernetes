/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package genericapiserver

import (
	"net"
)

const (
	// TODO: This can be tightened up. It still matches objects named watch or proxy.
	defaultLongRunningRequestRE = "(/|^)((watch|proxy)(/|$)|(logs?|portforward|exec|attach)/?$)"
)

// ServerRunOptions contains the options while running a generic api server.
type ServerRunOptions struct {
	BindAddress          net.IP
	CertDirectory        string
	ClientCAFile         string
	EtcdQuorumRead       bool
	InsecureBindAddress  net.IP
	InsecurePort         int
	LongRunningRequestRE string
	MaxRequestsInFlight  int
	SecurePort           int
	TLSCertFile          string
	TLSPrivateKeyFile    string
}

func NewServerRunOptions() *ServerRunOptions {
	return &ServerRunOptions{
		BindAddress:          net.ParseIP("0.0.0.0"),
		CertDirectory:        "/var/run/kubernetes",
		InsecureBindAddress:  net.ParseIP("127.0.0.1"),
		InsecurePort:         8080,
		LongRunningRequestRE: defaultLongRunningRequestRE,
		SecurePort:           6443,
	}
}
