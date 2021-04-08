/*
Copyright 2021 The Kubernetes Authors.

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

package v1alpha1

const (
	// KubeProxyOOMScoreAdj is the OOM score adjustment for kube-proxy
	KubeProxyOOMScoreAdj int = -999

	// ProxyStatusPort is the default port for the proxy metrics server.
	// May be overridden by a flag at startup.
	ProxyStatusPort = 10249

	// ProxyHealthzPort is the default port for the proxy healthz server.
	// May be overridden by a flag at startup.
	ProxyHealthzPort = 10256
)
