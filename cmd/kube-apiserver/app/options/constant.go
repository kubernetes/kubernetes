/*
Copyright 2018 The Kubernetes Authors.

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

package options

const (
	// KubeApiServerCloudProviderGceLbSrcCidrs sets google compute engine load balancer cidrs source.
	KubeApiServerCloudProviderGceLbSrcCidrs = "cloud-provider-gce-lb-src-cidrs"

	// KubeApiServerDefaultNotReadyTolerationSeconds sets default not ready toleration seconds.
	KubeApiServerDefaultNotReadyTolerationSeconds = "default-not-ready-toleration-seconds"

	// KubeApiServerDefaultUnreachableTolerationSeconds sets default unreachable toleration seconds.
	KubeApiServerDefaultUnreachableTolerationSeconds = "default-unreachable-toleration-seconds"

	// KubeApiServerGeneric sets generic kube apiserver options.
	KubeApiServerGeneric = "generic"

	// KubeApiServerEtcd sets etcd options.
	KubeApiServerEtcd = "etcd"

	// KubeApiServerSecureServing defines secure serving.
	KubeApiServerSecureServing = "secure serving"

	// KubeApiServerInsecureServing defines secure serving.
	KubeApiServerInsecureServing = "insecure serving"

	// KubeApiServerAuditing sets auditing option.
	KubeApiServerAuditing = "auditing"

	// KubeApiServerFeatures sets features.
	KubeApiServerFeatures = "features"

	// KubeApiServerAuthentication sets authentication parameters.
	KubeApiServerAuthentication = "authentication"

	// KubeApiServerAuthorization sets authorization parameters.
	KubeApiServerAuthorization = "authorization"

	// KubeApiServerCloudProvider sets cloud provider.
	KubeApiServerCloudProvider = "cloud provider"

	// KubeApiServerStorage sets storage.
	KubeApiServerStorage = "storage"

	// KubeApiServerApiEnablement sets api enablement status.
	KubeApiServerApiEnablement = "api enablement"

	// KubeApiServerAdmission sets admission options.
	KubeApiServerAdmission = "admission"

	// KubeApiServerMisc sets misc options.
	KubeApiServerMisc = "misc"

	// KubeApiServerEventTtl sets amount of time to retain events.
	KubeApiServerEventTtl = "event-ttl"

	// KubeApiServerAllowPrivileged if true, allow privileged containers.
	KubeApiServerAllowPrivileged = "allow-privileged"

	// KubeApiServerEnableLogsHandler if true, install a /logs handler for the apiserver logs.
	KubeApiServerEnableLogsHandler = "enable-logs-handler"

	// KubeApiServerSshUser if non-empty, use secure SSH proxy to the nodes, using this user name.
	KubeApiServerSshUser = "ssh-user"

	// KubeApiServerSshKeyFile if non-empty, use secure SSH proxy to the nodes, using this user keyfile.
	KubeApiServerSshKeyFile = "ssh-keyfile"

	// KubeApiServerMaxConnectionBytesPerSec if non-zero, throttle each user connection to this number of bytes/sec.
	KubeApiServerMaxConnectionBytesPerSec = "max-connection-bytes-per-sec"

	// KubeApiServerCount sets the number of apiservers running in the cluster, must be a positive number.
	KubeApiServerCount = "apiserver-count"

	// KubeApiServerEndpointReconcilerType defines if should use an endpoint reconciler.
	KubeApiServerEndpointReconcilerType = "endpoint-reconciler-type"

	// KubeApiServerKubernetesServiceNodePort defines default kubernetes service node port.
	KubeApiServerKubernetesServiceNodePort = "kubernetes-service-node-port"

	// KubeApiServerServiceClusterIpRange sets service cluster ip range.
	KubeApiServerServiceClusterIpRange = "service-cluster-ip-range"

	// KubeApiServerServiceNodePortRange sets service node port range.
	KubeApiServerServiceNodePortRange = "service-node-port-range"

	// KubeApiServerKubeletHttps difines if should use https for kubelet connections.
	KubeApiServerKubeletHttps = "kubelet-https"

	// KubeApiServerKubeletPreferredAddressTypes sets kubelet preferred address types.
	KubeApiServerKubeletPreferredAddressTypes = "kubelet-preferred-address-types"

	// KubeApiServerKubeletPort sets kubelet port.
	KubeApiServerKubeletPort = "kubelet-port"

	// KubeApiServerKubeletReadOnlyPort sets kubelet read only port.
	KubeApiServerKubeletReadOnlyPort = "kubelet-read-only-port"

	// KubeApiServerKubeletTimeout sets kubelet timeout.
	KubeApiServerKubeletTimeout = "kubelet-timeout"

	// KubeApiServerKubeletClientCertificate sets kubelet client certificate.
	KubeApiServerKubeletClientCertificate = "kubelet-client-certificate"

	// KubeApiServerKubeletClientKey sets kubelet client key.
	KubeApiServerKubeletClientKey = "kubelet-client-key"

	// KubeApiServerKubeletCertificateAuthority sets kubelet certificate authority.
	KubeApiServerKubeletCertificateAuthority = "kubelet-certificate-authority"

	// KubeApiServerRepairMalformedUpdates deprecated.
	KubeApiServerRepairMalformedUpdates = "repair-malformed-updates"

	// KubeApiServerProxyClientCertFile sets proxy client cert file.
	KubeApiServerProxyClientCertFile = "proxy-client-cert-file"

	// KubeApiServerProxyClientKeyFile sets proxy client key file.
	KubeApiServerProxyClientKeyFile = "proxy-client-key-file"

	// KubeApiServerEnableAggregatorRouting defines if aggregator routing is enabled.
	KubeApiServerEnableAggregatorRouting = "enable-aggregator-routing"

	// KubeApiServerServiceAccountSigningKeyFile sets service account signing key file.
	KubeApiServerServiceAccountSigningKeyFile = "service-account-signing-key-file"
)
