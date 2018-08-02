/*
Copyright 2016 The Kubernetes Authors.

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

package certs

/*

	PHASE: CERTIFICATES

	INPUTS:
		From InitConfiguration
			.API.AdvertiseAddress is an optional parameter that can be passed for an extra addition to the SAN IPs
			.APIServerCertSANs is an optional parameter for adding DNS names and IPs to the API Server serving cert SAN
			.Etcd.Local.ServerCertSANs is an optional parameter for adding DNS names and IPs to the etcd serving cert SAN
			.Etcd.Local.PeerCertSANs is an optional parameter for adding DNS names and IPs to the etcd peer cert SAN
			.Networking.DNSDomain is needed for knowing which DNS name the internal kubernetes service has
			.Networking.ServiceSubnet is needed for knowing which IP the internal kubernetes service is going to point to
			.CertificatesDir is required for knowing where all certificates should be stored

	OUTPUTS:
		Files to .CertificatesDir (default /etc/kubernetes/pki):
		 - ca.crt
		 - ca.key
		 - apiserver.crt
		 - apiserver.key
		 - apiserver-kubelet-client.crt
		 - apiserver-kubelet-client.key
		 - apiserver-etcd-client.crt
		 - apiserver-etcd-client.key
		 - etcd/ca.crt
		 - etcd/ca.key
		 - etcd/server.crt
		 - etcd/server.key
		 - etcd/peer.crt
		 - etcd/peer.key
		 - etcd/healthcheck-client.crt
		 - etcd/healthcheck-client.key
		 - sa.pub
		 - sa.key
		 - front-proxy-ca.crt
		 - front-proxy-ca.key
		 - front-proxy-client.crt
		 - front-proxy-client.key

*/
