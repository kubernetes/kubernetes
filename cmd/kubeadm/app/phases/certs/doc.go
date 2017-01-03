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
		From MasterConfiguration
			.API.AdvertiseAddresses is needed for knowing which IPs the certs should be signed for
			.API.ExternalDNSNames is needed for knowing which DNS names the certs should be signed for
			.Networking.DNSDomain is needed for knowing which DNS name the internal kubernetes service has
			.Networking.ServiceSubnet is needed for knowing which IP the internal kubernetes service is going to point to
			The PKIPath is required for knowing where all certificates should be stored

	OUTPUTS:
		Files to PKIPath (default /etc/kubernetes/pki):
		 - apiserver-key.pem
		 - apiserver-pub.pem
		 - apiserver.pem
		 - ca-key.pem
		 - ca-pub.pem
		 - ca.pem
		 - sa-key.pem
		 - sa-pub.pem
*/
