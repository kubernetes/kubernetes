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

package gcp

import (
	"net/http"
	"time"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/cloud-provider/credentialconfig"
	"k8s.io/kubernetes/pkg/credentialprovider"
	"k8s.io/legacy-cloud-providers/gce/gcpcredential"
)

// init registers the various means by which credentials may
// be resolved on GCP.
func init() {
	tr := utilnet.SetTransportDefaults(&http.Transport{})
	metadataHTTPClientTimeout := time.Second * 10
	httpClient := &http.Client{
		Transport: tr,
		Timeout:   metadataHTTPClientTimeout,
	}
	credentialprovider.RegisterCredentialProvider("google-dockercfg",
		&credentialconfig.CachingDockerConfigProvider{
			Provider: &gcpcredential.DockerConfigKeyProvider{
				gcpcredential.MetadataProvider{Client: httpClient},
			},
			Lifetime: 60 * time.Second,
		})

	credentialprovider.RegisterCredentialProvider("google-dockercfg-url",
		&credentialconfig.CachingDockerConfigProvider{
			Provider: &gcpcredential.DockerConfigURLKeyProvider{
				gcpcredential.MetadataProvider{Client: httpClient},
			},
			Lifetime: 60 * time.Second,
		})

	credentialprovider.RegisterCredentialProvider("google-container-registry",
		// Never cache this.  The access token is already
		// cached by the metadata service.
		&gcpcredential.ContainerRegistryProvider{
			gcpcredential.MetadataProvider{Client: httpClient},
		})
}
