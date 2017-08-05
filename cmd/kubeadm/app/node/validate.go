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

package node

import (
	"fmt"

	certsapi "k8s.io/api/certificates/v1beta1"
	clientset "k8s.io/client-go/kubernetes"
)

// ValidateAPIServer makes sure the server we're connecting to supports the Beta Certificates API
func ValidateAPIServer(client clientset.Interface) error {
	version, err := client.Discovery().ServerVersion()
	if err != nil {
		return fmt.Errorf("failed to check server version: %v", err)
	}
	fmt.Printf("[bootstrap] Detected server version: %s\n", version.String())

	// Check certificates API. If the server supports the version of the Certificates API we're using, we're good to go
	serverGroups, err := client.Discovery().ServerGroups()
	if err != nil {
		return fmt.Errorf("certificate API check failed: failed to retrieve a list of supported API objects [%v]", err)
	}
	for _, group := range serverGroups.Groups {
		if group.Name == certsapi.SchemeGroupVersion.Group {
			for _, version := range group.Versions {
				if version.Version == certsapi.SchemeGroupVersion.Version {
					fmt.Printf("[bootstrap] The server supports the Certificates API (%s/%s)\n", certsapi.SchemeGroupVersion.Group, certsapi.SchemeGroupVersion.Version)
					return nil
				}
			}
		}
	}
	return fmt.Errorf("certificate API check failed: API server with version %s doesn't support Certificates API (%s/%s), use v1.6.0 or newer",
		version.String(), certsapi.SchemeGroupVersion.Group, certsapi.SchemeGroupVersion.Version)
}
