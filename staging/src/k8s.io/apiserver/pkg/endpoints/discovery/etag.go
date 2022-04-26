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

package discovery

import (
	"crypto/sha512"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// This file exposes helper functions used for calculating the E-Tag header
// used in discovery endpoint responses

func CalculateETag(resurces *metav1.APIResourceList) (string, error) {
	serialized, err := resurces.Marshal()
	if err != nil {
		return "", err
	}

	return fmt.Sprintf("%X", sha512.Sum512(serialized)), nil
}
