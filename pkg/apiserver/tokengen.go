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

package apiserver

import (
	serviceaccountregistry "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/serviceaccount"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/serviceaccount"

	"github.com/golang/glog"
)

func NewTokenGenerator(serviceAccountPrivateKeyFile string) serviceaccountregistry.TokenGenerator {
	if len(serviceAccountPrivateKeyFile) == 0 {
		return nil
	}

	privateKey, err := serviceaccount.ReadPrivateKey(serviceAccountPrivateKeyFile)
	if err != nil {
		glog.Fatalf("Error reading key for service account token generator: %v", err)
	}

	return serviceaccount.JWTTokenGenerator(privateKey)
}
