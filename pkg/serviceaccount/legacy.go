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

package serviceaccount

import (
	"k8s.io/api/core/v1"
	apiserverserviceaccount "k8s.io/apiserver/pkg/authentication/serviceaccount"

	"gopkg.in/square/go-jose.v2/jwt"
)

func LegacyClaims(serviceAccount v1.ServiceAccount, secret v1.Secret) (*jwt.Claims, interface{}) {
	return &jwt.Claims{
			Subject: apiserverserviceaccount.MakeUsername(serviceAccount.Namespace, serviceAccount.Name),
		}, &legacyPrivateClaims{
			Namespace:          serviceAccount.Namespace,
			ServiceAccountName: serviceAccount.Name,
			ServiceAccountUID:  string(serviceAccount.UID),
			SecretName:         secret.Name,
		}
}

const LegacyIssuer = "kubernetes/serviceaccount"

type legacyPrivateClaims struct {
	ServiceAccountName string `json:"kubernetes.io/serviceaccount/service-account.name"`
	ServiceAccountUID  string `json:"kubernetes.io/serviceaccount/service-account.uid"`
	SecretName         string `json:"kubernetes.io/serviceaccount/secret.name"`
	Namespace          string `json:"kubernetes.io/serviceaccount/namespace"`
}
