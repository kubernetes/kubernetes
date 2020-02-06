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
	"bytes"
	"errors"
	"fmt"

	"gopkg.in/square/go-jose.v2/jwt"
	"k8s.io/klog"

	"k8s.io/api/core/v1"
	apiserverserviceaccount "k8s.io/apiserver/pkg/authentication/serviceaccount"
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

func NewLegacyValidator(lookup bool, getter ServiceAccountTokenGetter) Validator {
	return &legacyValidator{
		lookup: lookup,
		getter: getter,
	}
}

type legacyValidator struct {
	lookup bool
	getter ServiceAccountTokenGetter
}

var _ = Validator(&legacyValidator{})

func (v *legacyValidator) Validate(tokenData string, public *jwt.Claims, privateObj interface{}) (*ServiceAccountInfo, error) {
	private, ok := privateObj.(*legacyPrivateClaims)
	if !ok {
		klog.Errorf("jwt validator expected private claim of type *legacyPrivateClaims but got: %T", privateObj)
		return nil, errors.New("Token could not be validated.")
	}

	// Make sure the claims we need exist
	if len(public.Subject) == 0 {
		return nil, errors.New("sub claim is missing")
	}
	namespace := private.Namespace
	if len(namespace) == 0 {
		return nil, errors.New("namespace claim is missing")
	}
	secretName := private.SecretName
	if len(secretName) == 0 {
		return nil, errors.New("secretName claim is missing")
	}
	serviceAccountName := private.ServiceAccountName
	if len(serviceAccountName) == 0 {
		return nil, errors.New("serviceAccountName claim is missing")
	}
	serviceAccountUID := private.ServiceAccountUID
	if len(serviceAccountUID) == 0 {
		return nil, errors.New("serviceAccountUID claim is missing")
	}

	subjectNamespace, subjectName, err := apiserverserviceaccount.SplitUsername(public.Subject)
	if err != nil || subjectNamespace != namespace || subjectName != serviceAccountName {
		return nil, errors.New("sub claim is invalid")
	}

	if v.lookup {
		// Make sure token hasn't been invalidated by deletion of the secret
		secret, err := v.getter.GetSecret(namespace, secretName)
		if err != nil {
			klog.V(4).Infof("Could not retrieve token %s/%s for service account %s/%s: %v", namespace, secretName, namespace, serviceAccountName, err)
			return nil, errors.New("Token has been invalidated")
		}
		if secret.DeletionTimestamp != nil {
			klog.V(4).Infof("Token is deleted and awaiting removal: %s/%s for service account %s/%s", namespace, secretName, namespace, serviceAccountName)
			return nil, errors.New("Token has been invalidated")
		}
		if !bytes.Equal(secret.Data[v1.ServiceAccountTokenKey], []byte(tokenData)) {
			klog.V(4).Infof("Token contents no longer matches %s/%s for service account %s/%s", namespace, secretName, namespace, serviceAccountName)
			return nil, errors.New("Token does not match server's copy")
		}

		// Make sure service account still exists (name and UID)
		serviceAccount, err := v.getter.GetServiceAccount(namespace, serviceAccountName)
		if err != nil {
			klog.V(4).Infof("Could not retrieve service account %s/%s: %v", namespace, serviceAccountName, err)
			return nil, err
		}
		if serviceAccount.DeletionTimestamp != nil {
			klog.V(4).Infof("Service account has been deleted %s/%s", namespace, serviceAccountName)
			return nil, fmt.Errorf("ServiceAccount %s/%s has been deleted", namespace, serviceAccountName)
		}
		if string(serviceAccount.UID) != serviceAccountUID {
			klog.V(4).Infof("Service account UID no longer matches %s/%s: %q != %q", namespace, serviceAccountName, string(serviceAccount.UID), serviceAccountUID)
			return nil, fmt.Errorf("ServiceAccount UID (%s) does not match claim (%s)", serviceAccount.UID, serviceAccountUID)
		}
	}

	return &ServiceAccountInfo{
		Namespace: private.Namespace,
		Name:      private.ServiceAccountName,
		UID:       private.ServiceAccountUID,
	}, nil
}

func (v *legacyValidator) NewPrivateClaims() interface{} {
	return &legacyPrivateClaims{}
}
