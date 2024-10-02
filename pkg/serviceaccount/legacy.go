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
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"gopkg.in/square/go-jose.v2/jwt"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/audit"
	apiserverserviceaccount "k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/warning"
	applyv1 "k8s.io/client-go/applyconfigurations/core/v1"
	typedv1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/klog/v2"
)

const InvalidSinceLabelKey = "kubernetes.io/legacy-token-invalid-since"

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

const (
	LegacyIssuer     = "kubernetes/serviceaccount"
	LastUsedLabelKey = "kubernetes.io/legacy-token-last-used"
)

type legacyPrivateClaims struct {
	ServiceAccountName string `json:"kubernetes.io/serviceaccount/service-account.name"`
	ServiceAccountUID  string `json:"kubernetes.io/serviceaccount/service-account.uid"`
	SecretName         string `json:"kubernetes.io/serviceaccount/secret.name"`
	Namespace          string `json:"kubernetes.io/serviceaccount/namespace"`
}

func NewLegacyValidator(lookup bool, getter ServiceAccountTokenGetter, secretsWriter typedv1core.SecretsGetter) (Validator[legacyPrivateClaims], error) {
	if lookup && getter == nil {
		return nil, errors.New("ServiceAccountTokenGetter must be provided")
	}
	if lookup && secretsWriter == nil {
		return nil, errors.New("SecretsWriter must be provided")
	}
	return &legacyValidator{
		lookup:        lookup,
		getter:        getter,
		secretsWriter: secretsWriter,
	}, nil
}

type legacyValidator struct {
	lookup        bool
	getter        ServiceAccountTokenGetter
	secretsWriter typedv1core.SecretsGetter
}

var _ = Validator[legacyPrivateClaims](&legacyValidator{})

func (v *legacyValidator) Validate(ctx context.Context, tokenData string, public *jwt.Claims, private *legacyPrivateClaims) (*apiserverserviceaccount.ServiceAccountInfo, error) {
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
		if subtle.ConstantTimeCompare(secret.Data[v1.ServiceAccountTokenKey], []byte(tokenData)) == 0 {
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

		// Track secret-based long-lived service account tokens and add audit annotations and metrics.
		autoGenerated := false

		// Check if the secret has been marked as invalid
		if invalidSince := secret.Labels[InvalidSinceLabelKey]; invalidSince != "" {
			audit.AddAuditAnnotation(ctx, "authentication.k8s.io/legacy-token-invalidated", secret.Name+"/"+secret.Namespace)
			invalidatedAutoTokensTotal.WithContext(ctx).Inc()
			v.patchSecretWithLastUsedDate(ctx, secret)
			return nil, fmt.Errorf("the token in secret %s/%s for service account %s/%s has been marked invalid. Use tokens from the TokenRequest API or manually created secret-based tokens, or remove the '%s' label from the secret to temporarily allow use of this token", namespace, secretName, namespace, serviceAccountName, InvalidSinceLabelKey)
		}

		// Check if it is an auto-generated secret-based token
		for _, ref := range serviceAccount.Secrets {
			if ref.Name == secret.Name {
				autoGenerated = true
				warning.AddWarning(ctx, "", "Use tokens from the TokenRequest API or manually created secret-based tokens instead of auto-generated secret-based tokens.")
				audit.AddAuditAnnotation(ctx, "authentication.k8s.io/legacy-token-autogenerated-secret", secret.Name)
				autoGeneratedTokensTotal.WithContext(ctx).Inc()
				break
			}
		}

		// Check if it's a manually created secret-based token
		if !autoGenerated {
			audit.AddAuditAnnotation(ctx, "authentication.k8s.io/legacy-token-manual-secret", secret.Name)
			manuallyCreatedTokensTotal.WithContext(ctx).Inc()
		}

		v.patchSecretWithLastUsedDate(ctx, secret)
	}

	return &apiserverserviceaccount.ServiceAccountInfo{
		Namespace: private.Namespace,
		Name:      private.ServiceAccountName,
		UID:       private.ServiceAccountUID,
	}, nil
}

func (v *legacyValidator) patchSecretWithLastUsedDate(ctx context.Context, secret *v1.Secret) {
	now := time.Now().UTC()
	today := now.Format("2006-01-02")
	tomorrow := now.AddDate(0, 0, 1).Format("2006-01-02")
	lastUsed := secret.Labels[LastUsedLabelKey]
	if lastUsed != today && lastUsed != tomorrow {
		patchContent, err := json.Marshal(applyv1.Secret(secret.Name, secret.Namespace).WithUID(secret.UID).WithLabels(map[string]string{LastUsedLabelKey: today}))
		if err != nil {
			klog.Errorf("Failed to marshal legacy service account token %s/%s tracking labels, err: %v", secret.Name, secret.Namespace, err)
		} else {
			if _, err := v.secretsWriter.Secrets(secret.Namespace).Patch(ctx, secret.Name, types.MergePatchType, patchContent, metav1.PatchOptions{}); err != nil {
				klog.Errorf("Failed to label legacy service account token %s/%s with last-used date, err: %v", secret.Name, secret.Namespace, err)
			}
		}
	}
}
