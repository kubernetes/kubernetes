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
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
	"gopkg.in/square/go-jose.v2/jwt"

	"k8s.io/apiserver/pkg/audit"
	apiserverserviceaccount "k8s.io/apiserver/pkg/authentication/serviceaccount"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

const (
	// Injected bound service account token expiration which triggers monitoring of its time-bound feature.
	WarnOnlyBoundTokenExpirationSeconds = 60*60 + 7

	// Extended expiration for those modified tokens involved in safe rollout if time-bound feature.
	ExpirationExtensionSeconds = 24 * 365 * 60 * 60
)

var (
	// time.Now stubbed out to allow testing
	now = time.Now
	// uuid.New stubbed out to allow testing
	newUUID = uuid.NewString
)

type privateClaims struct {
	Kubernetes kubernetes `json:"kubernetes.io,omitempty"`
}

type kubernetes struct {
	Namespace string           `json:"namespace,omitempty"`
	Svcacct   ref              `json:"serviceaccount,omitempty"`
	Pod       *ref             `json:"pod,omitempty"`
	Secret    *ref             `json:"secret,omitempty"`
	Node      *ref             `json:"node,omitempty"`
	WarnAfter *jwt.NumericDate `json:"warnafter,omitempty"`
}

type ref struct {
	Name string `json:"name,omitempty"`
	UID  string `json:"uid,omitempty"`
}

func Claims(sa core.ServiceAccount, pod *core.Pod, secret *core.Secret, node *core.Node, expirationSeconds, warnafter int64, audience []string) (*jwt.Claims, interface{}, error) {
	now := now()
	sc := &jwt.Claims{
		Subject:   apiserverserviceaccount.MakeUsername(sa.Namespace, sa.Name),
		Audience:  jwt.Audience(audience),
		IssuedAt:  jwt.NewNumericDate(now),
		NotBefore: jwt.NewNumericDate(now),
		Expiry:    jwt.NewNumericDate(now.Add(time.Duration(expirationSeconds) * time.Second)),
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.ServiceAccountTokenJTI) {
		sc.ID = newUUID()
	}
	pc := &privateClaims{
		Kubernetes: kubernetes{
			Namespace: sa.Namespace,
			Svcacct: ref{
				Name: sa.Name,
				UID:  string(sa.UID),
			},
		},
	}

	if secret != nil && (node != nil || pod != nil) {
		return nil, nil, fmt.Errorf("internal error, token can only be bound to one object type")
	}
	switch {
	case pod != nil:
		pc.Kubernetes.Pod = &ref{
			Name: pod.Name,
			UID:  string(pod.UID),
		}
		if utilfeature.DefaultFeatureGate.Enabled(features.ServiceAccountTokenPodNodeInfo) {
			// if this is bound to a pod and the node information is available, persist that too
			if node != nil {
				pc.Kubernetes.Node = &ref{
					Name: node.Name,
					UID:  string(node.UID),
				}
			}
		}
	case secret != nil:
		pc.Kubernetes.Secret = &ref{
			Name: secret.Name,
			UID:  string(secret.UID),
		}
	case node != nil:
		if !utilfeature.DefaultFeatureGate.Enabled(features.ServiceAccountTokenNodeBinding) {
			return nil, nil, fmt.Errorf("token bound to Node object requested, but %q feature gate is disabled", features.ServiceAccountTokenNodeBinding)
		}
		pc.Kubernetes.Node = &ref{
			Name: node.Name,
			UID:  string(node.UID),
		}
	}

	if warnafter != 0 {
		pc.Kubernetes.WarnAfter = jwt.NewNumericDate(now.Add(time.Duration(warnafter) * time.Second))
	}

	return sc, pc, nil
}

func NewValidator(getter ServiceAccountTokenGetter) Validator[privateClaims] {
	return &validator{
		getter: getter,
	}
}

type validator struct {
	getter ServiceAccountTokenGetter
}

var _ = Validator[privateClaims](&validator{})

func (v *validator) Validate(ctx context.Context, _ string, public *jwt.Claims, private *privateClaims) (*apiserverserviceaccount.ServiceAccountInfo, error) {
	nowTime := now()
	err := public.Validate(jwt.Expected{
		Time: nowTime,
	})
	switch err {
	case nil:
		// successful validation

	case jwt.ErrExpired:
		return nil, errors.New("service account token has expired")

	case jwt.ErrNotValidYet:
		return nil, errors.New("service account token is not valid yet")

	case jwt.ErrIssuedInTheFuture:
		return nil, errors.New("service account token is issued in the future")

	// our current use of jwt.Expected above should make these cases impossible to hit
	case jwt.ErrInvalidAudience, jwt.ErrInvalidID, jwt.ErrInvalidIssuer, jwt.ErrInvalidSubject:
		klog.Errorf("service account token claim validation got unexpected validation failure: %v", err)
		return nil, fmt.Errorf("service account token claims could not be validated: %w", err) // safe to pass these errors back to the user

	default:
		klog.Errorf("service account token claim validation got unexpected error type: %T", err)                         // avoid leaking unexpected information into the logs
		return nil, errors.New("service account token claims could not be validated due to unexpected validation error") // return an opaque error
	}

	// consider things deleted prior to now()-leeway to be invalid
	invalidIfDeletedBefore := nowTime.Add(-jwt.DefaultLeeway)
	namespace := private.Kubernetes.Namespace
	saref := private.Kubernetes.Svcacct
	podref := private.Kubernetes.Pod
	noderef := private.Kubernetes.Node
	secref := private.Kubernetes.Secret
	// Make sure service account still exists (name and UID)
	serviceAccount, err := v.getter.GetServiceAccount(namespace, saref.Name)
	if err != nil {
		klog.V(4).Infof("Could not retrieve service account %s/%s: %v", namespace, saref.Name, err)
		return nil, err
	}

	if string(serviceAccount.UID) != saref.UID {
		klog.V(4).Infof("Service account UID no longer matches %s/%s: %q != %q", namespace, saref.Name, string(serviceAccount.UID), saref.UID)
		return nil, fmt.Errorf("service account UID (%s) does not match claim (%s)", serviceAccount.UID, saref.UID)
	}
	if serviceAccount.DeletionTimestamp != nil && serviceAccount.DeletionTimestamp.Time.Before(invalidIfDeletedBefore) {
		klog.V(4).Infof("Service account has been deleted %s/%s", namespace, saref.Name)
		return nil, fmt.Errorf("service account %s/%s has been deleted", namespace, saref.Name)
	}

	if secref != nil {
		// Make sure token hasn't been invalidated by deletion of the secret
		secret, err := v.getter.GetSecret(namespace, secref.Name)
		if err != nil {
			klog.V(4).Infof("Could not retrieve bound secret %s/%s for service account %s/%s: %v", namespace, secref.Name, namespace, saref.Name, err)
			return nil, errors.New("service account token has been invalidated")
		}
		if secref.UID != string(secret.UID) {
			klog.V(4).Infof("Secret UID no longer matches %s/%s: %q != %q", namespace, secref.Name, string(secret.UID), secref.UID)
			return nil, fmt.Errorf("secret UID (%s) does not match service account secret ref claim (%s)", secret.UID, secref.UID)
		}
		if secret.DeletionTimestamp != nil && secret.DeletionTimestamp.Time.Before(invalidIfDeletedBefore) {
			klog.V(4).Infof("Bound secret is deleted and awaiting removal: %s/%s for service account %s/%s", namespace, secref.Name, namespace, saref.Name)
			return nil, errors.New("service account token has been invalidated")
		}
	}

	var podName, podUID string
	if podref != nil {
		// Make sure token hasn't been invalidated by deletion of the pod
		pod, err := v.getter.GetPod(namespace, podref.Name)
		if err != nil {
			klog.V(4).Infof("Could not retrieve bound pod %s/%s for service account %s/%s: %v", namespace, podref.Name, namespace, saref.Name, err)
			return nil, errors.New("service account token has been invalidated")
		}
		if podref.UID != string(pod.UID) {
			klog.V(4).Infof("Pod UID no longer matches %s/%s: %q != %q", namespace, podref.Name, string(pod.UID), podref.UID)
			return nil, fmt.Errorf("pod UID (%s) does not match service account pod ref claim (%s)", pod.UID, podref.UID)
		}
		if pod.DeletionTimestamp != nil && pod.DeletionTimestamp.Time.Before(invalidIfDeletedBefore) {
			klog.V(4).Infof("Bound pod is deleted and awaiting removal: %s/%s for service account %s/%s", namespace, podref.Name, namespace, saref.Name)
			return nil, errors.New("service account token has been invalidated")
		}
		podName = podref.Name
		podUID = podref.UID
	}

	var nodeName, nodeUID string
	if noderef != nil {
		switch {
		case podref != nil:
			if utilfeature.DefaultFeatureGate.Enabled(features.ServiceAccountTokenPodNodeInfo) {
				// for pod-bound tokens, just extract the node claims
				nodeName = noderef.Name
				nodeUID = noderef.UID
			}
		case podref == nil:
			if !utilfeature.DefaultFeatureGate.Enabled(features.ServiceAccountTokenNodeBindingValidation) {
				klog.V(4).Infof("ServiceAccount token is bound to a Node object, but the node bound token validation feature is disabled")
				return nil, fmt.Errorf("token is bound to a Node object but the %s feature gate is disabled", features.ServiceAccountTokenNodeBindingValidation)
			}

			node, err := v.getter.GetNode(noderef.Name)
			if err != nil {
				klog.V(4).Infof("Could not retrieve node object %q for service account %s/%s: %v", noderef.Name, namespace, saref.Name, err)
				return nil, errors.New("service account token has been invalidated")
			}
			if noderef.UID != string(node.UID) {
				klog.V(4).Infof("Node UID no longer matches %s: %q != %q", noderef.Name, string(node.UID), noderef.UID)
				return nil, fmt.Errorf("node UID (%s) does not match service account node ref claim (%s)", node.UID, noderef.UID)
			}
			if node.DeletionTimestamp != nil && node.DeletionTimestamp.Time.Before(invalidIfDeletedBefore) {
				klog.V(4).Infof("Node %q is deleted and awaiting removal for service account %s/%s", node.Name, namespace, saref.Name)
				return nil, errors.New("service account token has been invalidated")
			}
			nodeName = noderef.Name
			nodeUID = noderef.UID
		}
	}

	// Check special 'warnafter' field for projected service account token transition.
	warnafter := private.Kubernetes.WarnAfter
	if warnafter != nil && *warnafter != 0 {
		if nowTime.After(warnafter.Time()) {
			secondsAfterWarn := nowTime.Unix() - warnafter.Time().Unix()
			auditInfo := fmt.Sprintf("subject: %s, seconds after warning threshold: %d", public.Subject, secondsAfterWarn)
			audit.AddAuditAnnotation(ctx, "authentication.k8s.io/stale-token", auditInfo)
			staleTokensTotal.WithContext(ctx).Inc()
		} else {
			validTokensTotal.WithContext(ctx).Inc()
		}
	}

	var jti string
	if utilfeature.DefaultFeatureGate.Enabled(features.ServiceAccountTokenJTI) {
		jti = public.ID
	}
	return &apiserverserviceaccount.ServiceAccountInfo{
		Namespace:    private.Kubernetes.Namespace,
		Name:         private.Kubernetes.Svcacct.Name,
		UID:          private.Kubernetes.Svcacct.UID,
		PodName:      podName,
		PodUID:       podUID,
		NodeName:     nodeName,
		NodeUID:      nodeUID,
		CredentialID: apiserverserviceaccount.CredentialIDForJTI(jti),
	}, nil
}
