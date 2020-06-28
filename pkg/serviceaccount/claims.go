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

	"gopkg.in/square/go-jose.v2/jwt"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/klog/v2"

	apiserverserviceaccount "k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/kubernetes/pkg/apis/core"
)

const (
	// Injected bound service account token expiration which triggers monitoring of its time-bound feature.
	WarnOnlyBoundTokenExpirationSeconds = 60*60 + 7

	// Extended expiration for those modifed tokens involved in safe rollout if time-bound feature.
	ExpirationExtensionSeconds = 24 * 365 * 60 * 60
)

// time.Now stubbed out to allow testing
var now = time.Now

type privateClaims struct {
	Kubernetes kubernetes `json:"kubernetes.io,omitempty"`
}

type kubernetes struct {
	Namespace string          `json:"namespace,omitempty"`
	Svcacct   ref             `json:"serviceaccount,omitempty"`
	Pod       *ref            `json:"pod,omitempty"`
	Secret    *ref            `json:"secret,omitempty"`
	WarnAfter jwt.NumericDate `json:"warnafter,omitempty"`
}

type ref struct {
	Name string `json:"name,omitempty"`
	UID  string `json:"uid,omitempty"`
}

func Claims(sa core.ServiceAccount, pod *core.Pod, secret *core.Secret, expirationSeconds, warnafter int64, audience []string) (*jwt.Claims, interface{}) {
	now := now()
	sc := &jwt.Claims{
		Subject:   apiserverserviceaccount.MakeUsername(sa.Namespace, sa.Name),
		Audience:  jwt.Audience(audience),
		IssuedAt:  jwt.NewNumericDate(now),
		NotBefore: jwt.NewNumericDate(now),
		Expiry:    jwt.NewNumericDate(now.Add(time.Duration(expirationSeconds) * time.Second)),
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
	switch {
	case pod != nil:
		pc.Kubernetes.Pod = &ref{
			Name: pod.Name,
			UID:  string(pod.UID),
		}
	case secret != nil:
		pc.Kubernetes.Secret = &ref{
			Name: secret.Name,
			UID:  string(secret.UID),
		}
	}

	if warnafter != 0 {
		pc.Kubernetes.WarnAfter = jwt.NewNumericDate(now.Add(time.Duration(warnafter) * time.Second))
	}

	return sc, pc
}

func NewValidator(getter ServiceAccountTokenGetter) Validator {
	return &validator{
		getter: getter,
	}
}

type validator struct {
	getter ServiceAccountTokenGetter
}

var _ = Validator(&validator{})

func (v *validator) Validate(ctx context.Context, _ string, public *jwt.Claims, privateObj interface{}) (*ServiceAccountInfo, error) {
	private, ok := privateObj.(*privateClaims)
	if !ok {
		klog.Errorf("jwt validator expected private claim of type *privateClaims but got: %T", privateObj)
		return nil, errors.New("Token could not be validated.")
	}
	nowTime := now()
	err := public.Validate(jwt.Expected{
		Time: nowTime,
	})
	switch {
	case err == nil:
	case err == jwt.ErrExpired:
		return nil, errors.New("Token has expired.")
	default:
		klog.Errorf("unexpected validation error: %T", err)
		return nil, errors.New("Token could not be validated.")
	}

	// consider things deleted prior to now()-leeway to be invalid
	invalidIfDeletedBefore := nowTime.Add(-jwt.DefaultLeeway)
	namespace := private.Kubernetes.Namespace
	saref := private.Kubernetes.Svcacct
	podref := private.Kubernetes.Pod
	secref := private.Kubernetes.Secret
	// Make sure service account still exists (name and UID)
	serviceAccount, err := v.getter.GetServiceAccount(namespace, saref.Name)
	if err != nil {
		klog.V(4).Infof("Could not retrieve service account %s/%s: %v", namespace, saref.Name, err)
		return nil, err
	}
	if serviceAccount.DeletionTimestamp != nil && serviceAccount.DeletionTimestamp.Time.Before(invalidIfDeletedBefore) {
		klog.V(4).Infof("Service account has been deleted %s/%s", namespace, saref.Name)
		return nil, fmt.Errorf("ServiceAccount %s/%s has been deleted", namespace, saref.Name)
	}
	if string(serviceAccount.UID) != saref.UID {
		klog.V(4).Infof("Service account UID no longer matches %s/%s: %q != %q", namespace, saref.Name, string(serviceAccount.UID), saref.UID)
		return nil, fmt.Errorf("ServiceAccount UID (%s) does not match claim (%s)", serviceAccount.UID, saref.UID)
	}

	if secref != nil {
		// Make sure token hasn't been invalidated by deletion of the secret
		secret, err := v.getter.GetSecret(namespace, secref.Name)
		if err != nil {
			klog.V(4).Infof("Could not retrieve bound secret %s/%s for service account %s/%s: %v", namespace, secref.Name, namespace, saref.Name, err)
			return nil, errors.New("Token has been invalidated")
		}
		if secret.DeletionTimestamp != nil && secret.DeletionTimestamp.Time.Before(invalidIfDeletedBefore) {
			klog.V(4).Infof("Bound secret is deleted and awaiting removal: %s/%s for service account %s/%s", namespace, secref.Name, namespace, saref.Name)
			return nil, errors.New("Token has been invalidated")
		}
		if secref.UID != string(secret.UID) {
			klog.V(4).Infof("Secret UID no longer matches %s/%s: %q != %q", namespace, secref.Name, string(secret.UID), secref.UID)
			return nil, fmt.Errorf("Secret UID (%s) does not match claim (%s)", secret.UID, secref.UID)
		}
	}

	var podName, podUID string
	if podref != nil {
		// Make sure token hasn't been invalidated by deletion of the pod
		pod, err := v.getter.GetPod(namespace, podref.Name)
		if err != nil {
			klog.V(4).Infof("Could not retrieve bound pod %s/%s for service account %s/%s: %v", namespace, podref.Name, namespace, saref.Name, err)
			return nil, errors.New("Token has been invalidated")
		}
		if pod.DeletionTimestamp != nil && pod.DeletionTimestamp.Time.Before(invalidIfDeletedBefore) {
			klog.V(4).Infof("Bound pod is deleted and awaiting removal: %s/%s for service account %s/%s", namespace, podref.Name, namespace, saref.Name)
			return nil, errors.New("Token has been invalidated")
		}
		if podref.UID != string(pod.UID) {
			klog.V(4).Infof("Pod UID no longer matches %s/%s: %q != %q", namespace, podref.Name, string(pod.UID), podref.UID)
			return nil, fmt.Errorf("Pod UID (%s) does not match claim (%s)", pod.UID, podref.UID)
		}
		podName = podref.Name
		podUID = podref.UID
	}

	// Check special 'warnafter' field for projected service account token transition.
	warnafter := private.Kubernetes.WarnAfter
	if warnafter != 0 {
		if nowTime.After(warnafter.Time()) {
			secondsAfterWarn := nowTime.Unix() - warnafter.Time().Unix()
			auditInfo := fmt.Sprintf("subject: %s, seconds after warning threshold: %d", public.Subject, secondsAfterWarn)
			audit.AddAuditAnnotation(ctx, "authentication.k8s.io/stale-token", auditInfo)
			staleTokensTotal.Inc()
		} else {
			validTokensTotal.Inc()
		}
	}

	return &ServiceAccountInfo{
		Namespace: private.Kubernetes.Namespace,
		Name:      private.Kubernetes.Svcacct.Name,
		UID:       private.Kubernetes.Svcacct.UID,
		PodName:   podName,
		PodUID:    podUID,
	}, nil
}

func (v *validator) NewPrivateClaims() interface{} {
	return &privateClaims{}
}
