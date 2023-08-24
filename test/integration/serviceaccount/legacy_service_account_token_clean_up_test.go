/*
Copyright 2023 The Kubernetes Authors.

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

// This file tests the legacy service account token cleaning-up.

import (
	"context"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientinformers "k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	listersv1 "k8s.io/client-go/listers/core/v1"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	serviceaccountcontroller "k8s.io/kubernetes/pkg/controller/serviceaccount"
	"k8s.io/kubernetes/pkg/controlplane/controller/legacytokentracking"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/serviceaccount"
	"k8s.io/utils/clock"
	testingclock "k8s.io/utils/clock/testing"
)

const (
	dateFormat    = "2006-01-02"
	cleanUpPeriod = 24 * time.Hour
	syncInterval  = 1 * time.Second
)

func TestLegacyServiceAccountTokenCleanUp(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, kubefeatures.LegacyServiceAccountTokenCleanUp, true)()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	c, config, stopFunc, informers, err := startServiceAccountTestServerAndWaitForCaches(ctx, t)
	defer stopFunc()
	if err != nil {
		t.Fatalf("failed to setup ServiceAccounts server: %v", err)
	}

	// wait configmap to label with tracking date
	if err := wait.PollImmediate(time.Second, 10*time.Second, func() (bool, error) {
		configMap, err := c.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(ctx, legacytokentracking.ConfigMapName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		_, exist := configMap.Data[legacytokentracking.ConfigMapDataKey]
		if !exist {
			return false, fmt.Errorf("configMap does not have since label")
		}
		return true, nil
	}); err != nil {
		t.Fatalf("failed to wait configmap starts to track: %v", err)
	}

	tests := []struct {
		name            string
		secretName      string
		secretTokenData string
		namespace       string
		expectCleanedUp bool
		lastUsedLabel   bool
		isPodMounted    bool
		isManual        bool
	}{
		{
			name:            "auto created legacy token without pod binding",
			secretName:      "auto-token-without-pod-mounting-a",
			namespace:       "clean-ns-1",
			lastUsedLabel:   true,
			isManual:        false,
			isPodMounted:    false,
			expectCleanedUp: true,
		},
		{
			name:            "manually created legacy token",
			secretName:      "manual-token",
			namespace:       "clean-ns-2",
			lastUsedLabel:   true,
			isManual:        true,
			isPodMounted:    false,
			expectCleanedUp: false,
		},
		{
			name:            "auto created legacy token with pod binding",
			secretName:      "auto-token-with-pod-mounting",
			namespace:       "clean-ns-3",
			lastUsedLabel:   true,
			isManual:        false,
			isPodMounted:    true,
			expectCleanedUp: false,
		},
		{
			name:            "auto created legacy token without pod binding, secret has not been used after tracking",
			secretName:      "auto-token-without-pod-mounting-b",
			namespace:       "clean-ns-4",
			lastUsedLabel:   false,
			isManual:        false,
			isPodMounted:    false,
			expectCleanedUp: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {

			fakeClock := testingclock.NewFakeClock(time.Now().UTC())

			// start legacy service account token cleaner
			ctxForCleaner, cancelFunc := context.WithCancel(context.Background())
			startLegacyServiceAccountTokenCleaner(ctxForCleaner, c, fakeClock, informers)
			informers.Start(ctx.Done())
			defer cancelFunc()

			// create service account
			_, err = c.CoreV1().Namespaces().Create(context.TODO(), &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: test.namespace}}, metav1.CreateOptions{})
			if err != nil && !apierrors.IsAlreadyExists(err) {
				t.Fatalf("could not create namespace: %v", err)
			}
			mysa, err := c.CoreV1().ServiceAccounts(test.namespace).Create(context.TODO(), &v1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: readOnlyServiceAccountName}}, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Service Account not created: %v", err)
			}

			// create secret
			secret, err := createServiceAccountToken(c, mysa, test.namespace, test.secretName)
			if err != nil {
				t.Fatalf("Secret not created: %v", err)
			}
			if !test.isManual {
				if err := addReferencedServiceAccountToken(c, test.namespace, readOnlyServiceAccountName, secret); err != nil {
					t.Fatal(err)
				}
			}
			podLister := informers.Core().V1().Pods().Lister()
			if test.isPodMounted {
				_, err = createAutotokenMountedPod(c, test.namespace, test.secretName, podLister)
				if err != nil {
					t.Fatalf("Pod not created: %v", err)
				}
			}

			myConfig := *config
			wh := &warningHandler{}
			myConfig.WarningHandler = wh
			myConfig.BearerToken = string(string(secret.Data[v1.ServiceAccountTokenKey]))
			roClient := clientset.NewForConfigOrDie(&myConfig)

			// the secret should not be labeled with LastUsedLabelKey.
			liveSecret, err := c.CoreV1().Secrets(test.namespace).Get(context.TODO(), test.secretName, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("Could not get secret: %v", err)
			}
			_, ok := liveSecret.GetLabels()[serviceaccount.LastUsedLabelKey]
			if ok {
				t.Fatalf("Secret %s should not have the lastUsed label", test.secretName)
			}

			// authenticate legacy tokens
			if test.lastUsedLabel {
				doServiceAccountAPIRequests(t, roClient, test.namespace, true, true, false)
				// all service account tokens should be labeled with LastUsedLabelKey.
				liveSecret, err = c.CoreV1().Secrets(test.namespace).Get(context.TODO(), test.secretName, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("Could not get secret: %v", err)
				}
				lastUsed, ok := liveSecret.GetLabels()[serviceaccount.LastUsedLabelKey]
				if !ok {
					t.Fatalf("The secret %s should be labeled lastUsed time: %s", test.secretName, lastUsed)
				} else {
					t.Logf("The secret %s has been labeled with %s", test.secretName, lastUsed)
				}
			}

			fakeClock.Step(cleanUpPeriod + 24*time.Hour)
			time.Sleep(2 * syncInterval)
			liveSecret, err = c.CoreV1().Secrets(test.namespace).Get(context.TODO(), test.secretName, metav1.GetOptions{})
			if test.expectCleanedUp {
				if err == nil {
					t.Fatalf("The secret %s should be cleaned up. time: %v; creationTime: %v", test.secretName, fakeClock.Now().UTC(), liveSecret.CreationTimestamp)
				} else if !apierrors.IsNotFound(err) {
					t.Fatalf("Failed to get secret %s, err: %v", test.secretName, err)
				}
			} else if err != nil {
				if apierrors.IsNotFound(err) {
					t.Fatalf("The secret %s should not be cleaned up, err: %v", test.secretName, err)
				} else {
					t.Fatalf("Failed to get secret %s, err: %v", test.secretName, err)
				}
			}
		})
	}
}

func startLegacyServiceAccountTokenCleaner(ctx context.Context, client clientset.Interface, fakeClock clock.Clock, informers clientinformers.SharedInformerFactory) {
	legacySATokenCleaner, _ := serviceaccountcontroller.NewLegacySATokenCleaner(
		informers.Core().V1().ServiceAccounts(),
		informers.Core().V1().Secrets(),
		informers.Core().V1().Pods(),
		client,
		fakeClock,
		serviceaccountcontroller.LegacySATokenCleanerOptions{
			SyncInterval:  syncInterval,
			CleanUpPeriod: cleanUpPeriod,
		})
	go legacySATokenCleaner.Run(ctx)
}

func createAutotokenMountedPod(c clientset.Interface, ns, secretName string, podLister listersv1.PodLister) (*v1.Pod, error) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "token-bound-pod",
			Namespace: ns,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{Name: "name", Image: "image"},
			},
			Volumes: []v1.Volume{{Name: "foo", VolumeSource: v1.VolumeSource{Secret: &v1.SecretVolumeSource{SecretName: secretName}}}},
		},
	}
	pod, err := c.CoreV1().Pods(ns).Create(context.TODO(), pod, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to create pod with token (%s:%s) bound, err: %v", ns, secretName, err)
	}
	err = wait.PollImmediate(time.Second, 10*time.Second, func() (bool, error) {
		pod, err = podLister.Pods(ns).Get("token-bound-pod")
		if err != nil {
			return false, fmt.Errorf("failed to get pod with token (%s:%s) bound, err: %v", ns, secretName, err)
		}
		return true, nil
	})
	return pod, nil
}
