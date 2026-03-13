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
	"encoding/json"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	applyv1 "k8s.io/client-go/applyconfigurations/core/v1"
	clientinformers "k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	listersv1 "k8s.io/client-go/listers/core/v1"
	serviceaccountcontroller "k8s.io/kubernetes/pkg/controller/serviceaccount"
	"k8s.io/kubernetes/pkg/controlplane/controller/legacytokentracking"
	"k8s.io/kubernetes/pkg/serviceaccount"
	"k8s.io/utils/clock"
	testingclock "k8s.io/utils/clock/testing"
)

const (
	dateFormat    = "2006-01-02"
	cleanUpPeriod = 24 * time.Hour
	syncInterval  = 5 * time.Second
	pollTimeout   = 15 * time.Second
	pollInterval  = time.Second
)

func TestLegacyServiceAccountTokenCleanUp(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	c, config, stopFunc, informers, err := startServiceAccountTestServerAndWaitForCaches(ctx, t)
	defer stopFunc()
	if err != nil {
		t.Fatalf("failed to setup ServiceAccounts server: %v", err)
	}

	// wait configmap to be labeled with tracking date
	waitConfigmapToBeLabeled(ctx, t, c)

	tests := []struct {
		name               string
		secretName         string
		secretTokenData    string
		namespace          string
		expectCleanedUp    bool
		expectInvalidLabel bool
		lastUsedLabel      bool
		isPodMounted       bool
		isManual           bool
	}{
		{
			name:               "auto created legacy token without pod binding",
			secretName:         "auto-token-without-pod-mounting-a",
			namespace:          "clean-ns-1",
			lastUsedLabel:      true,
			isManual:           false,
			isPodMounted:       false,
			expectCleanedUp:    true,
			expectInvalidLabel: true,
		},
		{
			name:               "manually created legacy token",
			secretName:         "manual-token",
			namespace:          "clean-ns-2",
			lastUsedLabel:      true,
			isManual:           true,
			isPodMounted:       false,
			expectCleanedUp:    false,
			expectInvalidLabel: false,
		},
		{
			name:               "auto created legacy token with pod binding",
			secretName:         "auto-token-with-pod-mounting",
			namespace:          "clean-ns-3",
			lastUsedLabel:      true,
			isManual:           false,
			isPodMounted:       true,
			expectCleanedUp:    false,
			expectInvalidLabel: false,
		},
		{
			name:               "auto created legacy token without pod binding, secret has not been used after tracking",
			secretName:         "auto-token-without-pod-mounting-b",
			namespace:          "clean-ns-4",
			lastUsedLabel:      false,
			isManual:           false,
			isPodMounted:       false,
			expectCleanedUp:    true,
			expectInvalidLabel: true,
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
				createAutotokenMountedPod(ctx, t, c, test.namespace, test.secretName, podLister)
			}

			myConfig := *config
			wh := &warningHandler{}
			myConfig.WarningHandler = wh
			myConfig.BearerToken = string(string(secret.Data[v1.ServiceAccountTokenKey]))
			roClient := clientset.NewForConfigOrDie(&myConfig)

			// the secret should not be labeled with LastUsedLabelKey.
			checkLastUsedLabel(ctx, t, c, secret, false)

			if test.lastUsedLabel {
				doServiceAccountAPIReadRequest(ctx, t, roClient, test.namespace, true)

				// all service account tokens should be labeled with LastUsedLabelKey.
				checkLastUsedLabel(ctx, t, c, secret, true)
			}

			// Test invalid labels
			fakeClock.Step(cleanUpPeriod + 24*time.Hour)
			checkInvalidSinceLabel(ctx, t, c, secret, fakeClock, test.expectInvalidLabel)

			// Test invalid secret cannot be used
			if test.expectInvalidLabel {
				t.Logf("Check the invalid token cannot authenticate request.")
				doServiceAccountAPIReadRequest(ctx, t, roClient, test.namespace, false)

				// Check the secret has been labelded with the LastUsedLabelKey.
				if !test.lastUsedLabel {
					checkLastUsedLabel(ctx, t, c, secret, true)
				}

				// Update secret by removing the invalid since label
				removeInvalidLabel(ctx, c, t, secret)

				t.Logf("Check the token can authenticate request after patching the secret by removing the invalid label.")
				doServiceAccountAPIReadRequest(ctx, t, roClient, test.namespace, true)

				// Update the lastUsed label date to the fakeClock date (as the validation function uses the real time to label the lastUsed date)
				patchSecret(ctx, c, t, fakeClock.Now().UTC().Format(dateFormat), secret)

				// The secret will be marked as invalid again after time period duration cleanUpPeriod + 24*time.Hour
				fakeClock.Step(cleanUpPeriod + 24*time.Hour)
				checkInvalidSinceLabel(ctx, t, c, secret, fakeClock, true)
			}

			fakeClock.Step(cleanUpPeriod + 24*time.Hour)
			checkSecretCleanUp(ctx, t, c, secret, test.expectCleanedUp)
		})
	}
}

func waitConfigmapToBeLabeled(ctx context.Context, t *testing.T, c clientset.Interface) {
	if err := wait.PollUntilContextTimeout(ctx, pollInterval, pollTimeout, true, func(ctx context.Context) (bool, error) {
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
}

func checkSecretCleanUp(ctx context.Context, t *testing.T, c clientset.Interface, secret *v1.Secret, shouldCleanUp bool) {
	err := wait.PollUntilContextTimeout(ctx, pollInterval, pollTimeout, true, func(ctx context.Context) (bool, error) {
		_, err := c.CoreV1().Secrets(secret.Namespace).Get(context.TODO(), secret.Name, metav1.GetOptions{})
		if shouldCleanUp {
			if err == nil {
				return false, nil
			} else if !apierrors.IsNotFound(err) {
				t.Fatalf("Failed to get secret %s, err: %v", secret.Name, err)
			}
			return true, nil
		}
		if err != nil {
			if apierrors.IsNotFound(err) {
				t.Fatalf("The secret %s should not be cleaned up, err: %v", secret.Name, err)
			} else {
				t.Fatalf("Failed to get secret %s, err: %v", secret.Name, err)
			}
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("Failed to check the existence for secret: %s, shouldCleanUp: %v, error: %v", secret.Name, shouldCleanUp, err)
	}
}

func checkInvalidSinceLabel(ctx context.Context, t *testing.T, c clientset.Interface, secret *v1.Secret, fakeClock *testingclock.FakeClock, shouldLabel bool) {
	err := wait.PollUntilContextTimeout(ctx, pollInterval, pollTimeout, true, func(ctx context.Context) (bool, error) {
		liveSecret, err := c.CoreV1().Secrets(secret.Namespace).Get(context.TODO(), secret.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get secret: %s, err: %v", secret.Name, err)
		}
		invalidSince, ok := liveSecret.GetLabels()[serviceaccount.InvalidSinceLabelKey]
		if shouldLabel {
			if !ok || invalidSince != fakeClock.Now().UTC().Format(dateFormat) {
				return false, nil
			}
			return true, nil
		}
		if invalidSince != "" {
			return false, nil
		}
		return true, nil
	})

	if err != nil {
		t.Fatalf("Failed to check secret invalid since label for secret: %s, shouldLabel: %v, error: %v", secret.Name, shouldLabel, err)
	}
}

func checkLastUsedLabel(ctx context.Context, t *testing.T, c clientset.Interface, secret *v1.Secret, shouldLabel bool) {
	err := wait.PollUntilContextTimeout(ctx, pollInterval, pollTimeout, true, func(ctx context.Context) (bool, error) {
		liveSecret, err := c.CoreV1().Secrets(secret.Namespace).Get(ctx, secret.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get secret: %s, err: %v", secret.Name, err)
		}
		lastUsed, ok := liveSecret.GetLabels()[serviceaccount.LastUsedLabelKey]
		if shouldLabel {
			if !ok || lastUsed != time.Now().UTC().Format(dateFormat) {
				return false, nil
			}
			t.Logf("The secret %s has been labeled with %s", secret.Name, lastUsed)
			return true, nil
		}
		if ok {
			t.Fatalf("Secret %s should not have the lastUsed label", secret.Name)
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("Failed to check secret last used label for secret: %s, shouldLabel: %v, error: %v", secret.Name, shouldLabel, err)
	}
}

func removeInvalidLabel(ctx context.Context, c clientset.Interface, t *testing.T, secret *v1.Secret) {
	lastUsed := secret.GetLabels()[serviceaccount.LastUsedLabelKey]
	patchContent, err := json.Marshal(applyv1.Secret(secret.Name, secret.Namespace).WithLabels(map[string]string{serviceaccount.InvalidSinceLabelKey: "", serviceaccount.LastUsedLabelKey: lastUsed}))
	if err != nil {
		t.Fatalf("Failed to marshal invalid since label, err: %v", err)
	}
	t.Logf("Patch the secret by removing the invalid label.")
	if _, err := c.CoreV1().Secrets(secret.Namespace).Patch(ctx, secret.Name, types.MergePatchType, patchContent, metav1.PatchOptions{}); err != nil {
		t.Fatalf("Failed to remove invalid since label, err: %v", err)
	}
	err = wait.PollUntilContextTimeout(ctx, pollInterval, pollTimeout, true, func(ctx context.Context) (bool, error) {
		secret, err = c.CoreV1().Secrets(secret.Namespace).Get(context.TODO(), secret.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get secret: %s, err: %v", secret.Name, err)
		}
		invalidSince := secret.GetLabels()[serviceaccount.InvalidSinceLabelKey]
		if invalidSince != "" {
			t.Log("Patch has not completed.")
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("Failed to patch secret: %s, err: %v", secret.Name, err)
	}
}

func patchSecret(ctx context.Context, c clientset.Interface, t *testing.T, lastUsed string, secret *v1.Secret) {
	patchContent, err := json.Marshal(applyv1.Secret(secret.Name, secret.Namespace).WithUID(secret.UID).WithLabels(map[string]string{serviceaccount.InvalidSinceLabelKey: "", serviceaccount.LastUsedLabelKey: lastUsed}))
	if err != nil {
		t.Fatalf("Failed to marshal invalid since label, err: %v", err)
	}
	t.Logf("Patch the secret by removing the invalid label.")
	if _, err := c.CoreV1().Secrets(secret.Namespace).Patch(ctx, secret.Name, types.MergePatchType, patchContent, metav1.PatchOptions{}); err != nil {
		t.Fatalf("Failed to remove invalid since label, err: %v", err)
	}
	err = wait.PollUntilContextTimeout(ctx, pollInterval, pollTimeout, true, func(ctx context.Context) (bool, error) {
		secret, err = c.CoreV1().Secrets(secret.Namespace).Get(context.TODO(), secret.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get secret: %s, err: %v", secret.Name, err)
		}
		lastUsedString := secret.GetLabels()[serviceaccount.LastUsedLabelKey]
		if lastUsedString != lastUsed {
			t.Log("Patch has not completed.")
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("Failed to patch secret: %s, err: %v", secret.Name, err)
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

func doServiceAccountAPIReadRequest(ctx context.Context, t *testing.T, c clientset.Interface, ns string, authenticated bool) {
	readOps := []testOperation{
		func() error {
			_, err := c.CoreV1().Secrets(ns).List(context.TODO(), metav1.ListOptions{})
			return err
		},
		func() error {
			_, err := c.CoreV1().Pods(ns).List(context.TODO(), metav1.ListOptions{})
			return err
		},
	}

	for _, op := range readOps {
		err := wait.PollUntilContextTimeout(ctx, pollInterval, pollTimeout, true, func(ctx context.Context) (bool, error) {
			err := op()
			if authenticated && err != nil || !authenticated && err == nil {
				return false, nil
			}
			return true, nil
		})
		if err != nil {
			t.Fatalf("Failed to check secret token authentication: error: %v", err)
		}
	}
}

func createAutotokenMountedPod(ctx context.Context, t *testing.T, c clientset.Interface, ns, secretName string, podLister listersv1.PodLister) *v1.Pod {
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
		t.Fatalf("Failed to create pod with token (%s:%s) bound, err: %v", ns, secretName, err)
	}
	err = wait.PollUntilContextTimeout(ctx, pollInterval, pollTimeout, true, func(ctx context.Context) (bool, error) {
		pod, err = podLister.Pods(ns).Get("token-bound-pod")
		if err != nil {
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("Failed to wait auto-token mounted pod: err: %v", err)
	}
	return pod
}
