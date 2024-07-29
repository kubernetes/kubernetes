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

package apimachinery

import (
	"context"
	"fmt"
	"reflect"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func noopNormalization(output []string) []string {
	return output
}

func normalizeInformerOutputFunc(initialVal string) func(output []string) []string {
	return func(output []string) []string {
		result := make([]string, 0, len(output))

		// Removes initial value and all of its direct repetitions
		lastVal := initialVal
		for _, v := range output {
			// Make values unique as informer(List+Watch) duplicates some events
			if v == lastVal {
				continue
			}
			result = append(result, v)
			lastVal = v
		}

		return result
	}
}

func noop() {}

func TestWatchRestartsIfTimeoutNotReached(t *testing.T) {
	// Has to be longer than 5 seconds
	timeout := 30 * time.Second

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--min-request-timeout=7"}, framework.SharedEtcd())
	defer server.TearDownFn()

	clientset, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	namespaceObject := framework.CreateNamespaceOrDie(clientset, "retry-watch", t)
	defer framework.DeleteNamespaceOrDie(clientset, namespaceObject, t)

	getListFunc := func(c *kubernetes.Clientset, secret *corev1.Secret) func(options metav1.ListOptions) *corev1.SecretList {
		return func(options metav1.ListOptions) *corev1.SecretList {
			options.FieldSelector = fields.OneTermEqualSelector("metadata.name", secret.Name).String()
			res, err := c.CoreV1().Secrets(secret.Namespace).List(context.TODO(), options)
			if err != nil {
				t.Fatalf("Failed to list Secrets: %v", err)
			}
			return res
		}
	}

	getWatchFunc := func(c *kubernetes.Clientset, secret *corev1.Secret) func(options metav1.ListOptions) (watch.Interface, error) {
		return func(options metav1.ListOptions) (watch.Interface, error) {
			options.FieldSelector = fields.OneTermEqualSelector("metadata.name", secret.Name).String()
			res, err := c.CoreV1().Secrets(secret.Namespace).Watch(context.TODO(), options)
			if err != nil {
				t.Fatalf("Failed to create a watcher on Secrets: %v", err)
			}
			return res, err
		}
	}

	generateEvents := func(t *testing.T, c *kubernetes.Clientset, secret *corev1.Secret, referenceOutput *[]string, stopChan chan struct{}, stoppedChan chan struct{}) {
		defer close(stoppedChan)
		counter := 0

		// These 5 seconds are here to protect against a race at the end when we could write something there at the same time as watch.Until ends
		softTimeout := timeout - 5*time.Second
		if softTimeout < 0 {
			panic("Timeout has to be grater than 5 seconds!")
		}
		endChannel := time.After(softTimeout)
		for {
			select {
			// TODO: get this lower once we figure out how to extend ETCD cache
			case <-time.After(1000 * time.Millisecond):
				counter = counter + 1

				patch := fmt.Sprintf(`{"metadata": {"annotations": {"count": "%d"}}}`, counter)
				_, err := c.CoreV1().Secrets(secret.Namespace).Patch(context.TODO(), secret.Name, types.StrategicMergePatchType, []byte(patch), metav1.PatchOptions{})
				if err != nil {
					t.Errorf("Failed to patch secret: %v", err)
					return
				}

				*referenceOutput = append(*referenceOutput, fmt.Sprintf("%d", counter))
			case <-endChannel:
				return
			case <-stopChan:
				return
			}
		}
	}

	initialCount := "0"
	newTestSecret := func(name string) *corev1.Secret {
		return &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: namespaceObject.Name,
				Annotations: map[string]string{
					"count": initialCount,
				},
			},
			Data: map[string][]byte{
				"data": []byte("value1\n"),
			},
		}
	}

	tt := []struct {
		name                string
		succeed             bool
		secret              *corev1.Secret
		getWatcher          func(ctx context.Context, c *kubernetes.Clientset, secret *corev1.Secret) (watch.Interface, func(), error)
		normalizeOutputFunc func(referenceOutput []string) []string
	}{
		{
			name:    "regular watcher should fail",
			succeed: false,
			secret:  newTestSecret("secret-01"),
			getWatcher: func(ctx context.Context, c *kubernetes.Clientset, secret *corev1.Secret) (watch.Interface, func(), error) {
				options := metav1.ListOptions{
					ResourceVersion: secret.ResourceVersion,
				}
				w, err := getWatchFunc(c, secret)(options)
				return w, noop, err
			}, // regular watcher; unfortunately destined to fail
			normalizeOutputFunc: noopNormalization,
		},
		{
			name:    "RetryWatcher survives closed watches",
			succeed: true,
			secret:  newTestSecret("secret-02"),
			getWatcher: func(ctx context.Context, c *kubernetes.Clientset, secret *corev1.Secret) (watch.Interface, func(), error) {
				lw := &cache.ListWatch{
					WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
						return getWatchFunc(c, secret)(options)
					},
				}
				w, err := watchtools.NewRetryWatcher(secret.ResourceVersion, lw)
				return w, func() { <-w.Done() }, err
			},
			normalizeOutputFunc: noopNormalization,
		},
		{
			name:    "InformerWatcher survives closed watches",
			succeed: true,
			secret:  newTestSecret("secret-03"),
			getWatcher: func(ctx context.Context, c *kubernetes.Clientset, secret *corev1.Secret) (watch.Interface, func(), error) {
				lw := &cache.ListWatch{
					ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
						return getListFunc(c, secret)(options), nil
					},
					WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
						return getWatchFunc(c, secret)(options)
					},
				}
				// there is an inherent race between a producer (generateEvents) and a consumer (the watcher) that needs to be solved here
				// since the watcher is driven by an informer it is crucial to start producing only after the informer has synced
				// otherwise we might not get all expected events since the informer LIST (or watchelist) and only then WATCHES
				// all events received during the initial LIST (or watchlist) will be seen as a single event (to most recent version of an obj)
				_, informer, w, done := watchtools.NewIndexerInformerWatcherWithContext(ctx, lw, &corev1.Secret{})
				cache.WaitForCacheSync(ctx.Done(), informer.HasSynced)
				return w, func() { <-done }, nil
			},
			normalizeOutputFunc: normalizeInformerOutputFunc(initialCount),
		},
	}

	t.Run("group", func(t *testing.T) {
		for _, tmptc := range tt {
			tc := tmptc // we need to copy it for parallel runs
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				tCtx := ktesting.Init(t)
				c, err := kubernetes.NewForConfig(server.ClientConfig)
				if err != nil {
					t.Fatalf("Failed to create clientset: %v", err)
				}

				secret, err := c.CoreV1().Secrets(tc.secret.Namespace).Create(tCtx, tc.secret, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Failed to create testing secret %s/%s: %v", tc.secret.Namespace, tc.secret.Name, err)
				}

				watcher, doneFn, err := tc.getWatcher(tCtx, c, secret)
				if err != nil {
					t.Fatalf("Failed to create watcher: %v", err)
				}
				defer doneFn()

				var referenceOutput []string
				var output []string
				stopChan := make(chan struct{})
				stoppedChan := make(chan struct{})
				go generateEvents(t, c, secret, &referenceOutput, stopChan, stoppedChan)

				// Record current time to be able to asses if the timeout has been reached
				startTime := time.Now()
				ctx, cancel := watchtools.ContextWithOptionalTimeout(tCtx, timeout)
				defer cancel()
				_, err = watchtools.UntilWithoutRetry(ctx, watcher, func(event watch.Event) (bool, error) {
					s, ok := event.Object.(*corev1.Secret)
					if !ok {
						t.Fatalf("Received an object that is not a Secret: %#v", event.Object)
					}
					output = append(output, s.Annotations["count"])
					// Watch will never end voluntarily
					return false, nil
				})
				watchDuration := time.Since(startTime)
				close(stopChan)
				<-stoppedChan

				output = tc.normalizeOutputFunc(output)

				t.Logf("Watch duration: %v; timeout: %v", watchDuration, timeout)

				if err == nil && !tc.succeed {
					t.Fatalf("Watch should have timed out but it exited without an error!")
				}

				if err != wait.ErrWaitTimeout && tc.succeed {
					t.Fatalf("Watch exited with error: %v!", err)
				}

				if watchDuration < timeout && tc.succeed {
					t.Fatalf("Watch should have timed out after %v but it timed out prematurely after %v!", timeout, watchDuration)
				}

				if watchDuration >= timeout && !tc.succeed {
					t.Fatalf("Watch should have timed out but it succeeded!")
				}

				if tc.succeed && !reflect.DeepEqual(referenceOutput, output) {
					t.Fatalf("Reference and real output differ! We must have lost some events or read some multiple times!\nRef:  %#v\nReal: %#v", referenceOutput, output)
				}
			})
		}
	})
}
