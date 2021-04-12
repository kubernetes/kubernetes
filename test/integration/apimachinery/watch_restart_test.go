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
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/kubernetes/test/integration/framework"
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

	// Set up a master
	masterConfig := framework.NewIntegrationTestMasterConfig()
	// Timeout is set random between MinRequestTimeout and 2x
	masterConfig.GenericConfig.MinRequestTimeout = int(timeout.Seconds()) / 4
	_, s, closeFn := framework.RunAMaster(masterConfig)
	defer closeFn()

	config := &restclient.Config{
		Host: s.URL,
	}

	namespaceObject := framework.CreateTestingNamespace("retry-watch", s, t)
	defer framework.DeleteTestingNamespace(namespaceObject, s, t)

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
					t.Fatalf("Failed to patch secret: %v", err)
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
		getWatcher          func(c *kubernetes.Clientset, secret *corev1.Secret) (watch.Interface, error, func())
		normalizeOutputFunc func(referenceOutput []string) []string
	}{
		{
			name:    "regular watcher should fail",
			succeed: false,
			secret:  newTestSecret("secret-01"),
			getWatcher: func(c *kubernetes.Clientset, secret *corev1.Secret) (watch.Interface, error, func()) {
				options := metav1.ListOptions{
					ResourceVersion: secret.ResourceVersion,
				}
				w, err := getWatchFunc(c, secret)(options)
				return w, err, noop
			}, // regular watcher; unfortunately destined to fail
			normalizeOutputFunc: noopNormalization,
		},
		{
			name:    "RetryWatcher survives closed watches",
			succeed: true,
			secret:  newTestSecret("secret-02"),
			getWatcher: func(c *kubernetes.Clientset, secret *corev1.Secret) (watch.Interface, error, func()) {
				lw := &cache.ListWatch{
					WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
						return getWatchFunc(c, secret)(options)
					},
				}
				w, err := watchtools.NewRetryWatcher(secret.ResourceVersion, lw)
				return w, err, func() { <-w.Done() }
			},
			normalizeOutputFunc: noopNormalization,
		},
		{
			name:    "InformerWatcher survives closed watches",
			succeed: true,
			secret:  newTestSecret("secret-03"),
			getWatcher: func(c *kubernetes.Clientset, secret *corev1.Secret) (watch.Interface, error, func()) {
				lw := &cache.ListWatch{
					ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
						return getListFunc(c, secret)(options), nil
					},
					WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
						return getWatchFunc(c, secret)(options)
					},
				}
				_, _, w, done := watchtools.NewIndexerInformerWatcher(lw, &corev1.Secret{})
				return w, nil, func() { <-done }
			},
			normalizeOutputFunc: normalizeInformerOutputFunc(initialCount),
		},
	}

	t.Run("group", func(t *testing.T) {
		for _, tmptc := range tt {
			tc := tmptc // we need to copy it for parallel runs
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				c, err := kubernetes.NewForConfig(config)
				if err != nil {
					t.Fatalf("Failed to create clientset: %v", err)
				}

				secret, err := c.CoreV1().Secrets(tc.secret.Namespace).Create(context.TODO(), tc.secret, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Failed to create testing secret %s/%s: %v", tc.secret.Namespace, tc.secret.Name, err)
				}

				watcher, err, doneFn := tc.getWatcher(c, secret)
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
				ctx, cancel := watchtools.ContextWithOptionalTimeout(context.Background(), timeout)
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

// TestWatchRestartsWhenTimeoutSet makes sure that the global timeout is respected for watch requests.
func TestWatchRestartsWhenTimeoutSet(t *testing.T) {
	// set up test environment
	masterConfig := framework.NewIntegrationTestMasterConfig()
	_, s, closeFn := framework.RunAMaster(masterConfig)
	defer closeFn()

	testNamespace := framework.CreateTestingNamespace("watch-restart-timeout", s, t)
	defer framework.DeleteTestingNamespace(testNamespace, s, t)

	config := &restclient.Config{
		Host:    s.URL,
		Timeout: 12 * time.Second,
	}

	client, err := kubernetes.NewForConfig(config)
	if err != nil {
		t.Fatalf("Failed to create clientset: %v", err)
	}

	_, err = client.CoreV1().Secrets(testNamespace.Name).Create(context.TODO(), &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "secret-01",
			Namespace: testNamespace.Name,
		},
		Data: map[string][]byte{
			"data": []byte("value1\n"),
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create secret %s/secret-01: %v", testNamespace.Name, err)
	}

	// act
	probes := 3
	clientGoTimeout := config.Timeout + time.Second
	res := []time.Duration{}
	for i := 0; i < probes; i++ {
		// record current time to be able to asses if the timeout has been reached
		startTime := time.Now()
		w, err := client.CoreV1().Secrets(testNamespace.Namespace).Watch(context.TODO(), metav1.ListOptions{})
		if err != nil {
			t.Fatal(err)
		}

		for {
			_, ok := <-w.ResultChan()
			if !ok {
				// watch closed
				break
			}
		}

		// record watch requests than ended later than we expected
		watchDuration := time.Since(startTime)
		if watchDuration >= clientGoTimeout {
			res = append(res, watchDuration)
		}
	}

	// validate
	if len(res) > 0 {
		t.Fatalf("we expected exactly %d watch reqeusts to end within %v"+
			" %d probes ended later than we expected %v", probes, clientGoTimeout, len(res), res)
	}
}

// TestWatchTimeoutIsRespectedWhenOverwritten makes sure that setting a new timeout via ListOptions overwrites the global timeout set on the rest.Config.
// It turned out that setting a timeout on HTTP client affected watch requests.
// For example, with a 10 second timeout watch requests ere being re-established exactly after 10 seconds even though the default request timeout for them is ~5 minutes.
//
// This is because if multiple timeouts were set, the stdlib picks the smaller timeout to be applied, leaving other useless.
// For more details see https://github.com/golang/go/blob/a937729c2c2f6950a32bc5cd0f5b88700882f078/src/net/http/client.go#L364
func TestWatchTimeoutIsRespectedWhenOverwritten(t *testing.T) {
	// set up test environment
	masterConfig := framework.NewIntegrationTestMasterConfig()
	_, s, closeFn := framework.RunAMaster(masterConfig)
	defer closeFn()

	testNamespace := framework.CreateTestingNamespace("watch-restart-timeout", s, t)
	defer framework.DeleteTestingNamespace(testNamespace, s, t)

	config := &restclient.Config{
		Host:    s.URL,
		Timeout: 10 * time.Second,
	}

	client, err := kubernetes.NewForConfig(config)
	if err != nil {
		t.Fatalf("Failed to create clientset: %v", err)
	}

	_, err = client.CoreV1().Secrets(testNamespace.Name).Create(context.TODO(), &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "secret-01",
			Namespace: testNamespace.Name,
		},
		Data: map[string][]byte{
			"data": []byte("value1\n"),
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create secret %s/secret-01: %v", testNamespace.Name, err)
	}

	// act
	probes := 6
	failedProbesTreshold := 2 // accounts for 33% of watch requests
	clientGoTimeout := config.Timeout + time.Second
	res := []time.Duration{}
	for i := 0; i < probes; i++ {
		// contextTimeout sets the upper bound time limit for our watch requests it must be > than clientGoTimeout
		watchTimeoutSeconds := int64(15)

		// record current time to be able to asses if the timeout has been reached
		startTime := time.Now()
		w, err := client.CoreV1().Secrets(testNamespace.Namespace).Watch(context.TODO(), metav1.ListOptions{TimeoutSeconds: &watchTimeoutSeconds})
		if err != nil {
			t.Fatal(err)
		}

		for {
			_, ok := <-w.ResultChan()
			if !ok {
				// watch closed
				break
			}
		}

		// record watch requests than ended sooner than we expected
		watchDuration := time.Since(startTime)
		if watchDuration <= clientGoTimeout+time.Second {
			res = append(res, watchDuration)
		}
	}

	// validate
	if len(res) > failedProbesTreshold {
		t.Fatalf("%d%% of probes failed. That means some (all) watch requests ended sooner than we expected. The current treshold was set to %d%%"+
			" We did %d probes in total. The upper-bound time limit for a single watch reqeust was set to %v."+
			" %v probes ended before that time %v", len(res)*100/probes, failedProbesTreshold*100/probes, probes, clientGoTimeout, len(res), res)
	}
}
