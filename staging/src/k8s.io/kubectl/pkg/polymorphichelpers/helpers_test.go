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

package polymorphichelpers

import (
	"fmt"
	"sort"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	fakeexternal "k8s.io/client-go/kubernetes/fake"
	testcore "k8s.io/client-go/testing"
	"k8s.io/kubectl/pkg/util/podutils"
)

func TestGetPodList(t *testing.T) {
	labelSet := map[string]string{"test": "selector"}
	tests := []struct {
		name string

		podList  *corev1.PodList
		watching []watch.Event
		sortBy   func([]*corev1.Pod) sort.Interface

		expected    *corev1.PodList
		expectedNum int
		expectedErr bool
	}{
		{
			name:    "kubectl logs - two ready pods",
			podList: newPodList(2, -1, -1, labelSet),
			sortBy:  func(pods []*corev1.Pod) sort.Interface { return podutils.ByLogging(pods) },
			expected: &corev1.PodList{
				Items: []corev1.Pod{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "pod-1",
							Namespace:         metav1.NamespaceDefault,
							CreationTimestamp: metav1.Date(2016, time.April, 1, 1, 0, 0, 0, time.UTC),
							Labels:            map[string]string{"test": "selector"},
						},
						Status: corev1.PodStatus{
							Conditions: []corev1.PodCondition{
								{
									Status: corev1.ConditionTrue,
									Type:   corev1.PodReady,
								},
							},
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "pod-2",
							Namespace:         metav1.NamespaceDefault,
							CreationTimestamp: metav1.Date(2016, time.April, 1, 1, 0, 1, 0, time.UTC),
							Labels:            map[string]string{"test": "selector"},
						},
						Status: corev1.PodStatus{
							Conditions: []corev1.PodCondition{
								{
									Status: corev1.ConditionTrue,
									Type:   corev1.PodReady,
								},
							},
						},
					},
				},
			},
			expectedNum: 2,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fake := fakeexternal.NewSimpleClientset(tc.podList)
			var watcher *watch.ProxyWatcher
			if len(tc.watching) > 0 {
				resultCh := make(chan watch.Event)
				watcher = watch.NewProxyWatcher(resultCh)
				go sendWatchEvents(resultCh, watcher.StopChan(), tc.watching)
				fake.PrependWatchReactor("pods", testcore.DefaultWatchReactor(watcher, nil))
			}
			selector := labels.Set(labelSet).AsSelector()
			podList, err := GetPodList(fake.CoreV1(), metav1.NamespaceDefault, selector.String(), 1*time.Minute, tc.sortBy)

			if !tc.expectedErr && err != nil {
				t.Fatalf("%s: unexpected error: %v", tc.name, err)
			}
			if tc.expectedErr && err == nil {
				t.Fatalf("%s: expected an error", tc.name)
			}
			if tc.expectedNum != len(podList.Items) {
				t.Fatalf("%s: expected %d pods, got %d", tc.name, tc.expectedNum, len(podList.Items))
			}
			if !apiequality.Semantic.DeepEqual(tc.expected, podList) {
				t.Fatalf("%s:\nexpected podList:\n%#v\ngot:\n%#v\n\n", tc.name, tc.expected, podList)
			}
			if watcher != nil {
				// Validate the GetPodList stopped the watcher when done
				select {
				case _, ok := <-watcher.StopChan():
					if !ok {
						// closed as expected
						break
					}
					t.Fatalf("Unexpected stop channel event")
				case <-time.After(wait.ForeverTestTimeout):
					t.Fatalf("Expected watcher to be stopped")
				}
			}
		})
	}
}

func TestGetFirstPod(t *testing.T) {
	labelSet := map[string]string{"test": "selector"}
	tests := []struct {
		name string

		podList  *corev1.PodList
		watching []watch.Event
		sortBy   func([]*corev1.Pod) sort.Interface

		expected    *corev1.Pod
		expectedNum int
		expectedErr bool
	}{
		{
			name:    "kubectl logs - two ready pods",
			podList: newPodList(2, -1, -1, labelSet),
			sortBy:  func(pods []*corev1.Pod) sort.Interface { return podutils.ByLogging(pods) },
			expected: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "pod-1",
					Namespace:         metav1.NamespaceDefault,
					CreationTimestamp: metav1.Date(2016, time.April, 1, 1, 0, 0, 0, time.UTC),
					Labels:            map[string]string{"test": "selector"},
				},
				Status: corev1.PodStatus{
					Conditions: []corev1.PodCondition{
						{
							Status: corev1.ConditionTrue,
							Type:   corev1.PodReady,
						},
					},
				},
			},
			expectedNum: 2,
		},
		{
			name:    "kubectl logs - one unhealthy, one healthy",
			podList: newPodList(2, -1, 1, labelSet),
			sortBy:  func(pods []*corev1.Pod) sort.Interface { return podutils.ByLogging(pods) },
			expected: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "pod-2",
					Namespace:         metav1.NamespaceDefault,
					CreationTimestamp: metav1.Date(2016, time.April, 1, 1, 0, 1, 0, time.UTC),
					Labels:            map[string]string{"test": "selector"},
				},
				Status: corev1.PodStatus{
					Conditions: []corev1.PodCondition{
						{
							Status: corev1.ConditionTrue,
							Type:   corev1.PodReady,
						},
					},
					ContainerStatuses: []corev1.ContainerStatus{{RestartCount: 5}},
				},
			},
			expectedNum: 2,
		},
		{
			name:    "kubectl attach - two ready pods",
			podList: newPodList(2, -1, -1, labelSet),
			sortBy:  func(pods []*corev1.Pod) sort.Interface { return sort.Reverse(podutils.ActivePods(pods)) },
			expected: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "pod-1",
					Namespace:         metav1.NamespaceDefault,
					CreationTimestamp: metav1.Date(2016, time.April, 1, 1, 0, 0, 0, time.UTC),
					Labels:            map[string]string{"test": "selector"},
				},
				Status: corev1.PodStatus{
					Conditions: []corev1.PodCondition{
						{
							Status: corev1.ConditionTrue,
							Type:   corev1.PodReady,
						},
					},
				},
			},
			expectedNum: 2,
		},
		{
			name:    "kubectl attach - wait for pod",
			podList: newPodList(0, -1, -1, labelSet),
			watching: []watch.Event{
				{
					Type: watch.Modified,
					Object: &corev1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "pod-1",
							Namespace:         metav1.NamespaceDefault,
							CreationTimestamp: metav1.Date(2016, time.April, 1, 1, 0, 0, 0, time.UTC),
							Labels:            map[string]string{"test": "selector"},
						},
					},
				},
			},
			sortBy: func(pods []*corev1.Pod) sort.Interface { return sort.Reverse(podutils.ActivePods(pods)) },
			expected: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "pod-1",
					Namespace:         metav1.NamespaceDefault,
					CreationTimestamp: metav1.Date(2016, time.April, 1, 1, 0, 0, 0, time.UTC),
					Labels:            map[string]string{"test": "selector"},
				},
			},
			expectedNum: 1,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fake := fakeexternal.NewSimpleClientset(tc.podList)
			var watcher *watch.ProxyWatcher
			if len(tc.watching) > 0 {
				resultCh := make(chan watch.Event)
				watcher = watch.NewProxyWatcher(resultCh)
				go sendWatchEvents(resultCh, watcher.StopChan(), tc.watching)
				fake.PrependWatchReactor("pods", testcore.DefaultWatchReactor(watcher, nil))
			}
			selector := labels.Set(labelSet).AsSelector()

			pod, numPods, err := GetFirstPod(fake.CoreV1(), metav1.NamespaceDefault, selector.String(), 1*time.Minute, tc.sortBy)
			pod.Spec.SecurityContext = nil
			if !tc.expectedErr && err != nil {
				t.Fatalf("%s: unexpected error: %v", tc.name, err)
			}
			if tc.expectedErr && err == nil {
				t.Fatalf("%s: expected an error", tc.name)
			}
			if tc.expectedNum != numPods {
				t.Fatalf("%s: expected %d pods, got %d", tc.name, tc.expectedNum, numPods)
			}
			if !apiequality.Semantic.DeepEqual(tc.expected, pod) {
				t.Fatalf("%s:\nexpected pod:\n%#v\ngot:\n%#v\n\n", tc.name, tc.expected, pod)
			}
			if watcher != nil {
				// Validate the GetPodList stopped the watcher when done
				select {
				case _, ok := <-watcher.StopChan():
					if !ok {
						// closed as expected
						break
					}
					t.Fatalf("Unexpected stop channel event")
				case <-time.After(wait.ForeverTestTimeout):
					t.Fatalf("Expected watcher to be stopped")
				}
			}
		})
	}
}

func newPodList(count, isUnready, isUnhealthy int, labels map[string]string) *corev1.PodList {
	pods := []corev1.Pod{}
	for i := 0; i < count; i++ {
		newPod := corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:              fmt.Sprintf("pod-%d", i+1),
				Namespace:         metav1.NamespaceDefault,
				CreationTimestamp: metav1.Date(2016, time.April, 1, 1, 0, i, 0, time.UTC),
				Labels:            labels,
			},
			Status: corev1.PodStatus{
				Conditions: []corev1.PodCondition{
					{
						Status: corev1.ConditionTrue,
						Type:   corev1.PodReady,
					},
				},
			},
		}
		pods = append(pods, newPod)
	}
	if isUnready > -1 && isUnready < count {
		pods[isUnready].Status.Conditions[0].Status = corev1.ConditionFalse
	}
	if isUnhealthy > -1 && isUnhealthy < count {
		pods[isUnhealthy].Status.ContainerStatuses = []corev1.ContainerStatus{{RestartCount: 5}}
	}
	return &corev1.PodList{
		Items: pods,
	}
}

func sendWatchEvents(resultCh chan watch.Event, stopCh <-chan struct{}, events []watch.Event) {
	defer close(resultCh)
	for _, event := range events {
		select {
		case resultCh <- event:
		case <-stopCh:
			return
		}
	}
}
