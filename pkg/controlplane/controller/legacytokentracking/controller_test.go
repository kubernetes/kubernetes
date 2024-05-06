/*
Copyright 2022 The Kubernetes Authors.

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

package legacytokentracking

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/time/rate"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	testingclock "k8s.io/utils/clock/testing"
)

const throttlePeriod = 30 * time.Second

func TestSyncConfigMap(t *testing.T) {
	now := time.Now().UTC()
	tests := []struct {
		name              string
		nextCreateAt      []time.Time
		clientObjects     []runtime.Object
		existingConfigMap *corev1.ConfigMap

		expectedErr     error
		expectedActions []core.Action
	}{
		{
			name:          "create configmap [no cache, no live object]",
			clientObjects: []runtime.Object{},
			expectedActions: []core.Action{
				core.NewCreateAction(schema.GroupVersionResource{Version: "v1", Resource: "configmaps"}, metav1.NamespaceSystem, &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: ConfigMapName}, Data: map[string]string{ConfigMapDataKey: now.Format(dateFormat)}}),
			},
		},
		{
			name: "create configmap should ignore AlreadyExists error [no cache, live object exists]",
			clientObjects: []runtime.Object{
				&corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: ConfigMapName}, Data: map[string]string{ConfigMapDataKey: now.Format(dateFormat)}},
			},
			expectedActions: []core.Action{
				core.NewCreateAction(schema.GroupVersionResource{Version: "v1", Resource: "configmaps"}, metav1.NamespaceSystem, &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: ConfigMapName}, Data: map[string]string{ConfigMapDataKey: now.Format(dateFormat)}}),
			},
		},
		{
			name:          "create configmap throttled [no cache, no live object]",
			nextCreateAt:  []time.Time{now.Add(throttlePeriod - 2*time.Second), now.Add(throttlePeriod - time.Second)},
			clientObjects: []runtime.Object{},
			expectedActions: []core.Action{
				core.NewCreateAction(schema.GroupVersionResource{Version: "v1", Resource: "configmaps"}, metav1.NamespaceSystem, &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: ConfigMapName}, Data: map[string]string{ConfigMapDataKey: now.Format(dateFormat)}}),
			},
		},
		{
			name:          "create configmap after throttle period [no cache, no live object]",
			nextCreateAt:  []time.Time{now.Add(throttlePeriod - 2*time.Second), now.Add(throttlePeriod - time.Second), now.Add(throttlePeriod + time.Second)},
			clientObjects: []runtime.Object{},
			expectedActions: []core.Action{
				core.NewCreateAction(schema.GroupVersionResource{Version: "v1", Resource: "configmaps"}, metav1.NamespaceSystem, &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: ConfigMapName}, Data: map[string]string{ConfigMapDataKey: now.Format(dateFormat)}}),
				core.NewCreateAction(schema.GroupVersionResource{Version: "v1", Resource: "configmaps"}, metav1.NamespaceSystem, &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: ConfigMapName}, Data: map[string]string{ConfigMapDataKey: now.Add(throttlePeriod + time.Second).Format(dateFormat)}}),
			},
		},
		{
			name: "skip update configmap [cache with expected date format exists, live object exists]",
			clientObjects: []runtime.Object{
				&corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: ConfigMapName}, Data: map[string]string{ConfigMapDataKey: now.Format(dateFormat)}},
			},
			existingConfigMap: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: ConfigMapName},
				Data:       map[string]string{ConfigMapDataKey: now.Format(dateFormat)},
			},
			expectedActions: []core.Action{},
		},
		{
			name: "update configmap [cache with unexpected date format, live object exists]",
			clientObjects: []runtime.Object{
				&corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: ConfigMapName}, Data: map[string]string{ConfigMapDataKey: now.Format(time.RFC3339)}},
			},
			existingConfigMap: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: ConfigMapName},
				Data:       map[string]string{ConfigMapDataKey: now.Format(time.RFC3339)},
			},
			expectedActions: []core.Action{
				core.NewUpdateAction(schema.GroupVersionResource{Version: "v1", Resource: "configmaps"}, metav1.NamespaceSystem, &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: ConfigMapName}, Data: map[string]string{ConfigMapDataKey: now.Format(dateFormat)}}),
			},
		},
		{
			name: "update configmap with no data",
			clientObjects: []runtime.Object{
				&corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: ConfigMapName}, Data: nil},
			},
			existingConfigMap: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: ConfigMapName},
				Data:       nil,
			},
			expectedActions: []core.Action{
				core.NewUpdateAction(schema.GroupVersionResource{Version: "v1", Resource: "configmaps"}, metav1.NamespaceSystem, &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: ConfigMapName}, Data: map[string]string{ConfigMapDataKey: now.Format(dateFormat)}}),
			},
		},
		{
			name: "update configmap should ignore NotFound error [cache with unexpected date format, no live object]",
			existingConfigMap: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: ConfigMapName},
				Data:       map[string]string{ConfigMapDataKey: "BAD_TIMESTAMP"},
			},
			expectedActions: []core.Action{
				core.NewUpdateAction(schema.GroupVersionResource{Version: "v1", Resource: "configmaps"}, metav1.NamespaceSystem, &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: ConfigMapName}, Data: map[string]string{ConfigMapDataKey: now.Format(dateFormat)}}),
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := fake.NewSimpleClientset(test.clientObjects...)
			limiter := rate.NewLimiter(rate.Every(throttlePeriod), 1)
			controller := newController(client, testingclock.NewFakeClock(now), limiter)
			if test.existingConfigMap != nil {
				controller.configMapCache.Add(test.existingConfigMap)
			}

			if err := controller.syncConfigMap(); err != nil {
				t.Errorf("Failed to sync ConfigMap, err: %v", err)
			}

			for _, createAt := range test.nextCreateAt {
				// delete the existing configmap to trigger second create
				controller.configMapCache.Delete(&corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: ConfigMapName},
				})
				controller.clock.(*testingclock.FakeClock).SetTime(createAt)
				if err := controller.syncConfigMap(); err != nil {
					t.Errorf("Failed to sync ConfigMap, err: %v", err)
				}
			}

			if diff := cmp.Diff(test.expectedActions, client.Actions()); diff != "" {
				t.Errorf("Unexpected diff (-want +got):\n%s", diff)
			}
		})
	}
}
