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

package daemon

import (
	"reflect"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

func TestMetricsCache(t *testing.T) {
	cacheSize := 10
	m := newMetricsCache(cacheSize)
	dsName := "foo"
	ds := newDaemonSet("foo")
	ds.Spec.UpdateStrategy.Type = v1beta1.RollingUpdateDaemonSetStrategyType
	nodeName := "node-1"
	pod := newPod(dsName+"-", nodeName, ds.Spec.Template.Labels)
	recreatedPod := newPod(dsName+"-", nodeName, ds.Spec.Template.Labels)
	now := time.Now()
	oneSecLater := now.Add(time.Second)
	fiveSecLater := now.Add(5 * time.Second)

	t.Logf("Record deleting pods")
	m.recordDeletePod(ds, pod, now)
	dsNode := daemonSetNode{namespace: ds.Namespace, daemonSetName: ds.Name, nodeName: nodeName}
	obj, cached := m.Get(dsNode)
	if !cached {
		t.Errorf("Unexpected DaemonSet metrics not cached")
	}
	got, ok := obj.(podMetrics)
	if !ok {
		t.Errorf("Unexpected DaemonSet metrics type: %t", obj)
	}
	expected := podMetrics{
		deletedPodName: pod.Name,
		deleteTime:     now,
	}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("Expected DaemonSet metrics %+v, got %+v", expected, got)
	}

	t.Logf("Record recreating pods")
	m.recordRecreatePod(ds, recreatedPod, oneSecLater)
	if obj, cached = m.Get(dsNode); !cached {
		t.Errorf("Unexpected DaemonSet metrics not cached")
	}
	if got, ok = obj.(podMetrics); !ok {
		t.Errorf("Unexpected DaemonSet metrics type: %t", obj)
	}
	expected = podMetrics{
		deletedPodName:  pod.Name,
		deleteTime:      now,
		recreatePodName: recreatedPod.Name,
		recreateTime:    oneSecLater,
	}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("Expected DaemonSet metrics %+v, got %+v", expected, got)
	}

	t.Logf("Record ready pods")
	m.recordReadyPod(ds, recreatedPod, fiveSecLater)
	if obj, cached = m.Get(dsNode); !cached {
		t.Errorf("Unexpected DaemonSet metrics not cached")
	}
	if got, ok = obj.(podMetrics); !ok {
		t.Errorf("Unexpected DaemonSet metrics type: %t", obj)
	}
	expected = podMetrics{}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("Expected DaemonSet metrics %+v, got %+v", expected, got)
	}
}
