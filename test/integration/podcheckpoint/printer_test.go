/*
Copyright The Kubernetes Authors.

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

package podcheckpoint

import (
	"encoding/json"
	"reflect"
	"testing"

	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestPodCheckpointTable(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	tCtx := ktesting.Init(t)
	client, _, closeFn := startAPIServer(tCtx, t, true)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "checkpoint-table", t)
	checkpoints := client.NodeV1alpha1().PodCheckpoints(ns.Name)
	checkpoint, err := checkpoints.Create(tCtx, newPodCheckpoint(ns.Name, "checkpoint", "source-pod"), metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	checkpoint.Status.Conditions = []metav1.Condition{{
		Type:               nodev1alpha1.PodCheckpointConditionReady,
		Status:             metav1.ConditionTrue,
		Reason:             nodev1alpha1.PodCheckpointReasonCompleted,
		LastTransitionTime: metav1.Now(),
	}}
	if _, err := checkpoints.UpdateStatus(tCtx, checkpoint, metav1.UpdateOptions{}); err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		name         string
		resourceName string
	}{
		{name: "get", resourceName: checkpoint.Name},
		{name: "list"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request := client.NodeV1alpha1().RESTClient().Get().Namespace(ns.Name).Resource("podcheckpoints").
				SetHeader("Accept", "application/json;as=Table;g=meta.k8s.io;v=v1")
			if tc.resourceName != "" {
				request = request.Name(tc.resourceName)
			}
			data, err := request.Do(tCtx).Raw()
			if err != nil {
				t.Fatalf("get PodCheckpoint table: %v", err)
			}
			var table metav1.Table
			if err := json.Unmarshal(data, &table); err != nil {
				t.Fatal(err)
			}
			var columns []string
			for _, column := range table.ColumnDefinitions {
				columns = append(columns, column.Name)
			}
			if want := []string{"Name", "Source Pod", "Status", "Age"}; !reflect.DeepEqual(columns, want) {
				t.Fatalf("table columns = %v, want %v", columns, want)
			}
			if len(table.Rows) != 1 || len(table.Rows[0].Cells) != 4 {
				t.Fatalf("expected one row with four cells, got %#v", table.Rows)
			}
			if want := []interface{}{"checkpoint", "source-pod", "CheckpointCompleted"}; !reflect.DeepEqual(table.Rows[0].Cells[:3], want) {
				t.Errorf("table cells = %v, want %v", table.Rows[0].Cells[:3], want)
			}
		})
	}
}
