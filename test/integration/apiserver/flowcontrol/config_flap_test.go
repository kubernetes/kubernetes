/*
Copyright 2021 The Kubernetes Authors.

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

package flowcontrol

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"
	"time"

	flowcontrolv1 "k8s.io/api/flowcontrol/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	flowcontrolbootstrap "k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// TestConfigFlap tests the ability of the APF config-producer to correctly switch
// between old and new config in the cluster.
// This test runs in three phases.
// First, it runs an apiserver configured with APFv134Config=true
// and checks that that config is established within 2.5 minutes;
// then the server is shut down.
// Second, it runs an apiserver configured with APFv134Config=false
// and checks that that config is established within 2.5 minutes;
// then the server is shut down.
// Finally, it runs an apiserver configured with APFv134Config=true
// and checks that that config is established within 2.5 minutes;
// then the server is shut down.
// Remember that the etcd server is bound in test_main, so all
// three phases share the same etcd server.
func TestConfigFlap(t *testing.T) {
	tCtx := ktesting.Init(t)
	if !flapTo(t, tCtx, "first", true) {
		return
	}
	if !flapTo(t, tCtx, "first", false) {
		return
	}
	if !flapTo(t, tCtx, "second", true) {
		return
	}
	tCtx.Cancel("test function done")
}

var errTestPhaseDone = errors.New("test phase done")

func flapTo(t *testing.T, ctx context.Context, trial string, v134 bool) bool {
	flapName := fmt.Sprintf("%s v134=%v", trial, v134)
	t.Log("Preparing for " + flapName)
	ctx, cancel := context.WithCancelCause(ctx)
	client, _, tearDown := setupAnother(t, ctx, 100, 100, v134)
	defer func() {
		cancel(errTestPhaseDone)
		tearDown()
	}()
	t.Log("Waiting for " + flapName)
	if !waitForConfig(t, ctx, client, flapName, v134) {
		t.Fatal("Failed to establish " + flapName)
		return false
	}
	t.Log("Established " + flapName)
	return true
}

var errSuccess = errors.New("gotit")

func waitForConfig(t *testing.T, ctx context.Context, client clientset.Interface, phaseName string, v134 bool) bool {
	expectedSlices := flowcontrolbootstrap.GetV1ConfigCollection(v134)
	expectedPLCs := slicesToMap(widenToObject, expectedSlices.Mandatory.PriorityLevelConfigurations, expectedSlices.Suggested.PriorityLevelConfigurations)
	expectedFSes := slicesToMap(widenToObject, expectedSlices.Mandatory.FlowSchemas, expectedSlices.Suggested.FlowSchemas)
	var iteration int
	err := wait.PollUntilContextTimeout(ctx, 20*time.Second, 150*time.Second, false, func(ctx context.Context) (done bool, err error) {
		defer func() { iteration++ }()
		logger := klog.FromContext(ctx)
		step := fmt.Sprintf("%s iteration %d", phaseName, iteration)
		t.Log("Starting " + step)
		actualPLCList, err := client.FlowcontrolV1().PriorityLevelConfigurations().List(ctx, metav1.ListOptions{})
		if err != nil {
			logger.Error(err, "Failed to list PriorityLevelConfiguration objects")
			return false, nil
		}
		actualPLCs := slicesToMap(plcToObject, actualPLCList.Items)
		plcsGood := compareMaps(t, step, "PriorityLevelConfiguration", plcToSpec, expectedPLCs, actualPLCs)
		actualFSList, err := client.FlowcontrolV1().FlowSchemas().List(ctx, metav1.ListOptions{})
		if err != nil {
			logger.Error(err, "Failed to list FlowSchema objects")
			return false, nil
		}
		actualFSes := slicesToMap(fsToObject, actualFSList.Items)
		fsesGood := compareMaps(t, step, "FlowSchema", fsToSpec, expectedFSes, actualFSes)
		t.Logf("In %s, plcsGood=%v, fsesGood=%v", step, plcsGood, fsesGood)
		if plcsGood && fsesGood {
			return false, errSuccess
		}
		return false, nil
	})
	return errors.Is(err, errSuccess)
}

func slicesToMap[Elt any](widen func(Elt) metav1.Object, slices ...[]Elt) map[string]Elt {
	ans := make(map[string]Elt)
	for _, aslice := range slices {
		for _, elt := range aslice {
			obj := widen(elt)
			ans[obj.GetName()] = elt
		}
	}
	return ans
}

func widenToObject[Specific metav1.Object](obj Specific) metav1.Object       { return obj }
func plcToObject(plc flowcontrolv1.PriorityLevelConfiguration) metav1.Object { return &plc }
func fsToObject(fs flowcontrolv1.FlowSchema) metav1.Object                   { return &fs }
func plcToSpec(plc *flowcontrolv1.PriorityLevelConfiguration) flowcontrolv1.PriorityLevelConfigurationSpec {
	return plc.Spec
}
func fsToSpec(fs *flowcontrolv1.FlowSchema) flowcontrolv1.FlowSchemaSpec {
	return fs.Spec
}

func compareMaps[Elt, Spec any](t *testing.T, step, kind string, getSpec func(*Elt) Spec, expected map[string]*Elt, actual map[string]Elt) bool {
	allGood := true
	for name, expectedVal := range expected {
		actualVal, ok := actual[name]
		if !ok {
			allGood = false
			t.Logf("At %s: did not find expected %s %s", step, kind, name)
			continue
		}
		expectedSpec := getSpec(expectedVal)
		actualSpec := getSpec(&actualVal)
		if !apiequality.Semantic.DeepEqual(expectedSpec, actualSpec) {
			allGood = false
			expectedBytes, err := json.Marshal(expectedSpec)
			if err != nil {
				t.Errorf("Failed to marshal expectedSpec %#v: %s", expectedSpec, err.Error())
				continue
			}
			actualBytes, err := json.Marshal(actualSpec)
			if err != nil {
				t.Errorf("Failed to marshal actualSpec %#v: %s", actualSpec, err.Error())
				continue
			}
			t.Logf("At %s: for %s %s, expectedSpec %s != actualSpec %s", step, kind, name, string(expectedBytes), string(actualBytes))
		}
	}
	for name := range actual {
		_, ok := expected[name]
		if !ok {
			allGood = false
			t.Logf("At %s: did not expect actual %s %s", step, kind, name)
		}
	}
	return allGood
}
