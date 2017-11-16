/*
Copyright 2016 The Kubernetes Authors.

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

package e2e_node

import (
	"fmt"
	"reflect"
	"time"

	"github.com/davecgh/go-spew/spew"

	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/status"

	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

type configState struct {
	desc           string
	configSource   *apiv1.NodeConfigSource
	expectConfigOK *apiv1.NodeCondition
	expectConfig   *kubeletconfig.KubeletConfiguration
}

// This test is marked [Disruptive] because the Kubelet restarts several times during this test.
var _ = framework.KubeDescribe("DynamicKubeletConfiguration [Feature:DynamicKubeletConfig] [Serial] [Disruptive]", func() {
	f := framework.NewDefaultFramework("dynamic-kubelet-configuration-test")
	var originalKC *kubeletconfig.KubeletConfiguration
	var originalConfigMap *apiv1.ConfigMap

	// Dummy context to prevent framework's AfterEach from cleaning up before this test's AfterEach can run
	Context("", func() {
		BeforeEach(func() {
			var err error
			if originalConfigMap == nil {
				originalKC, err = getCurrentKubeletConfig()
				framework.ExpectNoError(err)
				originalConfigMap = newKubeletConfigMap("original-values", originalKC)
				originalConfigMap, err = f.ClientSet.CoreV1().ConfigMaps("kube-system").Create(originalConfigMap)
				framework.ExpectNoError(err)
			}
			// make sure Dynamic Kubelet Configuration feature is enabled on the Kubelet we are about to test
			enabled, err := isKubeletConfigEnabled(f)
			framework.ExpectNoError(err)
			if !enabled {
				framework.ExpectNoError(fmt.Errorf("The Dynamic Kubelet Configuration feature is not enabled.\n" +
					"Pass --feature-gates=DynamicKubeletConfig=true to the Kubelet to enable this feature.\n" +
					"For `make test-e2e-node`, you can set `TEST_ARGS='--feature-gates=DynamicKubeletConfig=true'`."))
			}
		})

		AfterEach(func() {
			// Set the config back to the original values before moving on.
			// We care that the values are the same, not where they come from, so it
			// should be fine to reset the values using a remote config, even if they
			// were initially set via the locally provisioned configuration.
			// This is the same strategy several other e2e node tests use.
			setAndTestKubeletConfigState(f, &configState{desc: "reset to original values",
				configSource: &apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{
					UID:       originalConfigMap.UID,
					Namespace: originalConfigMap.Namespace,
					Name:      originalConfigMap.Name}},
				expectConfigOK: &apiv1.NodeCondition{Type: apiv1.NodeConfigOK, Status: apiv1.ConditionTrue,
					Message: fmt.Sprintf(status.CurRemoteMessageFmt, originalConfigMap.UID),
					Reason:  status.CurRemoteOKReason},
				expectConfig: originalKC})
		})

		Context("When setting new NodeConfigSources that cause transitions between ConfigOK conditions", func() {
			It("the Kubelet should report the appropriate status and configz", func() {
				var err error
				// we base the "correct" configmap off of the current configuration,
				// but we also set the trial duration very high to prevent changing the last-known-good
				correctKC := originalKC.DeepCopy()
				correctKC.ConfigTrialDuration = &metav1.Duration{Duration: time.Hour}
				correctConfigMap := newKubeletConfigMap("dynamic-kubelet-config-test-correct", correctKC)
				correctConfigMap, err = f.ClientSet.CoreV1().ConfigMaps("kube-system").Create(correctConfigMap)
				framework.ExpectNoError(err)

				// fail to parse, we insert some bogus stuff into the configMap
				failParseConfigMap := &apiv1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Name: "dynamic-kubelet-config-test-fail-parse"},
					Data: map[string]string{
						"kubelet": "{0xdeadbeef}",
					},
				}
				failParseConfigMap, err = f.ClientSet.CoreV1().ConfigMaps("kube-system").Create(failParseConfigMap)
				framework.ExpectNoError(err)

				// fail to validate, we make a copy and set an invalid KubeAPIQPS on kc before serializing
				invalidKC := correctKC.DeepCopy()

				invalidKC.KubeAPIQPS = -1
				failValidateConfigMap := newKubeletConfigMap("dynamic-kubelet-config-test-fail-validate", invalidKC)
				failValidateConfigMap, err = f.ClientSet.CoreV1().ConfigMaps("kube-system").Create(failValidateConfigMap)
				framework.ExpectNoError(err)

				states := []configState{
					// Node.Spec.ConfigSource is nil
					{desc: "Node.Spec.ConfigSource is nil",
						configSource: nil,
						expectConfigOK: &apiv1.NodeCondition{Type: apiv1.NodeConfigOK, Status: apiv1.ConditionTrue,
							Message: status.CurDefaultMessage,
							Reason:  status.CurDefaultOKReason},
						expectConfig: nil},

					// Node.Spec.ConfigSource has all nil subfields
					{desc: "Node.Spec.ConfigSource has all nil subfields",
						configSource: &apiv1.NodeConfigSource{ConfigMapRef: nil},
						expectConfigOK: &apiv1.NodeCondition{Type: apiv1.NodeConfigOK, Status: apiv1.ConditionFalse,
							Message: "",
							Reason:  fmt.Sprintf(status.FailSyncReasonFmt, status.FailSyncReasonAllNilSubfields)},
						expectConfig: nil},

					// Node.Spec.ConfigSource.ConfigMapRef is partial
					{desc: "Node.Spec.ConfigSource.ConfigMapRef is partial",
						// TODO(mtaufen): check the other 7 partials in a unit test
						configSource: &apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{
							UID:  "foo",
							Name: "bar"}}, // missing Namespace
						expectConfigOK: &apiv1.NodeCondition{Type: apiv1.NodeConfigOK, Status: apiv1.ConditionFalse,
							Message: "",
							Reason:  fmt.Sprintf(status.FailSyncReasonFmt, status.FailSyncReasonPartialObjectReference)},
						expectConfig: nil},

					// Node.Spec.ConfigSource's UID does not align with namespace/name
					{desc: "Node.Spec.ConfigSource's UID does not align with namespace/name",
						configSource: &apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{UID: "foo",
							Namespace: correctConfigMap.Namespace,
							Name:      correctConfigMap.Name}},
						expectConfigOK: &apiv1.NodeCondition{Type: apiv1.NodeConfigOK, Status: apiv1.ConditionFalse,
							Message: "",
							Reason:  fmt.Sprintf(status.FailSyncReasonFmt, fmt.Sprintf(status.FailSyncReasonUIDMismatchFmt, "foo", correctConfigMap.UID))},
						expectConfig: nil},

					// correct
					{desc: "correct",
						configSource: &apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{
							UID:       correctConfigMap.UID,
							Namespace: correctConfigMap.Namespace,
							Name:      correctConfigMap.Name}},
						expectConfigOK: &apiv1.NodeCondition{Type: apiv1.NodeConfigOK, Status: apiv1.ConditionTrue,
							Message: fmt.Sprintf(status.CurRemoteMessageFmt, correctConfigMap.UID),
							Reason:  status.CurRemoteOKReason},
						expectConfig: correctKC},

					// fail-parse
					{desc: "fail-parse",
						configSource: &apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{
							UID:       failParseConfigMap.UID,
							Namespace: failParseConfigMap.Namespace,
							Name:      failParseConfigMap.Name}},
						expectConfigOK: &apiv1.NodeCondition{Type: apiv1.NodeConfigOK, Status: apiv1.ConditionFalse,
							Message: status.LkgDefaultMessage,
							Reason:  fmt.Sprintf(status.CurFailParseReasonFmt, failParseConfigMap.UID)},
						expectConfig: nil},

					// fail-validate
					{desc: "fail-validate",
						configSource: &apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{
							UID:       failValidateConfigMap.UID,
							Namespace: failValidateConfigMap.Namespace,
							Name:      failValidateConfigMap.Name}},
						expectConfigOK: &apiv1.NodeCondition{Type: apiv1.NodeConfigOK, Status: apiv1.ConditionFalse,
							Message: status.LkgDefaultMessage,
							Reason:  fmt.Sprintf(status.CurFailValidateReasonFmt, failValidateConfigMap.UID)},
						expectConfig: nil},
				}

				L := len(states)
				for i := 1; i <= L; i++ { // need one less iteration than the number of states
					testBothDirections(f, &states[i-1 : i][0], states[i:L])
				}

			})
		})

		Context("When a remote config becomes the new last-known-good before the Kubelet is updated to use a new, bad config", func() {
			It("it should report a status and configz indicating that it rolled back to the new last-known-good", func() {
				var err error
				// we base the "lkg" configmap off of the current configuration, but set the trial
				// duration very low so that it quickly becomes the last-known-good
				lkgKC := originalKC.DeepCopy()
				lkgKC.ConfigTrialDuration = &metav1.Duration{Duration: time.Nanosecond}
				lkgConfigMap := newKubeletConfigMap("dynamic-kubelet-config-test-intended-lkg", lkgKC)
				lkgConfigMap, err = f.ClientSet.CoreV1().ConfigMaps("kube-system").Create(lkgConfigMap)
				framework.ExpectNoError(err)

				// bad config map, we insert some bogus stuff into the configMap
				badConfigMap := &apiv1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Name: "dynamic-kubelet-config-test-bad"},
					Data: map[string]string{
						"kubelet": "{0xdeadbeef}",
					},
				}
				badConfigMap, err = f.ClientSet.CoreV1().ConfigMaps("kube-system").Create(badConfigMap)
				framework.ExpectNoError(err)

				states := []configState{
					// intended lkg
					{desc: "intended last-known-good",
						configSource: &apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{
							UID:       lkgConfigMap.UID,
							Namespace: lkgConfigMap.Namespace,
							Name:      lkgConfigMap.Name}},
						expectConfigOK: &apiv1.NodeCondition{Type: apiv1.NodeConfigOK, Status: apiv1.ConditionTrue,
							Message: fmt.Sprintf(status.CurRemoteMessageFmt, lkgConfigMap.UID),
							Reason:  status.CurRemoteOKReason},
						expectConfig: lkgKC},

					// bad config
					{desc: "bad config",
						configSource: &apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{
							UID:       badConfigMap.UID,
							Namespace: badConfigMap.Namespace,
							Name:      badConfigMap.Name}},
						expectConfigOK: &apiv1.NodeCondition{Type: apiv1.NodeConfigOK, Status: apiv1.ConditionFalse,
							Message: fmt.Sprintf(status.LkgRemoteMessageFmt, lkgConfigMap.UID),
							Reason:  fmt.Sprintf(status.CurFailParseReasonFmt, badConfigMap.UID)},
						expectConfig: lkgKC},
				}

				testBothDirections(f, &states[0], states[1:])
			})
		})

		// This stress test will help turn up resource leaks across kubelet restarts that can, over time,
		// break our ability to dynamically update kubelet config
		Context("When changing the configuration 100 times", func() {
			It("the Kubelet should report the appropriate status and configz", func() {
				var err error

				// we just create two configmaps with the same config but different names and toggle between them
				kc1 := originalKC.DeepCopy()
				cm1 := newKubeletConfigMap("dynamic-kubelet-config-test-cm1", kc1)
				cm1, err = f.ClientSet.CoreV1().ConfigMaps("kube-system").Create(cm1)
				framework.ExpectNoError(err)

				// slightly change the config
				kc2 := kc1.DeepCopy()
				kc2.EventRecordQPS = kc1.EventRecordQPS + 1
				cm2 := newKubeletConfigMap("dynamic-kubelet-config-test-cm2", kc2)
				cm2, err = f.ClientSet.CoreV1().ConfigMaps("kube-system").Create(cm2)
				framework.ExpectNoError(err)

				states := []configState{
					{desc: "cm1",
						configSource: &apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{
							UID:       cm1.UID,
							Namespace: cm1.Namespace,
							Name:      cm1.Name}},
						expectConfigOK: &apiv1.NodeCondition{Type: apiv1.NodeConfigOK, Status: apiv1.ConditionTrue,
							Message: fmt.Sprintf(status.CurRemoteMessageFmt, cm1.UID),
							Reason:  status.CurRemoteOKReason},
						expectConfig: kc1},
					{desc: "cm2",
						configSource: &apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{
							UID:       cm2.UID,
							Namespace: cm2.Namespace,
							Name:      cm2.Name}},
						expectConfigOK: &apiv1.NodeCondition{Type: apiv1.NodeConfigOK, Status: apiv1.ConditionTrue,
							Message: fmt.Sprintf(status.CurRemoteMessageFmt, cm2.UID),
							Reason:  status.CurRemoteOKReason},
						expectConfig: kc2},
				}

				for i := 0; i < 50; i++ { // change the config 101 times (changes 3 times in the first iteration, 2 times in each subsequent iteration)
					testBothDirections(f, &states[0], states[1:])
				}
			})
		})
	})
})

// testBothDirections tests the state change represented by each edge, where each state is a vertex,
// and there are edges in each direction between first and each of the states.
func testBothDirections(f *framework.Framework, first *configState, states []configState) {
	// set to first and check that everything got set up properly
	By(fmt.Sprintf("setting configSource to state %q", first.desc))
	setAndTestKubeletConfigState(f, first)

	// for each state, set to that state, check condition and configz, then reset to first and check again
	for i := range states {
		By(fmt.Sprintf("from %q to %q", first.desc, states[i].desc))
		setAndTestKubeletConfigState(f, &states[i])

		By(fmt.Sprintf("back to %q from %q", first.desc, states[i].desc))
		setAndTestKubeletConfigState(f, first)
	}
}

// setAndTestKubeletConfigState tests that after setting the config source, the ConfigOK condition
// and (if appropriate) configuration exposed via conifgz are as expected.
// The configuration will be converted to the internal type prior to comparison.
func setAndTestKubeletConfigState(f *framework.Framework, state *configState) {
	// set the desired state, retry a few times in case we are competing with other editors
	Eventually(func() error {
		if err := setNodeConfigSource(f, state.configSource); err != nil {
			return err
		}
		return nil
	}, time.Minute, time.Second).Should(BeNil())
	// check that config source actually got set to what we expect
	checkNodeConfigSource(f, state.configSource)
	// check condition
	checkConfigOKCondition(f, state.expectConfigOK)
	// check expectConfig
	if state.expectConfig != nil {
		checkConfig(f, state.expectConfig)
	}
}

// make sure the node's config source matches what we expect, after setting it
func checkNodeConfigSource(f *framework.Framework, expect *apiv1.NodeConfigSource) {
	const (
		timeout  = time.Minute
		interval = time.Second
	)

	Eventually(func() error {
		node, err := f.ClientSet.CoreV1().Nodes().Get(framework.TestContext.NodeName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		actual := node.Spec.ConfigSource
		if !reflect.DeepEqual(expect, actual) {
			return fmt.Errorf(spew.Sprintf("expected %#v but got %#v", expect, actual))
		}
		return nil
	}, timeout, interval).Should(BeNil())
}

// make sure the ConfigOK node condition eventually matches what we expect
func checkConfigOKCondition(f *framework.Framework, expect *apiv1.NodeCondition) {
	const (
		timeout  = time.Minute
		interval = time.Second
	)

	Eventually(func() error {
		node, err := f.ClientSet.CoreV1().Nodes().Get(framework.TestContext.NodeName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		actual := getConfigOKCondition(node.Status.Conditions)
		if actual == nil {
			return fmt.Errorf("ConfigOK condition not found on node %q", framework.TestContext.NodeName)
		}
		if err := expectConfigOK(expect, actual); err != nil {
			return err
		}
		return nil
	}, timeout, interval).Should(BeNil())
}

// if the actual matches the expect, return nil, else error explaining the mismatch
// if a subfield of the expect is the empty string, that check is skipped
func expectConfigOK(expect, actual *apiv1.NodeCondition) error {
	if expect.Status != actual.Status {
		return fmt.Errorf("expected condition Status %q but got %q", expect.Status, actual.Status)
	}
	if len(expect.Message) > 0 && expect.Message != actual.Message {
		return fmt.Errorf("expected condition Message %q but got %q", expect.Message, actual.Message)
	}
	if len(expect.Reason) > 0 && expect.Reason != actual.Reason {
		return fmt.Errorf("expected condition Reason %q but got %q", expect.Reason, actual.Reason)
	}
	return nil
}

// make sure config exposed on configz matches what we expect
func checkConfig(f *framework.Framework, expect *kubeletconfig.KubeletConfiguration) {
	const (
		timeout  = time.Minute
		interval = time.Second
	)
	Eventually(func() error {
		actual, err := getCurrentKubeletConfig()
		if err != nil {
			return err
		}
		if !reflect.DeepEqual(expect, actual) {
			return fmt.Errorf(spew.Sprintf("expected %#v but got %#v", expect, actual))
		}
		return nil
	}, timeout, interval).Should(BeNil())
}
