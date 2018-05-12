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
	"strings"
	"time"

	"github.com/davecgh/go-spew/spew"

	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	controller "k8s.io/kubernetes/pkg/kubelet/kubeletconfig"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/status"

	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

type configState struct {
	desc           string
	configSource   *apiv1.NodeConfigSource
	expectConfigOk *apiv1.NodeCondition
	expectConfig   *kubeletconfig.KubeletConfiguration
	// whether to expect this substring in an error returned from the API server when updating the config source
	apierr string
	// whether the state would cause a config change event as a result of the update to Node.Spec.ConfigSource,
	// assuming that the current source would have also caused a config change event.
	// for example, some malformed references may result in a download failure, in which case the Kubelet
	// does not restart to change config, while an invalid payload will be detected upon restart
	event bool
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
				configSource: &apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
					UID:              originalConfigMap.UID,
					Namespace:        originalConfigMap.Namespace,
					Name:             originalConfigMap.Name,
					KubeletConfigKey: "kubelet",
				}},
				expectConfigOk: &apiv1.NodeCondition{Type: apiv1.NodeKubeletConfigOk, Status: apiv1.ConditionTrue,
					Message: fmt.Sprintf(status.CurRemoteMessageFmt, configMapAPIPath(originalConfigMap)),
					Reason:  status.CurRemoteOkayReason},
				expectConfig: originalKC,
			}, false)
		})

		Context("When setting new NodeConfigSources that cause transitions between ConfigOk conditions", func() {
			It("the Kubelet should report the appropriate status and configz", func() {
				var err error
				// we base the "correct" configmap off of the current configuration
				correctKC := originalKC.DeepCopy()
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
					{
						desc:         "Node.Spec.ConfigSource is nil",
						configSource: nil,
						expectConfigOk: &apiv1.NodeCondition{Type: apiv1.NodeKubeletConfigOk, Status: apiv1.ConditionTrue,
							Message: status.CurLocalMessage,
							Reason:  status.CurLocalOkayReason},
						expectConfig: nil,
						event:        true,
					},
					// Node.Spec.ConfigSource has all nil subfields
					{desc: "Node.Spec.ConfigSource has all nil subfields",
						configSource: &apiv1.NodeConfigSource{},
						apierr:       "exactly one reference subfield must be non-nil",
					},
					// Node.Spec.ConfigSource.ConfigMap is partial
					{desc: "Node.Spec.ConfigSource.ConfigMap is partial",
						configSource: &apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
							UID:              "foo",
							Name:             "bar",
							KubeletConfigKey: "kubelet",
						}}, // missing Namespace
						apierr: "spec.configSource.configMap.namespace: Required value: namespace must be set in spec",
					},
					{desc: "Node.Spec.ConfigSource.ConfigMap.ResourceVersion is illegally specified",
						configSource: &apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
							UID:              "foo",
							Name:             "bar",
							Namespace:        "baz",
							ResourceVersion:  "1",
							KubeletConfigKey: "kubelet",
						}},
						apierr: "spec.configSource.configMap.resourceVersion: Forbidden: resourceVersion must not be set in spec",
					},
					{desc: "Node.Spec.ConfigSource.ConfigMap has invalid namespace",
						configSource: &apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
							UID:              "foo",
							Name:             "bar",
							Namespace:        "../baz",
							KubeletConfigKey: "kubelet",
						}},
						apierr: "spec.configSource.configMap.namespace: Invalid value",
					},
					{desc: "Node.Spec.ConfigSource.ConfigMap has invalid name",
						configSource: &apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
							UID:              "foo",
							Name:             "../bar",
							Namespace:        "baz",
							KubeletConfigKey: "kubelet",
						}},
						apierr: "spec.configSource.configMap.name: Invalid value",
					},
					{desc: "Node.Spec.ConfigSource.ConfigMap has invalid kubeletConfigKey",
						configSource: &apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
							UID:              "foo",
							Name:             "bar",
							Namespace:        "baz",
							KubeletConfigKey: "../qux",
						}},
						apierr: "spec.configSource.configMap.kubeletConfigKey: Invalid value",
					},
					{
						desc: "Node.Spec.ConfigSource.ConfigMap.UID does not align with Namespace/Name",
						configSource: &apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
							UID:              "foo",
							Namespace:        correctConfigMap.Namespace,
							Name:             correctConfigMap.Name,
							KubeletConfigKey: "kubelet",
						}},
						expectConfigOk: &apiv1.NodeCondition{Type: apiv1.NodeKubeletConfigOk, Status: apiv1.ConditionFalse,
							Message: "",
							Reason:  fmt.Sprintf(status.FailSyncReasonFmt, fmt.Sprintf(status.FailSyncReasonUIDMismatchFmt, "foo", configMapAPIPath(correctConfigMap), correctConfigMap.UID))},
						expectConfig: nil,
						event:        false,
					},
					{
						desc: "correct",
						configSource: &apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
							UID:              correctConfigMap.UID,
							Namespace:        correctConfigMap.Namespace,
							Name:             correctConfigMap.Name,
							KubeletConfigKey: "kubelet",
						}},
						expectConfigOk: &apiv1.NodeCondition{Type: apiv1.NodeKubeletConfigOk, Status: apiv1.ConditionTrue,
							Message: fmt.Sprintf(status.CurRemoteMessageFmt, configMapAPIPath(correctConfigMap)),
							Reason:  status.CurRemoteOkayReason},
						expectConfig: correctKC,
						event:        true,
					},
					{
						desc: "fail-parse",
						configSource: &apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
							UID:              failParseConfigMap.UID,
							Namespace:        failParseConfigMap.Namespace,
							Name:             failParseConfigMap.Name,
							KubeletConfigKey: "kubelet",
						}},
						expectConfigOk: &apiv1.NodeCondition{Type: apiv1.NodeKubeletConfigOk, Status: apiv1.ConditionFalse,
							Message: status.LkgLocalMessage,
							Reason:  fmt.Sprintf(status.CurFailLoadReasonFmt, configMapAPIPath(failParseConfigMap))},
						expectConfig: nil,
						event:        true,
					},
					{
						desc: "fail-validate",
						configSource: &apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
							UID:              failValidateConfigMap.UID,
							Namespace:        failValidateConfigMap.Namespace,
							Name:             failValidateConfigMap.Name,
							KubeletConfigKey: "kubelet",
						}},
						expectConfigOk: &apiv1.NodeCondition{Type: apiv1.NodeKubeletConfigOk, Status: apiv1.ConditionFalse,
							Message: status.LkgLocalMessage,
							Reason:  fmt.Sprintf(status.CurFailValidateReasonFmt, configMapAPIPath(failValidateConfigMap))},
						expectConfig: nil,
						event:        true,
					},
				}

				L := len(states)
				for i := 1; i <= L; i++ { // need one less iteration than the number of states
					testBothDirections(f, &states[i-1 : i][0], states[i:L], 0)
				}

			})
		})

		Context("When a remote config becomes the new last-known-good, and then the Kubelet is updated to use a new, bad config", func() {
			It("the Kubelet should report a status and configz indicating that it rolled back to the new last-known-good", func() {
				var err error
				// we base the "lkg" configmap off of the current configuration
				lkgKC := originalKC.DeepCopy()
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
						configSource: &apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
							UID:              lkgConfigMap.UID,
							Namespace:        lkgConfigMap.Namespace,
							Name:             lkgConfigMap.Name,
							KubeletConfigKey: "kubelet",
						}},
						expectConfigOk: &apiv1.NodeCondition{Type: apiv1.NodeKubeletConfigOk, Status: apiv1.ConditionTrue,
							Message: fmt.Sprintf(status.CurRemoteMessageFmt, configMapAPIPath(lkgConfigMap)),
							Reason:  status.CurRemoteOkayReason},
						expectConfig: lkgKC,
						event:        true,
					},

					// bad config
					{desc: "bad config",
						configSource: &apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
							UID:              badConfigMap.UID,
							Namespace:        badConfigMap.Namespace,
							Name:             badConfigMap.Name,
							KubeletConfigKey: "kubelet",
						}},
						expectConfigOk: &apiv1.NodeCondition{Type: apiv1.NodeKubeletConfigOk, Status: apiv1.ConditionFalse,
							Message: fmt.Sprintf(status.LkgRemoteMessageFmt, configMapAPIPath(lkgConfigMap)),
							Reason:  fmt.Sprintf(status.CurFailLoadReasonFmt, configMapAPIPath(badConfigMap))},
						expectConfig: lkgKC,
						event:        true,
					},
				}

				// wait 12 minutes after setting the first config to ensure it has time to pass the trial duration
				testBothDirections(f, &states[0], states[1:], 12*time.Minute)
			})
		})

		Context("When a remote config becomes the new last-known-good, and then Node.ConfigSource.ConfigMap.KubeletConfigKey is updated to use a new, bad config", func() {
			It("the Kubelet should report a status and configz indicating that it rolled back to the new last-known-good", func() {
				const badConfigKey = "bad"
				var err error
				// we base the "lkg" configmap off of the current configuration
				lkgKC := originalKC.DeepCopy()
				combinedConfigMap := newKubeletConfigMap("dynamic-kubelet-config-test-combined", lkgKC)
				combinedConfigMap.Data[badConfigKey] = "{0xdeadbeef}"
				combinedConfigMap, err = f.ClientSet.CoreV1().ConfigMaps("kube-system").Create(combinedConfigMap)
				framework.ExpectNoError(err)

				states := []configState{
					// intended lkg
					{desc: "intended last-known-good",
						configSource: &apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
							UID:              combinedConfigMap.UID,
							Namespace:        combinedConfigMap.Namespace,
							Name:             combinedConfigMap.Name,
							KubeletConfigKey: "kubelet",
						}},
						expectConfigOk: &apiv1.NodeCondition{Type: apiv1.NodeKubeletConfigOk, Status: apiv1.ConditionTrue,
							Message: fmt.Sprintf(status.CurRemoteMessageFmt, configMapAPIPath(combinedConfigMap)),
							Reason:  status.CurRemoteOkayReason},
						expectConfig: lkgKC,
						event:        true,
					},

					// bad config
					{desc: "bad config",
						configSource: &apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
							UID:              combinedConfigMap.UID,
							Namespace:        combinedConfigMap.Namespace,
							Name:             combinedConfigMap.Name,
							KubeletConfigKey: badConfigKey,
						}},
						expectConfigOk: &apiv1.NodeCondition{Type: apiv1.NodeKubeletConfigOk, Status: apiv1.ConditionFalse,
							// TODO(mtaufen): status should be more informative, and report the key being used
							Message: fmt.Sprintf(status.LkgRemoteMessageFmt, configMapAPIPath(combinedConfigMap)),
							Reason:  fmt.Sprintf(status.CurFailLoadReasonFmt, configMapAPIPath(combinedConfigMap))},
						expectConfig: lkgKC,
						event:        true,
					},
				}

				// wait 12 minutes after setting the first config to ensure it has time to pass the trial duration
				testBothDirections(f, &states[0], states[1:], 12*time.Minute)
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
						configSource: &apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
							UID:              cm1.UID,
							Namespace:        cm1.Namespace,
							Name:             cm1.Name,
							KubeletConfigKey: "kubelet",
						}},
						expectConfigOk: &apiv1.NodeCondition{Type: apiv1.NodeKubeletConfigOk, Status: apiv1.ConditionTrue,
							Message: fmt.Sprintf(status.CurRemoteMessageFmt, configMapAPIPath(cm1)),
							Reason:  status.CurRemoteOkayReason},
						expectConfig: kc1,
						event:        true,
					},

					{desc: "cm2",
						configSource: &apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
							UID:              cm2.UID,
							Namespace:        cm2.Namespace,
							Name:             cm2.Name,
							KubeletConfigKey: "kubelet",
						}},
						expectConfigOk: &apiv1.NodeCondition{Type: apiv1.NodeKubeletConfigOk, Status: apiv1.ConditionTrue,
							Message: fmt.Sprintf(status.CurRemoteMessageFmt, configMapAPIPath(cm2)),
							Reason:  status.CurRemoteOkayReason},
						expectConfig: kc2,
						event:        true,
					},
				}

				for i := 0; i < 50; i++ { // change the config 101 times (changes 3 times in the first iteration, 2 times in each subsequent iteration)
					testBothDirections(f, &states[0], states[1:], 0)
				}
			})
		})
	})
})

// testBothDirections tests the state change represented by each edge, where each state is a vertex,
// and there are edges in each direction between first and each of the states.
func testBothDirections(f *framework.Framework, first *configState, states []configState, waitAfterFirst time.Duration) {
	// set to first and check that everything got set up properly
	By(fmt.Sprintf("setting configSource to state %q", first.desc))
	// we don't always expect an event here, because setting "first" might not represent
	// a change from the current configuration
	setAndTestKubeletConfigState(f, first, false)

	time.Sleep(waitAfterFirst)

	// for each state, set to that state, check condition and configz, then reset to first and check again
	for i := range states {
		By(fmt.Sprintf("from %q to %q", first.desc, states[i].desc))
		// from first -> states[i], states[i].event fully describes whether we should get a config change event
		setAndTestKubeletConfigState(f, &states[i], states[i].event)

		By(fmt.Sprintf("back to %q from %q", first.desc, states[i].desc))
		// whether first -> states[i] should have produced a config change event partially determines whether states[i] -> first should produce an event
		setAndTestKubeletConfigState(f, first, first.event && states[i].event)
	}
}

// setAndTestKubeletConfigState tests that after setting the config source, the KubeletConfigOk condition
// and (if appropriate) configuration exposed via conifgz are as expected.
// The configuration will be converted to the internal type prior to comparison.
func setAndTestKubeletConfigState(f *framework.Framework, state *configState, expectEvent bool) {
	// set the desired state, retry a few times in case we are competing with other editors
	Eventually(func() error {
		if err := setNodeConfigSource(f, state.configSource); err != nil {
			if len(state.apierr) == 0 {
				return fmt.Errorf("case %s: expect nil error but got %q", state.desc, err.Error())
			} else if !strings.Contains(err.Error(), state.apierr) {
				return fmt.Errorf("case %s: expect error to contain %q but got %q", state.desc, state.apierr, err.Error())
			}
		} else if len(state.apierr) > 0 {
			return fmt.Errorf("case %s: expect error to contain %q but got nil error", state.desc, state.apierr)
		}
		return nil
	}, time.Minute, time.Second).Should(BeNil())
	// skip further checks if we expected an API error
	if len(state.apierr) > 0 {
		return
	}
	// check that config source actually got set to what we expect
	checkNodeConfigSource(f, state.desc, state.configSource)
	// check condition
	checkConfigOkCondition(f, state.desc, state.expectConfigOk)
	// check expectConfig
	if state.expectConfig != nil {
		checkConfig(f, state.desc, state.expectConfig)
	}
	// check that an event was sent for the config change
	if expectEvent {
		checkEvent(f, state.desc, state.configSource)
	}
}

// make sure the node's config source matches what we expect, after setting it
func checkNodeConfigSource(f *framework.Framework, desc string, expect *apiv1.NodeConfigSource) {
	const (
		timeout  = time.Minute
		interval = time.Second
	)
	Eventually(func() error {
		node, err := f.ClientSet.CoreV1().Nodes().Get(framework.TestContext.NodeName, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("checkNodeConfigSource: case %s: %v", desc, err)
		}
		actual := node.Spec.ConfigSource
		if !reflect.DeepEqual(expect, actual) {
			return fmt.Errorf(spew.Sprintf("checkNodeConfigSource: case %s: expected %#v but got %#v", desc, expect, actual))
		}
		return nil
	}, timeout, interval).Should(BeNil())
}

// make sure the ConfigOk node condition eventually matches what we expect
func checkConfigOkCondition(f *framework.Framework, desc string, expect *apiv1.NodeCondition) {
	const (
		timeout  = time.Minute
		interval = time.Second
	)
	Eventually(func() error {
		node, err := f.ClientSet.CoreV1().Nodes().Get(framework.TestContext.NodeName, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("checkConfigOkCondition: case %s: %v", desc, err)
		}
		actual := getKubeletConfigOkCondition(node.Status.Conditions)
		if actual == nil {
			return fmt.Errorf("checkConfigOkCondition: case %s: ConfigOk condition not found on node %q", desc, framework.TestContext.NodeName)
		}
		if err := expectConfigOk(expect, actual); err != nil {
			return fmt.Errorf("checkConfigOkCondition: case %s: %v", desc, err)
		}
		return nil
	}, timeout, interval).Should(BeNil())
}

// if the actual matches the expect, return nil, else error explaining the mismatch
// if a subfield of the expect is the empty string, that check is skipped
func expectConfigOk(expect, actual *apiv1.NodeCondition) error {
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
func checkConfig(f *framework.Framework, desc string, expect *kubeletconfig.KubeletConfiguration) {
	const (
		timeout  = time.Minute
		interval = time.Second
	)
	Eventually(func() error {
		actual, err := getCurrentKubeletConfig()
		if err != nil {
			return fmt.Errorf("checkConfig: case %s: %v", desc, err)
		}
		if !reflect.DeepEqual(expect, actual) {
			return fmt.Errorf(spew.Sprintf("checkConfig: case %s: expected %#v but got %#v", desc, expect, actual))
		}
		return nil
	}, timeout, interval).Should(BeNil())
}

// checkEvent makes sure an event was sent marking the Kubelet's restart to use new config,
// and that it mentions the config we expect.
func checkEvent(f *framework.Framework, desc string, expect *apiv1.NodeConfigSource) {
	const (
		timeout  = time.Minute
		interval = time.Second
	)
	Eventually(func() error {
		events, err := f.ClientSet.CoreV1().Events("").List(metav1.ListOptions{})
		if err != nil {
			return fmt.Errorf("checkEvent: case %s: %v", desc, err)
		}
		// find config changed event with most recent timestamp
		var recent *apiv1.Event
		for i := range events.Items {
			if events.Items[i].Reason == controller.KubeletConfigChangedEventReason {
				if recent == nil {
					recent = &events.Items[i]
					continue
				}
				// for these events, first and last timestamp are always the same
				if events.Items[i].FirstTimestamp.Time.After(recent.FirstTimestamp.Time) {
					recent = &events.Items[i]
				}
			}
		}

		// we expect at least one config change event
		if recent == nil {
			return fmt.Errorf("checkEvent: case %s: no events found with reason %s", desc, controller.KubeletConfigChangedEventReason)
		}

		// ensure the message is what we expect (including the resource path)
		expectMessage := fmt.Sprintf(controller.EventMessageFmt, controller.LocalConfigMessage)
		if expect != nil {
			if expect.ConfigMap != nil {
				expectMessage = fmt.Sprintf(controller.EventMessageFmt, fmt.Sprintf("/api/v1/namespaces/%s/configmaps/%s", expect.ConfigMap.Namespace, expect.ConfigMap.Name))
			}
		}
		if expectMessage != recent.Message {
			return fmt.Errorf("checkEvent: case %s: expected event message %q but got %q", desc, expectMessage, recent.Message)
		}

		return nil
	}, timeout, interval).Should(BeNil())
}

// constructs the expected SelfLink for a config map
func configMapAPIPath(cm *apiv1.ConfigMap) string {
	return fmt.Sprintf("/api/v1/namespaces/%s/configmaps/%s", cm.Namespace, cm.Name)
}
