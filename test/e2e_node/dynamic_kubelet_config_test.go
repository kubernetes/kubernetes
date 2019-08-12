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

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	controller "k8s.io/kubernetes/pkg/kubelet/kubeletconfig"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/status"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	frameworkmetrics "k8s.io/kubernetes/test/e2e/framework/metrics"

	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/prometheus/common/model"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

const itDescription = "status and events should match expectations"

type expectNodeConfigStatus struct {
	lastKnownGood *v1.NodeConfigSource
	err           string
	// If true, expect Status.Config.Active == Status.Config.LastKnownGood,
	// otherwise expect Status.Config.Active == Status.Config.Assigned.
	lkgActive bool
}

type nodeConfigTestCase struct {
	desc               string
	configSource       *v1.NodeConfigSource
	configMap          *v1.ConfigMap
	expectConfigStatus expectNodeConfigStatus
	expectConfig       *kubeletconfig.KubeletConfiguration
	// whether to expect this substring in an error returned from the API server when updating the config source
	apierr string
	// whether the state would cause a config change event as a result of the update to Node.Spec.ConfigSource,
	// assuming that the current source would have also caused a config change event.
	// for example, some malformed references may result in a download failure, in which case the Kubelet
	// does not restart to change config, while an invalid payload will be detected upon restart
	event bool
}

// This test is marked [Disruptive] because the Kubelet restarts several times during this test.
var _ = framework.KubeDescribe("[Feature:DynamicKubeletConfig][NodeFeature:DynamicKubeletConfig][Serial][Disruptive]", func() {
	f := framework.NewDefaultFramework("dynamic-kubelet-configuration-test")
	var beforeNode *v1.Node
	var beforeConfigMap *v1.ConfigMap
	var beforeKC *kubeletconfig.KubeletConfiguration
	var localKC *kubeletconfig.KubeletConfiguration

	// Dummy context to prevent framework's AfterEach from cleaning up before this test's AfterEach can run
	ginkgo.Context("", func() {
		ginkgo.BeforeEach(func() {
			// make sure Dynamic Kubelet Configuration feature is enabled on the Kubelet we are about to test
			enabled, err := isKubeletConfigEnabled(f)
			framework.ExpectNoError(err)
			if !enabled {
				framework.ExpectNoError(fmt.Errorf("The Dynamic Kubelet Configuration feature is not enabled.\n" +
					"Pass --feature-gates=DynamicKubeletConfig=true to the Kubelet and API server to enable this feature.\n" +
					"For `make test-e2e-node`, you can set `TEST_ARGS='--feature-gates=DynamicKubeletConfig=true'`."))
			}
			// record before state so we can restore it after the test
			if beforeNode == nil {
				node, err := f.ClientSet.CoreV1().Nodes().Get(framework.TestContext.NodeName, metav1.GetOptions{})
				framework.ExpectNoError(err)
				beforeNode = node
			}
			if source := beforeNode.Spec.ConfigSource; source != nil {
				if source.ConfigMap != nil {
					cm, err := f.ClientSet.CoreV1().ConfigMaps(source.ConfigMap.Namespace).Get(source.ConfigMap.Name, metav1.GetOptions{})
					framework.ExpectNoError(err)
					beforeConfigMap = cm
				}
			}
			if beforeKC == nil {
				kc, err := getCurrentKubeletConfig()
				framework.ExpectNoError(err)
				beforeKC = kc
			}
			// reset the node's assigned/active/last-known-good config by setting the source to nil,
			// so each test starts from a clean-slate
			(&nodeConfigTestCase{
				desc:         "reset via nil config source",
				configSource: nil,
			}).run(f, setConfigSourceFunc, false, 0)
			// record local KC so we can check it during tests that roll back to nil last-known-good
			if localKC == nil {
				kc, err := getCurrentKubeletConfig()
				framework.ExpectNoError(err)
				localKC = kc
			}
		})

		ginkgo.AfterEach(func() {
			// clean-slate the Node again (prevents last-known-good from any tests from leaking through)
			(&nodeConfigTestCase{
				desc:         "reset via nil config source",
				configSource: nil,
			}).run(f, setConfigSourceFunc, false, 0)
			// restore the values from before the test before moving on
			restore := &nodeConfigTestCase{
				desc:         "restore values from before test",
				configSource: beforeNode.Spec.ConfigSource,
				configMap:    beforeConfigMap,
				expectConfig: beforeKC,
			}
			restore.run(f, setConfigSourceFunc, false, 0)
		})

		ginkgo.Context("update Node.Spec.ConfigSource: state transitions:", func() {
			ginkgo.It(itDescription, func() {
				var err error
				// we base the "correct" configmap off of the configuration from before the test
				correctKC := beforeKC.DeepCopy()
				correctConfigMap := newKubeletConfigMap("dynamic-kubelet-config-test-correct", correctKC)
				correctConfigMap, err = f.ClientSet.CoreV1().ConfigMaps("kube-system").Create(correctConfigMap)
				framework.ExpectNoError(err)

				// fail to parse, we insert some bogus stuff into the configMap
				failParseConfigMap := &v1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Name: "dynamic-kubelet-config-test-fail-parse"},
					Data: map[string]string{
						"kubelet": "{0xdeadbeef}",
					},
				}
				failParseConfigMap, err = f.ClientSet.CoreV1().ConfigMaps("kube-system").Create(failParseConfigMap)
				framework.ExpectNoError(err)

				// fail to validate, we make a copy of correct and set an invalid KubeAPIQPS on kc before serializing
				invalidKC := correctKC.DeepCopy()
				invalidKC.KubeAPIQPS = -1
				failValidateConfigMap := newKubeletConfigMap("dynamic-kubelet-config-test-fail-validate", invalidKC)
				failValidateConfigMap, err = f.ClientSet.CoreV1().ConfigMaps("kube-system").Create(failValidateConfigMap)
				framework.ExpectNoError(err)

				correctSource := &v1.NodeConfigSource{ConfigMap: &v1.ConfigMapNodeConfigSource{
					Namespace:        correctConfigMap.Namespace,
					Name:             correctConfigMap.Name,
					KubeletConfigKey: "kubelet",
				}}
				failParseSource := &v1.NodeConfigSource{ConfigMap: &v1.ConfigMapNodeConfigSource{
					Namespace:        failParseConfigMap.Namespace,
					Name:             failParseConfigMap.Name,
					KubeletConfigKey: "kubelet",
				}}
				failValidateSource := &v1.NodeConfigSource{ConfigMap: &v1.ConfigMapNodeConfigSource{
					Namespace:        failValidateConfigMap.Namespace,
					Name:             failValidateConfigMap.Name,
					KubeletConfigKey: "kubelet",
				}}

				cases := []nodeConfigTestCase{
					{
						desc:               "Node.Spec.ConfigSource is nil",
						configSource:       nil,
						expectConfigStatus: expectNodeConfigStatus{},
						expectConfig:       nil,
						event:              true,
					},
					{
						desc:         "Node.Spec.ConfigSource has all nil subfields",
						configSource: &v1.NodeConfigSource{},
						apierr:       "exactly one reference subfield must be non-nil",
					},
					{
						desc: "Node.Spec.ConfigSource.ConfigMap is missing namespace",
						configSource: &v1.NodeConfigSource{ConfigMap: &v1.ConfigMapNodeConfigSource{
							Name:             "bar",
							KubeletConfigKey: "kubelet",
						}}, // missing Namespace
						apierr: "spec.configSource.configMap.namespace: Required value: namespace must be set",
					},
					{
						desc: "Node.Spec.ConfigSource.ConfigMap is missing name",
						configSource: &v1.NodeConfigSource{ConfigMap: &v1.ConfigMapNodeConfigSource{
							Namespace:        "foo",
							KubeletConfigKey: "kubelet",
						}}, // missing Name
						apierr: "spec.configSource.configMap.name: Required value: name must be set",
					},
					{
						desc: "Node.Spec.ConfigSource.ConfigMap is missing kubeletConfigKey",
						configSource: &v1.NodeConfigSource{ConfigMap: &v1.ConfigMapNodeConfigSource{
							Namespace: "foo",
							Name:      "bar",
						}}, // missing KubeletConfigKey
						apierr: "spec.configSource.configMap.kubeletConfigKey: Required value: kubeletConfigKey must be set",
					},
					{
						desc: "Node.Spec.ConfigSource.ConfigMap.UID is illegally specified",
						configSource: &v1.NodeConfigSource{ConfigMap: &v1.ConfigMapNodeConfigSource{
							UID:              "foo",
							Name:             "bar",
							Namespace:        "baz",
							KubeletConfigKey: "kubelet",
						}},
						apierr: "spec.configSource.configMap.uid: Forbidden: uid must not be set in spec",
					},
					{
						desc: "Node.Spec.ConfigSource.ConfigMap.ResourceVersion is illegally specified",
						configSource: &v1.NodeConfigSource{ConfigMap: &v1.ConfigMapNodeConfigSource{
							Name:             "bar",
							Namespace:        "baz",
							ResourceVersion:  "1",
							KubeletConfigKey: "kubelet",
						}},
						apierr: "spec.configSource.configMap.resourceVersion: Forbidden: resourceVersion must not be set in spec",
					},
					{
						desc: "Node.Spec.ConfigSource.ConfigMap has invalid namespace",
						configSource: &v1.NodeConfigSource{ConfigMap: &v1.ConfigMapNodeConfigSource{
							Name:             "bar",
							Namespace:        "../baz",
							KubeletConfigKey: "kubelet",
						}},
						apierr: "spec.configSource.configMap.namespace: Invalid value",
					},
					{
						desc: "Node.Spec.ConfigSource.ConfigMap has invalid name",
						configSource: &v1.NodeConfigSource{ConfigMap: &v1.ConfigMapNodeConfigSource{
							Name:             "../bar",
							Namespace:        "baz",
							KubeletConfigKey: "kubelet",
						}},
						apierr: "spec.configSource.configMap.name: Invalid value",
					},
					{
						desc: "Node.Spec.ConfigSource.ConfigMap has invalid kubeletConfigKey",
						configSource: &v1.NodeConfigSource{ConfigMap: &v1.ConfigMapNodeConfigSource{
							Name:             "bar",
							Namespace:        "baz",
							KubeletConfigKey: "../qux",
						}},
						apierr: "spec.configSource.configMap.kubeletConfigKey: Invalid value",
					},
					{
						desc:         "correct",
						configSource: correctSource,
						configMap:    correctConfigMap,
						expectConfig: correctKC,
						event:        true,
					},
					{
						desc:         "fail-parse",
						configSource: failParseSource,
						configMap:    failParseConfigMap,
						expectConfigStatus: expectNodeConfigStatus{
							err:       status.LoadError,
							lkgActive: true,
						},
						expectConfig: localKC,
						event:        true,
					},
					{
						desc:         "fail-validate",
						configSource: failValidateSource,
						configMap:    failValidateConfigMap,
						expectConfigStatus: expectNodeConfigStatus{
							err:       status.ValidateError,
							lkgActive: true,
						},
						expectConfig: localKC,
						event:        true,
					},
				}

				L := len(cases)
				for i := 1; i <= L; i++ { // need one less iteration than the number of cases
					testBothDirections(f, setConfigSourceFunc, &cases[i-1 : i][0], cases[i:L], 0)
				}

			})
		})

		ginkgo.Context("update Node.Spec.ConfigSource: recover to last-known-good ConfigMap:", func() {
			ginkgo.It(itDescription, func() {
				var err error
				// we base the "lkg" configmap off of the configuration from before the test
				lkgKC := beforeKC.DeepCopy()
				lkgConfigMap := newKubeletConfigMap("dynamic-kubelet-config-test-intended-lkg", lkgKC)
				lkgConfigMap, err = f.ClientSet.CoreV1().ConfigMaps("kube-system").Create(lkgConfigMap)
				framework.ExpectNoError(err)

				// bad config map, we insert some bogus stuff into the configMap
				badConfigMap := &v1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Name: "dynamic-kubelet-config-test-bad"},
					Data: map[string]string{
						"kubelet": "{0xdeadbeef}",
					},
				}
				badConfigMap, err = f.ClientSet.CoreV1().ConfigMaps("kube-system").Create(badConfigMap)
				framework.ExpectNoError(err)

				lkgSource := &v1.NodeConfigSource{ConfigMap: &v1.ConfigMapNodeConfigSource{
					Namespace:        lkgConfigMap.Namespace,
					Name:             lkgConfigMap.Name,
					KubeletConfigKey: "kubelet",
				}}
				lkgStatus := lkgSource.DeepCopy()
				lkgStatus.ConfigMap.UID = lkgConfigMap.UID
				lkgStatus.ConfigMap.ResourceVersion = lkgConfigMap.ResourceVersion

				badSource := &v1.NodeConfigSource{ConfigMap: &v1.ConfigMapNodeConfigSource{
					Namespace:        badConfigMap.Namespace,
					Name:             badConfigMap.Name,
					KubeletConfigKey: "kubelet",
				}}

				cases := []nodeConfigTestCase{
					{
						desc:         "intended last-known-good",
						configSource: lkgSource,
						configMap:    lkgConfigMap,
						expectConfigStatus: expectNodeConfigStatus{
							lastKnownGood: lkgStatus,
						},
						expectConfig: lkgKC,
						event:        true,
					},
					{
						desc:         "bad config",
						configSource: badSource,
						configMap:    badConfigMap,
						expectConfigStatus: expectNodeConfigStatus{
							lastKnownGood: lkgStatus,
							err:           status.LoadError,
							lkgActive:     true,
						},
						expectConfig: lkgKC,
						event:        true,
					},
				}

				// wait 12 minutes after setting the first config to ensure it has time to pass the trial duration
				testBothDirections(f, setConfigSourceFunc, &cases[0], cases[1:], 12*time.Minute)
			})
		})

		ginkgo.Context("update Node.Spec.ConfigSource: recover to last-known-good ConfigMap.KubeletConfigKey:", func() {
			ginkgo.It(itDescription, func() {
				const badConfigKey = "bad"
				var err error
				// we base the "lkg" configmap off of the configuration from before the test
				lkgKC := beforeKC.DeepCopy()
				combinedConfigMap := newKubeletConfigMap("dynamic-kubelet-config-test-combined", lkgKC)
				combinedConfigMap.Data[badConfigKey] = "{0xdeadbeef}"
				combinedConfigMap, err = f.ClientSet.CoreV1().ConfigMaps("kube-system").Create(combinedConfigMap)
				framework.ExpectNoError(err)

				lkgSource := &v1.NodeConfigSource{ConfigMap: &v1.ConfigMapNodeConfigSource{
					Namespace:        combinedConfigMap.Namespace,
					Name:             combinedConfigMap.Name,
					KubeletConfigKey: "kubelet",
				}}

				lkgStatus := lkgSource.DeepCopy()
				lkgStatus.ConfigMap.UID = combinedConfigMap.UID
				lkgStatus.ConfigMap.ResourceVersion = combinedConfigMap.ResourceVersion

				badSource := lkgSource.DeepCopy()
				badSource.ConfigMap.KubeletConfigKey = badConfigKey

				cases := []nodeConfigTestCase{
					{
						desc:         "intended last-known-good",
						configSource: lkgSource,
						configMap:    combinedConfigMap,
						expectConfigStatus: expectNodeConfigStatus{
							lastKnownGood: lkgStatus,
						},
						expectConfig: lkgKC,
						event:        true,
					},
					{
						desc:         "bad config",
						configSource: badSource,
						configMap:    combinedConfigMap,
						expectConfigStatus: expectNodeConfigStatus{
							lastKnownGood: lkgStatus,
							err:           status.LoadError,
							lkgActive:     true,
						},
						expectConfig: lkgKC,
						event:        true,
					},
				}

				// wait 12 minutes after setting the first config to ensure it has time to pass the trial duration
				testBothDirections(f, setConfigSourceFunc, &cases[0], cases[1:], 12*time.Minute)
			})
		})

		// previously, we missed a panic because we were not exercising this path
		ginkgo.Context("update Node.Spec.ConfigSource: non-nil last-known-good to a new non-nil last-known-good", func() {
			ginkgo.It(itDescription, func() {
				var err error
				// we base the "lkg" configmap off of the configuration from before the test
				lkgKC := beforeKC.DeepCopy()
				lkgConfigMap1 := newKubeletConfigMap("dynamic-kubelet-config-test-lkg-1", lkgKC)
				lkgConfigMap1, err = f.ClientSet.CoreV1().ConfigMaps("kube-system").Create(lkgConfigMap1)
				framework.ExpectNoError(err)

				lkgSource1 := &v1.NodeConfigSource{ConfigMap: &v1.ConfigMapNodeConfigSource{
					Namespace:        lkgConfigMap1.Namespace,
					Name:             lkgConfigMap1.Name,
					KubeletConfigKey: "kubelet",
				}}
				lkgStatus1 := lkgSource1.DeepCopy()
				lkgStatus1.ConfigMap.UID = lkgConfigMap1.UID
				lkgStatus1.ConfigMap.ResourceVersion = lkgConfigMap1.ResourceVersion

				lkgConfigMap2 := newKubeletConfigMap("dynamic-kubelet-config-test-lkg-2", lkgKC)
				lkgConfigMap2, err = f.ClientSet.CoreV1().ConfigMaps("kube-system").Create(lkgConfigMap2)
				framework.ExpectNoError(err)

				lkgSource2 := &v1.NodeConfigSource{ConfigMap: &v1.ConfigMapNodeConfigSource{
					Namespace:        lkgConfigMap2.Namespace,
					Name:             lkgConfigMap2.Name,
					KubeletConfigKey: "kubelet",
				}}
				lkgStatus2 := lkgSource2.DeepCopy()
				lkgStatus2.ConfigMap.UID = lkgConfigMap2.UID
				lkgStatus2.ConfigMap.ResourceVersion = lkgConfigMap2.ResourceVersion

				// cases
				first := nodeConfigTestCase{
					desc:         "last-known-good-1",
					configSource: lkgSource1,
					configMap:    lkgConfigMap1,
					expectConfigStatus: expectNodeConfigStatus{
						lastKnownGood: lkgStatus1,
					},
					expectConfig: lkgKC,
					event:        true,
				}

				second := nodeConfigTestCase{
					desc:         "last-known-good-2",
					configSource: lkgSource2,
					configMap:    lkgConfigMap2,
					expectConfigStatus: expectNodeConfigStatus{
						lastKnownGood: lkgStatus2,
					},
					expectConfig: lkgKC,
					event:        true,
				}

				// Manually actuate this to ensure we wait for each case to become the last-known-good
				const lkgDuration = 12 * time.Minute
				ginkgo.By(fmt.Sprintf("setting initial state %q", first.desc))
				first.run(f, setConfigSourceFunc, true, lkgDuration)
				ginkgo.By(fmt.Sprintf("from %q to %q", first.desc, second.desc))
				second.run(f, setConfigSourceFunc, true, lkgDuration)
			})
		})

		// exposes resource leaks across config changes
		ginkgo.Context("update Node.Spec.ConfigSource: 100 update stress test:", func() {
			ginkgo.It(itDescription, func() {
				var err error

				// we just create two configmaps with the same config but different names and toggle between them
				kc1 := beforeKC.DeepCopy()
				cm1 := newKubeletConfigMap("dynamic-kubelet-config-test-cm1", kc1)
				cm1, err = f.ClientSet.CoreV1().ConfigMaps("kube-system").Create(cm1)
				framework.ExpectNoError(err)

				// slightly change the config
				kc2 := kc1.DeepCopy()
				kc2.EventRecordQPS = kc1.EventRecordQPS + 1
				cm2 := newKubeletConfigMap("dynamic-kubelet-config-test-cm2", kc2)
				cm2, err = f.ClientSet.CoreV1().ConfigMaps("kube-system").Create(cm2)
				framework.ExpectNoError(err)

				cm1Source := &v1.NodeConfigSource{ConfigMap: &v1.ConfigMapNodeConfigSource{
					Namespace:        cm1.Namespace,
					Name:             cm1.Name,
					KubeletConfigKey: "kubelet",
				}}

				cm2Source := &v1.NodeConfigSource{ConfigMap: &v1.ConfigMapNodeConfigSource{
					Namespace:        cm2.Namespace,
					Name:             cm2.Name,
					KubeletConfigKey: "kubelet",
				}}

				cases := []nodeConfigTestCase{
					{
						desc:         "cm1",
						configSource: cm1Source,
						configMap:    cm1,
						expectConfig: kc1,
						event:        true,
					},
					{
						desc:         "cm2",
						configSource: cm2Source,
						configMap:    cm2,
						expectConfig: kc2,
						event:        true,
					},
				}

				for i := 0; i < 50; i++ { // change the config 101 times (changes 3 times in the first iteration, 2 times in each subsequent iteration)
					testBothDirections(f, setConfigSourceFunc, &cases[0], cases[1:], 0)
				}
			})
		})

		// Please note: This behavior is tested to ensure implementation correctness. We do not, however, recommend ConfigMap mutations
		// as a usage pattern for dynamic Kubelet config in large clusters. It is much safer to create a new ConfigMap, and incrementally
		// roll out a new Node.Spec.ConfigSource that references the new ConfigMap. In-place ConfigMap updates, including deletion
		// followed by re-creation, will cause all observing Kubelets to immediately restart for new config, because these operations
		// change the ResourceVersion of the ConfigMap.
		ginkgo.Context("update ConfigMap in-place: state transitions:", func() {
			ginkgo.It(itDescription, func() {
				var err error
				// we base the "correct" configmap off of the configuration from before the test
				correctKC := beforeKC.DeepCopy()
				correctConfigMap := newKubeletConfigMap("dynamic-kubelet-config-test-in-place", correctKC)
				correctConfigMap, err = f.ClientSet.CoreV1().ConfigMaps("kube-system").Create(correctConfigMap)
				framework.ExpectNoError(err)

				// we reuse the same name, namespace
				failParseConfigMap := correctConfigMap.DeepCopy()
				failParseConfigMap.Data = map[string]string{
					"kubelet": "{0xdeadbeef}",
				}

				// fail to validate, we make a copy and set an invalid KubeAPIQPS on kc before serializing
				invalidKC := correctKC.DeepCopy()
				invalidKC.KubeAPIQPS = -1
				failValidateConfigMap := correctConfigMap.DeepCopy()
				failValidateConfigMap.Data = newKubeletConfigMap("", invalidKC).Data

				// ensure node config source is set to the config map we will mutate in-place,
				// since updateConfigMapFunc doesn't mutate Node.Spec.ConfigSource
				source := &v1.NodeConfigSource{
					ConfigMap: &v1.ConfigMapNodeConfigSource{
						Namespace:        correctConfigMap.Namespace,
						Name:             correctConfigMap.Name,
						KubeletConfigKey: "kubelet",
					},
				}
				(&nodeConfigTestCase{
					desc:         "initial state (correct)",
					configSource: source,
					configMap:    correctConfigMap,
					expectConfig: correctKC,
				}).run(f, setConfigSourceFunc, false, 0)

				cases := []nodeConfigTestCase{
					{
						desc:         "correct",
						configSource: source,
						configMap:    correctConfigMap,
						expectConfig: correctKC,
						event:        true,
					},
					{
						desc:         "fail-parse",
						configSource: source,
						configMap:    failParseConfigMap,
						expectConfigStatus: expectNodeConfigStatus{
							err:       status.LoadError,
							lkgActive: true,
						},
						expectConfig: localKC,
						event:        true,
					},
					{
						desc:         "fail-validate",
						configSource: source,
						configMap:    failValidateConfigMap,
						expectConfigStatus: expectNodeConfigStatus{
							err:       status.ValidateError,
							lkgActive: true,
						},
						expectConfig: localKC,
						event:        true,
					},
				}
				L := len(cases)
				for i := 1; i <= L; i++ { // need one less iteration than the number of cases
					testBothDirections(f, updateConfigMapFunc, &cases[i-1 : i][0], cases[i:L], 0)
				}
			})
		})

		// Please note: This behavior is tested to ensure implementation correctness. We do not, however, recommend ConfigMap mutations
		// as a usage pattern for dynamic Kubelet config in large clusters. It is much safer to create a new ConfigMap, and incrementally
		// roll out a new Node.Spec.ConfigSource that references the new ConfigMap. In-place ConfigMap updates, including deletion
		// followed by re-creation, will cause all observing Kubelets to immediately restart for new config, because these operations
		// change the ResourceVersion of the ConfigMap.
		ginkgo.Context("update ConfigMap in-place: recover to last-known-good version:", func() {
			ginkgo.It(itDescription, func() {
				var err error
				// we base the "lkg" configmap off of the configuration from before the test
				lkgKC := beforeKC.DeepCopy()
				lkgConfigMap := newKubeletConfigMap("dynamic-kubelet-config-test-in-place-lkg", lkgKC)
				lkgConfigMap, err = f.ClientSet.CoreV1().ConfigMaps("kube-system").Create(lkgConfigMap)
				framework.ExpectNoError(err)

				// bad config map, we insert some bogus stuff into the configMap
				badConfigMap := lkgConfigMap.DeepCopy()
				badConfigMap.Data = map[string]string{
					"kubelet": "{0xdeadbeef}",
				}
				// ensure node config source is set to the config map we will mutate in-place
				source := &v1.NodeConfigSource{
					ConfigMap: &v1.ConfigMapNodeConfigSource{
						Namespace:        lkgConfigMap.Namespace,
						Name:             lkgConfigMap.Name,
						KubeletConfigKey: "kubelet",
					},
				}

				// Even though the first test case will PUT the lkgConfigMap again, no-op writes don't increment
				// ResourceVersion, so the expected status we record here will still be correct.
				lkgStatus := source.DeepCopy()
				lkgStatus.ConfigMap.UID = lkgConfigMap.UID
				lkgStatus.ConfigMap.ResourceVersion = lkgConfigMap.ResourceVersion

				(&nodeConfigTestCase{
					desc:         "initial state (correct)",
					configSource: source,
					configMap:    lkgConfigMap,
					expectConfig: lkgKC,
				}).run(f, setConfigSourceFunc, false, 0) // wait 0 here, and we should not expect LastKnownGood to have changed yet (hence nil)

				cases := []nodeConfigTestCase{
					{
						desc:         "intended last-known-good",
						configSource: source,
						configMap:    lkgConfigMap,
						expectConfigStatus: expectNodeConfigStatus{
							lastKnownGood: lkgStatus,
						},
						expectConfig: lkgKC,
						event:        true,
					},
					{
						// NOTE(mtaufen): If you see a strange "expected assigned x but got assigned y" error on this case,
						// it is possible that the Kubelet didn't start the informer that watches the currently assigned
						// ConfigMap, or didn't get updates from that informer. Other tests don't always catch this because
						// they quickly change config. The sync loop will always happen once, a bit after the Kubelet starts
						// up, because other informers' initial "add" events can queue a sync. If you wait long enough before
						// changing config (waiting for the config to become last-known-good, for example), the syncs queued by
						// add events will have already been processed, and the lack of a running ConfigMap informer will result
						// in a missed update, no config change, and the above error when we check the status.
						desc:         "bad config",
						configSource: source,
						configMap:    badConfigMap,
						expectConfigStatus: expectNodeConfigStatus{
							lastKnownGood: lkgStatus,
							err:           status.LoadError,
							lkgActive:     true,
						},
						expectConfig: lkgKC,
						event:        true,
					},
				}

				// wait 12 minutes after setting the first config to ensure it has time to pass the trial duration
				testBothDirections(f, updateConfigMapFunc, &cases[0], cases[1:], 12*time.Minute)
			})
		})

		// Please note: This behavior is tested to ensure implementation correctness. We do not, however, recommend ConfigMap mutations
		// as a usage pattern for dynamic Kubelet config in large clusters. It is much safer to create a new ConfigMap, and incrementally
		// roll out a new Node.Spec.ConfigSource that references the new ConfigMap. In-place ConfigMap updates, including deletion
		// followed by re-creation, will cause all observing Kubelets to immediately restart for new config, because these operations
		// change the ResourceVersion of the ConfigMap.
		ginkgo.Context("delete and recreate ConfigMap: state transitions:", func() {
			ginkgo.It(itDescription, func() {
				var err error
				// we base the "correct" configmap off of the configuration from before the test
				correctKC := beforeKC.DeepCopy()
				correctConfigMap := newKubeletConfigMap("dynamic-kubelet-config-test-delete-createe", correctKC)
				correctConfigMap, err = f.ClientSet.CoreV1().ConfigMaps("kube-system").Create(correctConfigMap)
				framework.ExpectNoError(err)

				// we reuse the same name, namespace
				failParseConfigMap := correctConfigMap.DeepCopy()
				failParseConfigMap.Data = map[string]string{
					"kubelet": "{0xdeadbeef}",
				}

				// fail to validate, we make a copy and set an invalid KubeAPIQPS on kc before serializing
				invalidKC := correctKC.DeepCopy()
				invalidKC.KubeAPIQPS = -1
				failValidateConfigMap := correctConfigMap.DeepCopy()
				failValidateConfigMap.Data = newKubeletConfigMap("", invalidKC).Data

				// ensure node config source is set to the config map we will mutate in-place,
				// since recreateConfigMapFunc doesn't mutate Node.Spec.ConfigSource
				source := &v1.NodeConfigSource{
					ConfigMap: &v1.ConfigMapNodeConfigSource{
						Namespace:        correctConfigMap.Namespace,
						Name:             correctConfigMap.Name,
						KubeletConfigKey: "kubelet",
					},
				}
				(&nodeConfigTestCase{
					desc:         "initial state (correct)",
					configSource: source,
					configMap:    correctConfigMap,
					expectConfig: correctKC,
				}).run(f, setConfigSourceFunc, false, 0)

				cases := []nodeConfigTestCase{
					{
						desc:         "correct",
						configSource: source,
						configMap:    correctConfigMap,
						expectConfig: correctKC,
						event:        true,
					},
					{
						desc:         "fail-parse",
						configSource: source,
						configMap:    failParseConfigMap,
						expectConfigStatus: expectNodeConfigStatus{
							err:       status.LoadError,
							lkgActive: true,
						},
						expectConfig: localKC,
						event:        true,
					},
					{
						desc:         "fail-validate",
						configSource: source,
						configMap:    failValidateConfigMap,
						expectConfigStatus: expectNodeConfigStatus{
							err:       status.ValidateError,
							lkgActive: true,
						},
						expectConfig: localKC,
						event:        true,
					},
				}
				L := len(cases)
				for i := 1; i <= L; i++ { // need one less iteration than the number of cases
					testBothDirections(f, recreateConfigMapFunc, &cases[i-1 : i][0], cases[i:L], 0)
				}
			})
		})

		// Please note: This behavior is tested to ensure implementation correctness. We do not, however, recommend ConfigMap mutations
		// as a usage pattern for dynamic Kubelet config in large clusters. It is much safer to create a new ConfigMap, and incrementally
		// roll out a new Node.Spec.ConfigSource that references the new ConfigMap. In-place ConfigMap updates, including deletion
		// followed by re-creation, will cause all observing Kubelets to immediately restart for new config, because these operations
		// change the ResourceVersion of the ConfigMap.
		ginkgo.Context("delete and recreate ConfigMap: error while ConfigMap is absent:", func() {
			ginkgo.It(itDescription, func() {
				var err error
				// we base the "correct" configmap off of the configuration from before the test
				correctKC := beforeKC.DeepCopy()
				correctConfigMap := newKubeletConfigMap("dynamic-kubelet-config-test-delete-createe", correctKC)
				correctConfigMap, err = f.ClientSet.CoreV1().ConfigMaps("kube-system").Create(correctConfigMap)
				framework.ExpectNoError(err)

				// ensure node config source is set to the config map we will mutate in-place,
				// since our mutation functions don't mutate Node.Spec.ConfigSource
				source := &v1.NodeConfigSource{
					ConfigMap: &v1.ConfigMapNodeConfigSource{
						Namespace:        correctConfigMap.Namespace,
						Name:             correctConfigMap.Name,
						KubeletConfigKey: "kubelet",
					},
				}
				(&nodeConfigTestCase{
					desc:         "correct",
					configSource: source,
					configMap:    correctConfigMap,
					expectConfig: correctKC,
				}).run(f, setConfigSourceFunc, false, 0)

				// delete the ConfigMap, and ensure an error is reported by the Kubelet while the ConfigMap is absent
				(&nodeConfigTestCase{
					desc:         "correct",
					configSource: source,
					configMap:    correctConfigMap,
					expectConfigStatus: expectNodeConfigStatus{
						err: fmt.Sprintf(status.SyncErrorFmt, status.DownloadError),
					},
					expectConfig: correctKC,
				}).run(f, deleteConfigMapFunc, false, 0)

				// re-create the ConfigMap, and ensure the error disappears
				(&nodeConfigTestCase{
					desc:         "correct",
					configSource: source,
					configMap:    correctConfigMap,
					expectConfig: correctKC,
				}).run(f, createConfigMapFunc, false, 0)
			})
		})
	})
})

// testBothDirections tests the state change represented by each edge, where each case is a vertex,
// and there are edges in each direction between first and each of the cases.
func testBothDirections(f *framework.Framework, fn func(f *framework.Framework, tc *nodeConfigTestCase) error,
	first *nodeConfigTestCase, cases []nodeConfigTestCase, waitAfterFirst time.Duration) {
	// set to first and check that everything got set up properly
	ginkgo.By(fmt.Sprintf("setting initial state %q", first.desc))
	// we don't always expect an event here, because setting "first" might not represent
	// a change from the current configuration
	first.run(f, fn, false, waitAfterFirst)

	// for each case, set up, check expectations, then reset to first and check again
	for i := range cases {
		tc := &cases[i]
		ginkgo.By(fmt.Sprintf("from %q to %q", first.desc, tc.desc))
		// from first -> tc, tc.event fully describes whether we should get a config change event
		tc.run(f, fn, tc.event, 0)

		ginkgo.By(fmt.Sprintf("back to %q from %q", first.desc, tc.desc))
		// whether first -> tc should have produced a config change event partially determines whether tc -> first should produce an event
		first.run(f, fn, first.event && tc.event, 0)
	}
}

// run tests that, after performing fn, the node spec, status, configz, and latest event match
// the expectations described by state.
func (tc *nodeConfigTestCase) run(f *framework.Framework, fn func(f *framework.Framework, tc *nodeConfigTestCase) error,
	expectEvent bool, wait time.Duration) {
	// set the desired state, retry a few times in case we are competing with other editors
	gomega.Eventually(func() error {
		if err := fn(f, tc); err != nil {
			if len(tc.apierr) == 0 {
				return fmt.Errorf("case %s: expect nil error but got %q", tc.desc, err.Error())
			} else if !strings.Contains(err.Error(), tc.apierr) {
				return fmt.Errorf("case %s: expect error to contain %q but got %q", tc.desc, tc.apierr, err.Error())
			}
		} else if len(tc.apierr) > 0 {
			return fmt.Errorf("case %s: expect error to contain %q but got nil error", tc.desc, tc.apierr)
		}
		return nil
	}, time.Minute, time.Second).Should(gomega.BeNil())
	// skip further checks if we expected an API error
	if len(tc.apierr) > 0 {
		return
	}
	// wait for the designated duration before checking the reconciliation
	time.Sleep(wait)
	// check config source
	tc.checkNodeConfigSource(f)
	// check status
	tc.checkConfigStatus(f)
	// check that the Kubelet's config-related metrics are correct
	tc.checkConfigMetrics(f)
	// check expectConfig
	if tc.expectConfig != nil {
		tc.checkConfig(f)
	}
	// check that an event was sent for the config change
	if expectEvent {
		tc.checkEvent(f)
	}
}

// setConfigSourceFunc sets Node.Spec.ConfigSource to tc.configSource
func setConfigSourceFunc(f *framework.Framework, tc *nodeConfigTestCase) error {
	return setNodeConfigSource(f, tc.configSource)
}

// updateConfigMapFunc updates the ConfigMap described by tc.configMap to contain matching data.
// It also updates the resourceVersion in any non-nil NodeConfigSource.ConfigMap in the expected
// status to match the resourceVersion of the updated ConfigMap.
func updateConfigMapFunc(f *framework.Framework, tc *nodeConfigTestCase) error {
	// Clear ResourceVersion from the ConfigMap objects we use to initiate mutations
	// so that we don't get 409 (conflict) responses. ConfigMaps always allow updates
	// (with respect to concurrency control) when you omit ResourceVersion.
	// We know that we won't perform concurrent updates during this test.
	tc.configMap.ResourceVersion = ""
	cm, err := f.ClientSet.CoreV1().ConfigMaps(tc.configMap.Namespace).Update(tc.configMap)
	if err != nil {
		return err
	}
	// update tc.configMap's ResourceVersion to match the updated ConfigMap, this makes
	// sure our derived status checks have up-to-date information
	tc.configMap.ResourceVersion = cm.ResourceVersion
	return nil
}

// recreateConfigMapFunc deletes and recreates the ConfigMap described by tc.configMap.
// The new ConfigMap will match tc.configMap.
func recreateConfigMapFunc(f *framework.Framework, tc *nodeConfigTestCase) error {
	// need to ignore NotFound error, since there could be cases where delete
	// fails during a retry because the delete in a previous attempt succeeded,
	// before some other error occurred.
	err := deleteConfigMapFunc(f, tc)
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return createConfigMapFunc(f, tc)
}

// deleteConfigMapFunc simply deletes tc.configMap
func deleteConfigMapFunc(f *framework.Framework, tc *nodeConfigTestCase) error {
	return f.ClientSet.CoreV1().ConfigMaps(tc.configMap.Namespace).Delete(tc.configMap.Name, &metav1.DeleteOptions{})
}

// createConfigMapFunc creates tc.configMap and updates the UID and ResourceVersion on tc.configMap
// to match the created configMap
func createConfigMapFunc(f *framework.Framework, tc *nodeConfigTestCase) error {
	tc.configMap.ResourceVersion = ""
	cm, err := f.ClientSet.CoreV1().ConfigMaps(tc.configMap.Namespace).Create(tc.configMap)
	if err != nil {
		return err
	}
	// update tc.configMap's UID and ResourceVersion to match the new ConfigMap, this makes
	// sure our derived status checks have up-to-date information
	tc.configMap.UID = cm.UID
	tc.configMap.ResourceVersion = cm.ResourceVersion
	return nil
}

// make sure the node's config source matches what we expect, after setting it
func (tc *nodeConfigTestCase) checkNodeConfigSource(f *framework.Framework) {
	const (
		timeout  = time.Minute
		interval = time.Second
	)
	gomega.Eventually(func() error {
		node, err := f.ClientSet.CoreV1().Nodes().Get(framework.TestContext.NodeName, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("checkNodeConfigSource: case %s: %v", tc.desc, err)
		}
		actual := node.Spec.ConfigSource
		if !apiequality.Semantic.DeepEqual(tc.configSource, actual) {
			return fmt.Errorf(spew.Sprintf("checkNodeConfigSource: case %s: expected %#v but got %#v", tc.desc, tc.configSource, actual))
		}
		return nil
	}, timeout, interval).Should(gomega.BeNil())
}

// make sure the node status eventually matches what we expect
func (tc *nodeConfigTestCase) checkConfigStatus(f *framework.Framework) {
	const (
		timeout  = time.Minute
		interval = time.Second
	)
	errFmt := fmt.Sprintf("checkConfigStatus: case %s:", tc.desc) + " %v"
	gomega.Eventually(func() error {
		node, err := f.ClientSet.CoreV1().Nodes().Get(framework.TestContext.NodeName, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf(errFmt, err)
		}
		if err := expectConfigStatus(tc, node.Status.Config); err != nil {
			return fmt.Errorf(errFmt, err)
		}
		return nil
	}, timeout, interval).Should(gomega.BeNil())
}

func expectConfigStatus(tc *nodeConfigTestCase, actual *v1.NodeConfigStatus) error {
	var errs []string
	if actual == nil {
		return fmt.Errorf("expectConfigStatus requires actual to be non-nil (possible Kubelet failed to update status)")
	}
	// check Assigned matches tc.configSource, with UID and ResourceVersion from tc.configMap
	expectAssigned := tc.configSource.DeepCopy()
	if expectAssigned != nil && expectAssigned.ConfigMap != nil {
		expectAssigned.ConfigMap.UID = tc.configMap.UID
		expectAssigned.ConfigMap.ResourceVersion = tc.configMap.ResourceVersion
	}
	if !apiequality.Semantic.DeepEqual(expectAssigned, actual.Assigned) {
		errs = append(errs, spew.Sprintf("expected Assigned %#v but got %#v", expectAssigned, actual.Assigned))
	}
	// check LastKnownGood matches tc.expectConfigStatus.lastKnownGood
	if !apiequality.Semantic.DeepEqual(tc.expectConfigStatus.lastKnownGood, actual.LastKnownGood) {
		errs = append(errs, spew.Sprintf("expected LastKnownGood %#v but got %#v", tc.expectConfigStatus.lastKnownGood, actual.LastKnownGood))
	}
	// check Active matches Assigned or LastKnownGood, depending on tc.expectConfigStatus.lkgActive
	expectActive := expectAssigned
	if tc.expectConfigStatus.lkgActive {
		expectActive = tc.expectConfigStatus.lastKnownGood
	}
	if !apiequality.Semantic.DeepEqual(expectActive, actual.Active) {
		errs = append(errs, spew.Sprintf("expected Active %#v but got %#v", expectActive, actual.Active))
	}
	// check Error
	if tc.expectConfigStatus.err != actual.Error {
		errs = append(errs, fmt.Sprintf("expected Error %q but got %q", tc.expectConfigStatus.err, actual.Error))
	}
	// format error list
	if len(errs) > 0 {
		return fmt.Errorf("%s", strings.Join(errs, ", "))
	}
	return nil
}

// make sure config exposed on configz matches what we expect
func (tc *nodeConfigTestCase) checkConfig(f *framework.Framework) {
	const (
		timeout  = time.Minute
		interval = time.Second
	)
	gomega.Eventually(func() error {
		actual, err := getCurrentKubeletConfig()
		if err != nil {
			return fmt.Errorf("checkConfig: case %s: %v", tc.desc, err)
		}
		if !apiequality.Semantic.DeepEqual(tc.expectConfig, actual) {
			return fmt.Errorf(spew.Sprintf("checkConfig: case %s: expected %#v but got %#v", tc.desc, tc.expectConfig, actual))
		}
		return nil
	}, timeout, interval).Should(gomega.BeNil())
}

// checkEvent makes sure an event was sent marking the Kubelet's restart to use new config,
// and that it mentions the config we expect.
func (tc *nodeConfigTestCase) checkEvent(f *framework.Framework) {
	const (
		timeout  = time.Minute
		interval = time.Second
	)
	gomega.Eventually(func() error {
		events, err := f.ClientSet.CoreV1().Events("").List(metav1.ListOptions{})
		if err != nil {
			return fmt.Errorf("checkEvent: case %s: %v", tc.desc, err)
		}
		// find config changed event with most recent timestamp
		var recent *v1.Event
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
			return fmt.Errorf("checkEvent: case %s: no events found with reason %s", tc.desc, controller.KubeletConfigChangedEventReason)
		}
		// construct expected message, based on the test case
		expectMessage := controller.LocalEventMessage
		if tc.configSource != nil {
			if tc.configSource.ConfigMap != nil {
				expectMessage = fmt.Sprintf(controller.RemoteEventMessageFmt,
					fmt.Sprintf("/api/v1/namespaces/%s/configmaps/%s", tc.configSource.ConfigMap.Namespace, tc.configSource.ConfigMap.Name),
					tc.configMap.UID, tc.configMap.ResourceVersion, tc.configSource.ConfigMap.KubeletConfigKey)
			}
		}
		// compare messages
		if expectMessage != recent.Message {
			return fmt.Errorf("checkEvent: case %s: expected event message %q but got %q", tc.desc, expectMessage, recent.Message)
		}
		return nil
	}, timeout, interval).Should(gomega.BeNil())
}

// checkConfigMetrics makes sure the Kubelet's config related metrics are as we expect, given the test case
func (tc *nodeConfigTestCase) checkConfigMetrics(f *framework.Framework) {
	const (
		timeout                = time.Minute
		interval               = time.Second
		assignedConfigKey      = metrics.KubeletSubsystem + "_" + metrics.AssignedConfigKey
		activeConfigKey        = metrics.KubeletSubsystem + "_" + metrics.ActiveConfigKey
		lastKnownGoodConfigKey = metrics.KubeletSubsystem + "_" + metrics.LastKnownGoodConfigKey
		configErrorKey         = metrics.KubeletSubsystem + "_" + metrics.ConfigErrorKey
	)
	// local config helper
	mkLocalSample := func(name model.LabelValue) *model.Sample {
		return &model.Sample{
			Metric: model.Metric(map[model.LabelName]model.LabelValue{
				model.MetricNameLabel:                 name,
				metrics.ConfigSourceLabelKey:          metrics.ConfigSourceLabelValueLocal,
				metrics.ConfigUIDLabelKey:             "",
				metrics.ConfigResourceVersionLabelKey: "",
				metrics.KubeletConfigKeyLabelKey:      "",
			}),
			Value: 1,
		}
	}
	// remote config helper
	mkRemoteSample := func(name model.LabelValue, source *v1.NodeConfigSource) *model.Sample {
		return &model.Sample{
			Metric: model.Metric(map[model.LabelName]model.LabelValue{
				model.MetricNameLabel:                 name,
				metrics.ConfigSourceLabelKey:          model.LabelValue(fmt.Sprintf("/api/v1/namespaces/%s/configmaps/%s", source.ConfigMap.Namespace, source.ConfigMap.Name)),
				metrics.ConfigUIDLabelKey:             model.LabelValue(source.ConfigMap.UID),
				metrics.ConfigResourceVersionLabelKey: model.LabelValue(source.ConfigMap.ResourceVersion),
				metrics.KubeletConfigKeyLabelKey:      model.LabelValue(source.ConfigMap.KubeletConfigKey),
			}),
			Value: 1,
		}
	}
	// error helper
	mkErrorSample := func(expectError bool) *model.Sample {
		v := model.SampleValue(0)
		if expectError {
			v = model.SampleValue(1)
		}
		return &model.Sample{
			Metric: model.Metric(map[model.LabelName]model.LabelValue{model.MetricNameLabel: configErrorKey}),
			Value:  v,
		}
	}
	// construct expected metrics
	// assigned
	assignedSamples := model.Samples{mkLocalSample(assignedConfigKey)}
	assignedSource := tc.configSource.DeepCopy()
	if assignedSource != nil && assignedSource.ConfigMap != nil {
		assignedSource.ConfigMap.UID = tc.configMap.UID
		assignedSource.ConfigMap.ResourceVersion = tc.configMap.ResourceVersion
		assignedSamples = model.Samples{mkRemoteSample(assignedConfigKey, assignedSource)}
	}
	// last-known-good
	lastKnownGoodSamples := model.Samples{mkLocalSample(lastKnownGoodConfigKey)}
	lastKnownGoodSource := tc.expectConfigStatus.lastKnownGood
	if lastKnownGoodSource != nil && lastKnownGoodSource.ConfigMap != nil {
		lastKnownGoodSamples = model.Samples{mkRemoteSample(lastKnownGoodConfigKey, lastKnownGoodSource)}
	}
	// active
	activeSamples := model.Samples{mkLocalSample(activeConfigKey)}
	activeSource := assignedSource
	if tc.expectConfigStatus.lkgActive {
		activeSource = lastKnownGoodSource
	}
	if activeSource != nil && activeSource.ConfigMap != nil {
		activeSamples = model.Samples{mkRemoteSample(activeConfigKey, activeSource)}
	}
	// error
	errorSamples := model.Samples{mkErrorSample(len(tc.expectConfigStatus.err) > 0)}
	// expected metrics
	expect := frameworkmetrics.KubeletMetrics(map[string]model.Samples{
		assignedConfigKey:      assignedSamples,
		activeConfigKey:        activeSamples,
		lastKnownGoodConfigKey: lastKnownGoodSamples,
		configErrorKey:         errorSamples,
	})
	// wait for expected metrics to appear
	gomega.Eventually(func() error {
		actual, err := getKubeletMetrics(sets.NewString(
			assignedConfigKey,
			activeConfigKey,
			lastKnownGoodConfigKey,
			configErrorKey,
		))
		if err != nil {
			return err
		}
		// clear timestamps from actual, so DeepEqual is time-invariant
		for _, samples := range actual {
			for _, sample := range samples {
				sample.Timestamp = 0
			}
		}
		// compare to expected
		if !reflect.DeepEqual(expect, actual) {
			return fmt.Errorf("checkConfigMetrics: case: %s: expect metrics %s but got %s", tc.desc, spew.Sprintf("%#v", expect), spew.Sprintf("%#v", actual))
		}
		return nil
	}, timeout, interval).Should(gomega.BeNil())
}

// constructs the expected SelfLink for a config map
func configMapAPIPath(cm *v1.ConfigMap) string {
	return fmt.Sprintf("/api/v1/namespaces/%s/configmaps/%s", cm.Namespace, cm.Name)
}
