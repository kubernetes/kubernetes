/*
Copyright 2024 The Kubernetes Authors.

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

package e2enode

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/cpuset"
)

/*const (
	fakeExtendedResource = "dummy.com/dummy"
)*/

/*func patchNode(ctx context.Context, client clientset.Interface, old *v1.Node, new *v1.Node) error {
	oldData, err := json.Marshal(old)
	if err != nil {
		return err
	}

	newData, err := json.Marshal(new)
	if err != nil {
		return err
	}
	patchBytes, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, &v1.Node{})
	if err != nil {
		return fmt.Errorf("failed to create merge patch for node %q: %w", old.Name, err)
	}
	_, err = client.CoreV1().Nodes().Patch(ctx, old.Name, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{}, "status")
	return err
}

func addExtendedResource(clientSet clientset.Interface, nodeName, extendedResourceName string, extendedResourceQuantity resource.Quantity) {
	extendedResource := v1.ResourceName(extendedResourceName)

	ginkgo.By("Adding a custom resource")
	OriginalNode, err := clientSet.CoreV1().Nodes().Get(context.Background(), nodeName, metav1.GetOptions{})
	framework.ExpectNoError(err)

	node := OriginalNode.DeepCopy()
	node.Status.Capacity[extendedResource] = extendedResourceQuantity
	node.Status.Allocatable[extendedResource] = extendedResourceQuantity
	err = patchNode(context.Background(), clientSet, OriginalNode.DeepCopy(), node)
	framework.ExpectNoError(err)

	gomega.Eventually(func() error {
		node, err = clientSet.CoreV1().Nodes().Get(context.Background(), node.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		fakeResourceCapacity, exists := node.Status.Capacity[extendedResource]
		if !exists {
			return fmt.Errorf("node %s has no %s resource capacity", node.Name, extendedResourceName)
		}
		if expectedResource := resource.MustParse("123"); fakeResourceCapacity.Cmp(expectedResource) != 0 {
			return fmt.Errorf("node %s has resource capacity %s, expected: %s", node.Name, fakeResourceCapacity.String(), expectedResource.String())
		}

		return nil
	}).WithTimeout(30 * time.Second).WithPolling(time.Second).ShouldNot(gomega.HaveOccurred())
}

func removeExtendedResource(clientSet clientset.Interface, nodeName, extendedResourceName string) {
	extendedResource := v1.ResourceName(extendedResourceName)

	ginkgo.By("Removing a custom resource")
	originalNode, err := clientSet.CoreV1().Nodes().Get(context.Background(), nodeName, metav1.GetOptions{})
	framework.ExpectNoError(err)

	node := originalNode.DeepCopy()
	delete(node.Status.Capacity, extendedResource)
	delete(node.Status.Allocatable, extendedResource)
	err = patchNode(context.Background(), clientSet, originalNode.DeepCopy(), node)
	framework.ExpectNoError(err)

	gomega.Eventually(func() error {
		node, err = clientSet.CoreV1().Nodes().Get(context.Background(), nodeName, metav1.GetOptions{})
		framework.ExpectNoError(err)

		if _, exists := node.Status.Capacity[extendedResource]; exists {
			return fmt.Errorf("node %s has resource capacity %s which is expected to be removed", node.Name, extendedResourceName)
		}

		return nil
	}).WithTimeout(30 * time.Second).WithPolling(time.Second).ShouldNot(gomega.HaveOccurred())
}*/

func cpuManagerPolicyKubeletConfig(ctx context.Context, f *framework.Framework, oldCfg *kubeletconfig.KubeletConfiguration, cpuManagerPolicyName string, cpuManagerPolicyOptions map[string]string, isInPlacePodVerticalScalingAllocatedStatusEnabled bool, isInPlacePodVerticalScalingExclusiveCPUsEnabled bool) {
	if cpuManagerPolicyName != "" {
		if cpuManagerPolicyOptions != nil {
			func() {
				var cpuAlloc int64
				for policyOption, policyOptionValue := range cpuManagerPolicyOptions {
					if policyOption == cpumanager.FullPCPUsOnlyOption && policyOptionValue == "true" {
						_, cpuAlloc, _ = getLocalNodeCPUDetails(ctx, f)
						smtLevel := getSMTLevel()

						// strict SMT alignment is trivially verified and granted on non-SMT systems
						if smtLevel < 2 {
							e2eskipper.Skipf("Skipping Pod Resize along side CPU Manager %s tests since SMT disabled", policyOption)
						}

						// our tests want to allocate a full core, so we need at last 2*2=4 virtual cpus
						if cpuAlloc < int64(smtLevel*2) {
							e2eskipper.Skipf("Skipping Pod resize along side CPU Manager %s tests since the CPU capacity < 4", policyOption)
						}

						framework.Logf("SMT level %d", smtLevel)
						return
					}
				}
			}()

			// TODO: we assume the first available CPUID is 0, which is pretty fair, but we should probably
			// check what we do have in the node.
			newCfg := configureCPUManagerInKubelet(oldCfg,
				&cpuManagerKubeletArguments{
					policyName:              cpuManagerPolicyName,
					reservedSystemCPUs:      cpuset.New(0),
					enableCPUManagerOptions: true,
					options:                 cpuManagerPolicyOptions,
				},
				isInPlacePodVerticalScalingAllocatedStatusEnabled,
				isInPlacePodVerticalScalingExclusiveCPUsEnabled,
			)
			updateKubeletConfig(ctx, f, newCfg, true)
		} else {
			var cpuCap int64
			cpuCap, _, _ = getLocalNodeCPUDetails(ctx, f)
			// Skip CPU Manager tests altogether if the CPU capacity < 2.
			if cpuCap < 2 {
				e2eskipper.Skipf("Skipping Pod Resize alongside CPU Manager tests since the CPU capacity < 2")
			}
			// Enable CPU Manager in the kubelet.
			newCfg := configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
				policyName:         cpuManagerPolicyName,
				reservedSystemCPUs: cpuset.CPUSet{},
			}, isInPlacePodVerticalScalingAllocatedStatusEnabled, isInPlacePodVerticalScalingExclusiveCPUsEnabled)
			updateKubeletConfig(ctx, f, newCfg, true)
		}
	}
}

type cpuManagerPolicyConfig struct {
	name    string
	title   string
	options map[string]string
}

func doPodResizeExtendTests(policy cpuManagerPolicyConfig, isInPlacePodVerticalScalingAllocatedStatusEnabled bool, isInPlacePodVerticalScalingExclusiveCPUsEnabled bool) {
	f := framework.NewDefaultFramework("pod-resize-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var podClient *e2epod.PodClient
	var oldCfg *kubeletconfig.KubeletConfiguration
	ginkgo.BeforeEach(func(ctx context.Context) {
		var err error
		node := getLocalNode(ctx, f)
		if framework.NodeOSDistroIs("windows") || e2enode.IsARM64(node) {
			e2eskipper.Skipf("runtime does not support InPlacePodVerticalScaling -- skipping")
		}
		if isMultiNUMA() {
			e2eskipper.Skipf("For simple test, only test one NUMA, multi NUMA -- skipping")
		}
		podClient = e2epod.NewPodClient(f)
		if oldCfg == nil {
			oldCfg, err = getCurrentKubeletConfig(ctx)
			framework.ExpectNoError(err)
		}
	})

	type testCase struct {
		name        string
 		containers  []e2epod.ResizableContainerInfo
 		patchString string
 		expected    []e2epod.ResizableContainerInfo
        skipFlag    bool
	}

	setCPUsForTestCase := func(ctx context.Context, tests *testCase, fullPCPUsOnly string) {
		cpuCap, _, _ := getLocalNodeCPUDetails(ctx, f)
		firstContainerCpuset := cpuset.New()
		firstAdditionCpuset := cpuset.New()
		firstExpectedCpuset := cpuset.New()
		secondContainerCpuset := cpuset.New()
		secondAdditionCpuset := cpuset.New()
		secondExpectedCpuset := cpuset.New()

		if tests.name == "1 Guaranteed QoS pod, one container - increase CPU, FullPCPUsOnlyOption = false" {
			if cpuCap < 3 {
				tests.skipFlag = true
			}
			firstContainerCpuset = cpuset.New(1)
			if isHTEnabled() {
				cpuList := mustParseCPUSet(getCPUSiblingList(0)).List()
				firstContainerCpuset = cpuset.New(cpuList[1])
			}
			tests.containers[0].CPUsAllowedList = firstContainerCpuset.String()

			firstAdditionCpuset = cpuset.New(2)
			if isHTEnabled() {
				cpuList := mustParseCPUSet(getCPUSiblingList(1)).List()
				firstAdditionCpuset = cpuset.New(cpuList[0])
			}
			firstExpectedCpuset = firstAdditionCpuset.Union(firstContainerCpuset)
			tests.expected[0].CPUsAllowedList = firstExpectedCpuset.String()
		} else if tests.name == "1 Guaranteed QoS pod, two containers - increase CPU, FullPCPUsOnlyOption = false" {
			if cpuCap < 6 {
				tests.skipFlag = true
			}
			firstContainerCpuset = cpuset.New(1)
			if isHTEnabled() {
				cpuList := mustParseCPUSet(getCPUSiblingList(0)).List()
				firstContainerCpuset = cpuset.New(cpuList[1])
			}
			tests.containers[0].CPUsAllowedList = firstContainerCpuset.String()

			secondContainerCpuset = cpuset.New(1)
			if isHTEnabled() {
				cpuList := mustParseCPUSet(getCPUSiblingList(1)).List()
				secondContainerCpuset = cpuset.New(cpuList[0])
			}
			tests.containers[1].CPUsAllowedList = secondContainerCpuset.String()

			firstAdditionCpuset = cpuset.New(2)
			if isHTEnabled() {
				cpuList := mustParseCPUSet(getCPUSiblingList(1)).List()
				firstAdditionCpuset = cpuset.New(cpuList[1])
			}
			firstExpectedCpuset = firstAdditionCpuset.Union(firstContainerCpuset)
			tests.expected[0].CPUsAllowedList = firstExpectedCpuset.String()

			secondAdditionCpuset = cpuset.New(2)
			if isHTEnabled() {
				cpuList := mustParseCPUSet(getCPUSiblingList(2)).List()
				if len(cpuList) > 0 {
					secondAdditionCpuset = cpuset.New(cpuList[0])
				}
			}
			secondExpectedCpuset = secondAdditionCpuset.Union(secondContainerCpuset)
			tests.expected[1].CPUsAllowedList = secondExpectedCpuset.String()
		} else if (tests.name == "1 Guaranteed QoS pod, one container - decrease CPU, FullPCPUsOnlyOption = false") || (tests.name == "1 Guaranteed QoS pod, one container - decrease CPU with mustKeepCPUs, FullPCPUsOnlyOption = false") {
			if cpuCap < 3 {
				tests.skipFlag = true
			}
			firstContainerCpuset = cpuset.New(2, 3)
			if isHTEnabled() {
				cpuList := mustParseCPUSet(getCPUSiblingList(0)).List()
				if cpuList[1] != 1 {
					firstContainerCpuset = mustParseCPUSet(getCPUSiblingList(1))
				}
			}
			tests.containers[0].CPUsAllowedList = firstContainerCpuset.String()

			firstExpectedCpuset = cpuset.New(firstContainerCpuset.List()[0])
			tests.expected[0].CPUsAllowedList = firstExpectedCpuset.String()
			if tests.name == "1 Guaranteed QoS pod, one container - decrease CPU with mustKeepCPUs, FullPCPUsOnlyOption = false" {
				startIndex := strings.Index(tests.patchString, `"mustKeepCPUs","value": "`) + len(`"mustKeepCPUs","value": "`)
				endIndex := strings.Index(tests.patchString[startIndex:], `"`) + startIndex
				tests.expected[0].CPUsAllowedList = tests.patchString[startIndex:endIndex]
				ginkgo.By(fmt.Sprintf("startIndex:%d, endIndex:%d", startIndex, endIndex))
			}
		} else if (tests.name == "1 Guaranteed QoS pod, one container - decrease CPU, FullPCPUsOnlyOption = true") || (tests.name == "1 Guaranteed QoS pod, one container - decrease CPU with wrong mustKeepCPU, FullPCPUsOnlyOption = ture") || (tests.name == "1 Guaranteed QoS pod, one container - decrease CPU with correct mustKeepCPU, FullPCPUsOnlyOption = true") {
			if cpuCap < 6 {
				tests.skipFlag = true
			}
			firstContainerCpuset = cpuset.New(2, 3, 4, 5)
			if isHTEnabled() {
				cpuList := mustParseCPUSet(getCPUSiblingList(0)).List()
				if cpuList[1] != 1 {
					firstContainerCpuset = mustParseCPUSet(getCPUSiblingList(1))
					firstContainerCpuset = firstContainerCpuset.Union(mustParseCPUSet(getCPUSiblingList(2)))
				}
			}
			tests.containers[0].CPUsAllowedList = firstContainerCpuset.String()

			firstExpectedCpuset = mustParseCPUSet(getCPUSiblingList(1))
			tests.expected[0].CPUsAllowedList = firstExpectedCpuset.String()
			if tests.name == "1 Guaranteed QoS pod, one container - decrease CPU with correct mustKeepCPU, FullPCPUsOnlyOption = true" {
				startIndex := strings.Index(tests.patchString, `"mustKeepCPUs","value": "`) + len(`"mustKeepCPUs","value": "`)
				endIndex := strings.Index(tests.patchString[startIndex:], `"`) + startIndex
				tests.expected[0].CPUsAllowedList = tests.patchString[startIndex:endIndex]
				ginkgo.By(fmt.Sprintf("startIndex:%d, endIndex:%d", startIndex, endIndex))
			}
		}

		ginkgo.By(fmt.Sprintf("firstContainerCpuset:%v, firstAdditionCpuset:%v, firstExpectedCpuset:%v", firstContainerCpuset, firstAdditionCpuset, firstExpectedCpuset))
		ginkgo.By(fmt.Sprintf("secondContainerCpuset:%v, secondAdditionCpuset:%v, secondExpectedCpuset:%v", secondContainerCpuset, secondAdditionCpuset, secondExpectedCpuset))
	}

	noRestart := v1.NotRequired
	testsWithFalseFullCPUs := []testCase{
		{
			name: "1 Guaranteed QoS pod, one container - increase CPU, FullPCPUsOnlyOption = false",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:                 "c1",
					Resources:            &e2epod.ContainerResources{CPUReq: "1", CPULim: "1", MemReq: "200Mi", MemLim: "200Mi"},
					CPUPolicy:            &noRestart,
					MemPolicy:            &noRestart,
					CPUsAllowedListValue: "1",
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"2"},"limits":{"cpu":"2"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:                 "c1",
					Resources:            &e2epod.ContainerResources{CPUReq: "2", CPULim: "2", MemReq: "200Mi", MemLim: "200Mi"},
					CPUPolicy:            &noRestart,
					MemPolicy:            &noRestart,
					CPUsAllowedListValue: "2",
				},
			},
		},
		{
			name: "1 Guaranteed QoS pod, two containers - increase CPU, FullPCPUsOnlyOption = false",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:                 "c1",
					Resources:            &e2epod.ContainerResources{CPUReq: "1", CPULim: "1", MemReq: "200Mi", MemLim: "200Mi"},
					CPUPolicy:            &noRestart,
					MemPolicy:            &noRestart,
					CPUsAllowedListValue: "1",
				},
				{
					Name:                 "c2",
					Resources:            &e2epod.ContainerResources{CPUReq: "1", CPULim: "1", MemReq: "200Mi", MemLim: "200Mi"},
					CPUPolicy:            &noRestart,
					MemPolicy:            &noRestart,
					CPUsAllowedListValue: "1",
				},
			},
			patchString: `{"spec":{"containers":[
                        {"name":"c1",  "resources":{"requests":{"cpu":"2"},"limits":{"cpu":"2"}}},
                        {"name":"c2",  "resources":{"requests":{"cpu":"2"},"limits":{"cpu":"2"}}}
                    ]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:                 "c1",
					Resources:            &e2epod.ContainerResources{CPUReq: "2", CPULim: "2", MemReq: "200Mi", MemLim: "200Mi"},
					CPUPolicy:            &noRestart,
					MemPolicy:            &noRestart,
					CPUsAllowedListValue: "2",
				},
				{
					Name:                 "c2",
					Resources:            &e2epod.ContainerResources{CPUReq: "2", CPULim: "2", MemReq: "200Mi", MemLim: "200Mi"},
					CPUPolicy:            &noRestart,
					MemPolicy:            &noRestart,
					CPUsAllowedListValue: "2",
				},
			},
		},
		{
			name: "1 Guaranteed QoS pod, one container - decrease CPU, FullPCPUsOnlyOption = false",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:                 "c1",
					Resources:            &e2epod.ContainerResources{CPUReq: "2", CPULim: "2", MemReq: "400Mi", MemLim: "400Mi"},
					CPUPolicy:            &noRestart,
					MemPolicy:            &noRestart,
					CPUsAllowedListValue: "2",
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"1"},"limits":{"cpu":"1"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:                 "c1",
					Resources:            &e2epod.ContainerResources{CPUReq: "1", CPULim: "1", MemReq: "400Mi", MemLim: "400Mi"},
					CPUPolicy:            &noRestart,
					MemPolicy:            &noRestart,
					CPUsAllowedListValue: "1",
				},
			},
		},
		{
			name: "1 Guaranteed QoS pod, one container - decrease CPU with mustKeepCPUs, FullPCPUsOnlyOption = false",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:                 "c1",
					Resources:            &e2epod.ContainerResources{CPUReq: "2", CPULim: "2", MemReq: "200Mi", MemLim: "200Mi"},
					CPUPolicy:            &noRestart,
					MemPolicy:            &noRestart,
					CPUsAllowedListValue: "2",
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "env":[{"name":"mustKeepCPUs","value": "11"}], "resources":{"requests":{"cpu":"1"},"limits":{"cpu":"1"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:                 "c1",
					Resources:            &e2epod.ContainerResources{CPUReq: "1", CPULim: "1", MemReq: "200Mi", MemLim: "200Mi"},
					CPUPolicy:            &noRestart,
					MemPolicy:            &noRestart,
					CPUsAllowedListValue: "1",
				},
			},
		},
	}

	testsWithTrueFullCPUs := []testCase{
		{
			name: "1 Guaranteed QoS pod, one container - decrease CPU, FullPCPUsOnlyOption = true",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:                 "c1",
					Resources:            &e2epod.ContainerResources{CPUReq: "4", CPULim: "4", MemReq: "400Mi", MemLim: "400Mi"},
					CPUPolicy:            &noRestart,
					MemPolicy:            &noRestart,
					CPUsAllowedListValue: "4",
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"2"},"limits":{"cpu":"2"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:                 "c1",
					Resources:            &e2epod.ContainerResources{CPUReq: "2", CPULim: "2", MemReq: "400Mi", MemLim: "400Mi"},
					CPUPolicy:            &noRestart,
					MemPolicy:            &noRestart,
					CPUsAllowedListValue: "2",
				},
			},
		},
		{
			name: "1 Guaranteed QoS pod, one container - decrease CPU with correct mustKeepCPU, FullPCPUsOnlyOption = true",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:                 "c1",
					Resources:            &e2epod.ContainerResources{CPUReq: "4", CPULim: "4", MemReq: "200Mi", MemLim: "200Mi"},
					CPUPolicy:            &noRestart,
					MemPolicy:            &noRestart,
					CPUsAllowedListValue: "4",
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "env":[{"name":"mustKeepCPUs","value": "2,12"}], "resources":{"requests":{"cpu":"2"},"limits":{"cpu":"2"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:                 "c1",
					Resources:            &e2epod.ContainerResources{CPUReq: "2", CPULim: "2", MemReq: "200Mi", MemLim: "200Mi"},
					CPUPolicy:            &noRestart,
					MemPolicy:            &noRestart,
					CPUsAllowedListValue: "2",
				},
			},
		},
		// Abnormal case, CPUs in mustKeepCPUs not full PCPUs, the mustKeepCPUs will be ignored
		{
			name: "1 Guaranteed QoS pod, one container - decrease CPU with wrong mustKeepCPU, FullPCPUsOnlyOption = ture",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:                 "c1",
					Resources:            &e2epod.ContainerResources{CPUReq: "4", CPULim: "4", MemReq: "200Mi", MemLim: "200Mi"},
					CPUPolicy:            &noRestart,
					MemPolicy:            &noRestart,
					CPUsAllowedListValue: "4",
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "env":[{"name":"mustKeepCPUs","value": "1,2"}], "resources":{"requests":{"cpu":"2"},"limits":{"cpu":"2"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:                 "c1",
					Resources:            &e2epod.ContainerResources{CPUReq: "2", CPULim: "2", MemReq: "200Mi", MemLim: "200Mi"},
					CPUPolicy:            &noRestart,
					MemPolicy:            &noRestart,
					CPUsAllowedListValue: "2",
				},
			},
		},
	}

	timeouts := framework.NewTimeoutContext()

	var tests []testCase
	if policy.options[cpumanager.FullPCPUsOnlyOption] == "false" {
		tests = testsWithFalseFullCPUs
	} else if policy.options[cpumanager.FullPCPUsOnlyOption] == "true" {
		tests = testsWithTrueFullCPUs
	}

	for idx := range tests {
		tc := tests[idx]
		ginkgo.It(tc.name+policy.title+" (InPlacePodVerticalScalingAllocatedStatus="+strconv.FormatBool(isInPlacePodVerticalScalingAllocatedStatusEnabled)+", InPlacePodVerticalScalingExclusiveCPUs="+strconv.FormatBool(isInPlacePodVerticalScalingExclusiveCPUsEnabled)+")", func(ctx context.Context) {
			cpuManagerPolicyKubeletConfig(ctx, f, oldCfg, policy.name, policy.options, isInPlacePodVerticalScalingAllocatedStatusEnabled, isInPlacePodVerticalScalingExclusiveCPUsEnabled)

			setCPUsForTestCase(ctx, &tc, policy.options[cpumanager.FullPCPUsOnlyOption])
			if tc.skipFlag {
				e2eskipper.Skipf("Skipping CPU Manager tests since the CPU not enough")
			}

			var testPod, patchedPod *v1.Pod
			var pErr error

			tStamp := strconv.Itoa(time.Now().Nanosecond())
			testPod = e2epod.MakePodWithResizableContainers(f.Namespace.Name, "testpod", tStamp, tc.containers)
			testPod.GenerateName = "resize-test-"
			testPod = e2epod.MustMixinRestrictedPodSecurity(testPod)

			/*if tc.addExtendedResource {
				nodes, err := e2enode.GetReadySchedulableNodes(context.Background(), f.ClientSet)
				framework.ExpectNoError(err)

				for _, node := range nodes.Items {
					addExtendedResource(f.ClientSet, node.Name, fakeExtendedResource, resource.MustParse("123"))
				}
				defer func() {
					for _, node := range nodes.Items {
						removeExtendedResource(f.ClientSet, node.Name, fakeExtendedResource)
					}
				}()
			}*/

			ginkgo.By("creating pod")
			newPod := podClient.CreateSync(ctx, testPod)

			ginkgo.By("verifying initial pod resources, allocations are as expected")
			e2epod.VerifyPodResources(newPod, tc.containers)
			ginkgo.By("verifying initial pod resize policy is as expected")
			e2epod.VerifyPodResizePolicy(newPod, tc.containers)

			ginkgo.By("verifying initial pod status resources are as expected")
			framework.ExpectNoError(e2epod.VerifyPodStatusResources(newPod, tc.containers))
			ginkgo.By("verifying initial cgroup config are as expected")
			framework.ExpectNoError(e2epod.VerifyPodContainersCgroupValues(ctx, f, newPod, tc.containers))
			// TODO make this dynamic depending on Policy Name, Resources input and topology of target
			// machine.
			// For the moment skip below if CPU Manager Policy is set to none
			if policy.name == string(cpumanager.PolicyStatic) {
				ginkgo.By("verifying initial pod Cpus allowed list value")
				gomega.Eventually(ctx, e2epod.VerifyPodContainersCPUsAllowedListValue, timeouts.PodStartShort, timeouts.Poll).
					WithArguments(f, newPod, tc.containers).
					Should(gomega.Succeed(), "failed to verify initial Pod CPUsAllowedListValue")
			}

			patchAndVerify := func(patchString string, expectedContainers []e2epod.ResizableContainerInfo, initialContainers []e2epod.ResizableContainerInfo, opStr string, isRollback bool) {
				ginkgo.By(fmt.Sprintf("patching pod for %s", opStr))
				patchedPod, pErr = f.ClientSet.CoreV1().Pods(newPod.Namespace).Patch(ctx, newPod.Name,
					types.StrategicMergePatchType, []byte(patchString), metav1.PatchOptions{}, "resize")
				framework.ExpectNoError(pErr, fmt.Sprintf("failed to patch pod for %s", opStr))

				ginkgo.By(fmt.Sprintf("verifying pod patched for %s", opStr))
				e2epod.VerifyPodResources(patchedPod, expectedContainers)

				ginkgo.By(fmt.Sprintf("waiting for %s to be actuated", opStr))
				resizedPod := e2epod.WaitForPodResizeActuation(ctx, f, podClient, newPod)
				e2epod.ExpectPodResized(ctx, f, resizedPod, expectedContainers)

				// Check cgroup values only for containerd versions before 1.6.9
				ginkgo.By(fmt.Sprintf("verifying pod container's cgroup values after %s", opStr))
				framework.ExpectNoError(e2epod.VerifyPodContainersCgroupValues(ctx, f, resizedPod, expectedContainers))

				ginkgo.By(fmt.Sprintf("verifying pod resources after %s", opStr))
				e2epod.VerifyPodResources(resizedPod, expectedContainers)

				// TODO make this dynamic depending on Policy Name, Resources input and topology of target
				// machine.
				// For the moment skip below if CPU Manager Policy is set to none
				if policy.name == string(cpumanager.PolicyStatic) {
					ginkgo.By(fmt.Sprintf("verifying pod Cpus allowed list value after %s", opStr))
					if isInPlacePodVerticalScalingExclusiveCPUsEnabled {
						gomega.Eventually(ctx, e2epod.VerifyPodContainersCPUsAllowedListValue, timeouts.PodStartShort, timeouts.Poll).
							WithArguments(f, resizedPod, expectedContainers).
							Should(gomega.Succeed(), "failed to verify Pod CPUsAllowedListValue for resizedPod with InPlacePodVerticalScalingExclusiveCPUs enabled")
					} else {
						gomega.Eventually(ctx, e2epod.VerifyPodContainersCPUsAllowedListValue, timeouts.PodStartShort, timeouts.Poll).
							WithArguments(f, resizedPod, tc.containers).
							Should(gomega.Succeed(), "failed to verify Pod CPUsAllowedListValue for resizedPod with InPlacePodVerticalScalingExclusiveCPUs disabled (default)")
					}
				}
			}

			ginkgo.By("First patch")
			patchAndVerify(tc.patchString, tc.expected, tc.containers, "resize", false)

			rbPatchStr, err := e2epod.ResizeContainerPatch(tc.containers)
			framework.ExpectNoError(err)
			// Resize has been actuated, test rollback
			ginkgo.By("Second patch for rollback")
			patchAndVerify(rbPatchStr, tc.containers, tc.expected, "rollback", true)

			ginkgo.By("deleting pod")
			deletePodSyncByName(ctx, f, newPod.Name)
			// we need to wait for all containers to really be gone so cpumanager reconcile loop will not rewrite the cpu_manager_state.
			// this is in turn needed because we will have an unavoidable (in the current framework) race with the
			// reconcile loop which will make our attempt to delete the state file and to restore the old config go haywire
			waitForAllContainerRemoval(ctx, newPod.Name, newPod.Namespace)
		})
	}

	ginkgo.AfterEach(func(ctx context.Context) {
		if oldCfg != nil {
			updateKubeletConfig(ctx, f, oldCfg, true)
		}
	})

}

func doMultiPodResizeTests(policy cpuManagerPolicyConfig, isInPlacePodVerticalScalingAllocatedStatusEnabled bool, isInPlacePodVerticalScalingExclusiveCPUsEnabled bool) {
	f := framework.NewDefaultFramework("pod-resize-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var podClient *e2epod.PodClient
	var oldCfg *kubeletconfig.KubeletConfiguration
	ginkgo.BeforeEach(func(ctx context.Context) {
		var err error
		node := getLocalNode(ctx, f)
		if framework.NodeOSDistroIs("windows") || e2enode.IsARM64(node) {
			e2eskipper.Skipf("runtime does not support InPlacePodVerticalScaling -- skipping")
		}
		podClient = e2epod.NewPodClient(f)
		if oldCfg == nil {
			oldCfg, err = getCurrentKubeletConfig(ctx)
			framework.ExpectNoError(err)
		}
	})

	type testPod struct {
		containers  []e2epod.ResizableContainerInfo
		patchString string
		expected    []e2epod.ResizableContainerInfo
	}

	type testCase struct {
		name     string
		testPod1 testPod
		testPod2 testPod
		skipFlag bool
	}

	setCPUsForTestCase := func(ctx context.Context, tests *testCase, fullPCPUsOnly string) {
		cpuCap, _, _ := getLocalNodeCPUDetails(ctx, f)
		firstContainerCpuset := cpuset.New()
		firstAdditionCpuset := cpuset.New()
		firstExpectedCpuset := cpuset.New()
		secondContainerCpuset := cpuset.New()
		secondAdditionCpuset := cpuset.New()
		secondExpectedCpuset := cpuset.New()

		if tests.name == "2 Guaranteed QoS pod, one container - increase CPU, FullPCPUsOnlyOption = false" {
			// In cpuManagerPolicyKubeletConfig, reservedSystemCPUs is 0
			if cpuCap < 6 {
				tests.skipFlag = true
			}
			firstContainerCpuset = cpuset.New(1)
			if isHTEnabled() {
				cpuList := mustParseCPUSet(getCPUSiblingList(0)).List()
				firstContainerCpuset = cpuset.New(cpuList[1])
			}
			tests.testPod1.containers[0].CPUsAllowedList = firstContainerCpuset.String()

			secondContainerCpuset = cpuset.New(1)
			if isHTEnabled() {
				cpuList := mustParseCPUSet(getCPUSiblingList(1)).List()
				secondContainerCpuset = cpuset.New(cpuList[0])
			}
			tests.testPod2.containers[0].CPUsAllowedList = secondContainerCpuset.String()

			firstAdditionCpuset = cpuset.New(2)
			if isHTEnabled() {
				cpuList := mustParseCPUSet(getCPUSiblingList(1)).List()
				firstAdditionCpuset = cpuset.New(cpuList[1])
			}
			firstExpectedCpuset = firstAdditionCpuset.Union(firstContainerCpuset)
			tests.testPod1.expected[0].CPUsAllowedList = firstExpectedCpuset.String()

			secondAdditionCpuset = cpuset.New(2)
			if isHTEnabled() {
				cpuList := mustParseCPUSet(getCPUSiblingList(2)).List()
				if len(cpuList) > 0 {
					secondAdditionCpuset = cpuset.New(cpuList[0])
				}
			}
			secondExpectedCpuset = secondAdditionCpuset.Union(secondContainerCpuset)
			tests.testPod2.expected[0].CPUsAllowedList = secondExpectedCpuset.String()
		}
		ginkgo.By(fmt.Sprintf("firstContainerCpuset:%v, firstAdditionCpuset:%v, firstExpectedCpuset:%v", firstContainerCpuset, firstAdditionCpuset, firstExpectedCpuset))
		ginkgo.By(fmt.Sprintf("secondContainerCpuset:%v, secondAdditionCpuset:%v, secondExpectedCpuset:%v", secondContainerCpuset, secondAdditionCpuset, secondExpectedCpuset))
	}

	noRestart := v1.NotRequired
	tests := []testCase{
		{
			name: "2 Guaranteed QoS pod, one container - increase CPU, FullPCPUsOnlyOption = false",
			testPod1: testPod{
				containers: []e2epod.ResizableContainerInfo{
					{
						Name:                 "c1",
						Resources:            &e2epod.ContainerResources{CPUReq: "1", CPULim: "1", MemReq: "200Mi", MemLim: "200Mi"},
						CPUPolicy:            &noRestart,
						MemPolicy:            &noRestart,
						CPUsAllowedListValue: "1",
					},
				},
				patchString: `{"spec":{"containers":[
							{"name":"c1", "resources":{"requests":{"cpu":"2"},"limits":{"cpu":"2"}}}
						]}}`,
				expected: []e2epod.ResizableContainerInfo{
					{
						Name:                 "c1",
						Resources:            &e2epod.ContainerResources{CPUReq: "2", CPULim: "2", MemReq: "200Mi", MemLim: "200Mi"},
						CPUPolicy:            &noRestart,
						MemPolicy:            &noRestart,
						CPUsAllowedListValue: "2",
					},
				},
			},
			testPod2: testPod{
				containers: []e2epod.ResizableContainerInfo{
					{
						Name:                 "c2",
						Resources:            &e2epod.ContainerResources{CPUReq: "1", CPULim: "1", MemReq: "200Mi", MemLim: "200Mi"},
						CPUPolicy:            &noRestart,
						MemPolicy:            &noRestart,
						CPUsAllowedListValue: "1",
					},
				},
				patchString: `{"spec":{"containers":[
							{"name":"c2", "resources":{"requests":{"cpu":"2"},"limits":{"cpu":"2"}}}
						]}}`,
				expected: []e2epod.ResizableContainerInfo{
					{
						Name:                 "c2",
						Resources:            &e2epod.ContainerResources{CPUReq: "2", CPULim: "2", MemReq: "200Mi", MemLim: "200Mi"},
						CPUPolicy:            &noRestart,
						MemPolicy:            &noRestart,
						CPUsAllowedListValue: "2",
					},
				},
			},
		},
	}

	timeouts := framework.NewTimeoutContext()

	for idx := range tests {
		tc := tests[idx]
		ginkgo.It(tc.name+policy.title+" (InPlacePodVerticalScalingAllocatedStatus="+strconv.FormatBool(isInPlacePodVerticalScalingAllocatedStatusEnabled)+", InPlacePodVerticalScalingExclusiveCPUs="+strconv.FormatBool(isInPlacePodVerticalScalingExclusiveCPUsEnabled)+")", func(ctx context.Context) {
			cpuManagerPolicyKubeletConfig(ctx, f, oldCfg, policy.name, policy.options, isInPlacePodVerticalScalingAllocatedStatusEnabled, isInPlacePodVerticalScalingExclusiveCPUsEnabled)

			setCPUsForTestCase(ctx, &tc, policy.options[cpumanager.FullPCPUsOnlyOption])
			if tc.skipFlag {
				e2eskipper.Skipf("Skipping CPU Manager tests since the CPU not enough")
			}

			var patchedPod *v1.Pod
			var pErr error

			createAndVerify := func(podName string, podClient *e2epod.PodClient, testContainers []e2epod.ResizableContainerInfo) (newPod *v1.Pod) {
				var testPod *v1.Pod

				tStamp := strconv.Itoa(time.Now().Nanosecond())
				testPod = e2epod.MakePodWithResizableContainers(f.Namespace.Name, fmt.Sprintf("resizepod-%s", podName), tStamp, testContainers)
				testPod.GenerateName = "resize-test-"
				testPod = e2epod.MustMixinRestrictedPodSecurity(testPod)

				ginkgo.By("creating pod")
				newPod = podClient.CreateSync(ctx, testPod)

				ginkgo.By("verifying initial pod resources, allocations are as expected")
				e2epod.VerifyPodResources(newPod, testContainers)
				ginkgo.By("verifying initial pod resize policy is as expected")
				e2epod.VerifyPodResizePolicy(newPod, testContainers)

				ginkgo.By("verifying initial pod status resources are as expected")
				framework.ExpectNoError(e2epod.VerifyPodStatusResources(newPod, testContainers))
				ginkgo.By("verifying initial cgroup config are as expected")
				framework.ExpectNoError(e2epod.VerifyPodContainersCgroupValues(ctx, f, newPod, testContainers))
				// TODO make this dynamic depending on Policy Name, Resources input and topology of target
				// machine.
				// For the moment skip below if CPU Manager Policy is set to none
				if policy.name == string(cpumanager.PolicyStatic) {
					ginkgo.By("verifying initial pod Cpus allowed list value")
					gomega.Eventually(ctx, e2epod.VerifyPodContainersCPUsAllowedListValue, timeouts.PodStartShort, timeouts.Poll).
						WithArguments(f, newPod, testContainers).
						Should(gomega.Succeed(), "failed to verify initial Pod CPUsAllowedListValue")
				}
				return newPod
			}

			newPod1 := createAndVerify("testpod1", podClient, tc.testPod1.containers)
			newPod2 := createAndVerify("testpod2", podClient, tc.testPod2.containers)

			patchAndVerify := func(patchString string, expectedContainers []e2epod.ResizableContainerInfo, initialContainers []e2epod.ResizableContainerInfo, opStr string, isRollback bool, newPod *v1.Pod) {
				ginkgo.By(fmt.Sprintf("patching pod for %s", opStr))
				patchedPod, pErr = f.ClientSet.CoreV1().Pods(newPod.Namespace).Patch(ctx, newPod.Name,
					types.StrategicMergePatchType, []byte(patchString), metav1.PatchOptions{}, "resize")
				framework.ExpectNoError(pErr, fmt.Sprintf("failed to patch pod for %s", opStr))

				ginkgo.By(fmt.Sprintf("verifying pod patched for %s", opStr))
				e2epod.VerifyPodResources(patchedPod, expectedContainers)

				ginkgo.By(fmt.Sprintf("waiting for %s to be actuated", opStr))
				resizedPod := e2epod.WaitForPodResizeActuation(ctx, f, podClient, newPod)
				e2epod.ExpectPodResized(ctx, f, resizedPod, expectedContainers)

				// Check cgroup values only for containerd versions before 1.6.9
				ginkgo.By(fmt.Sprintf("verifying pod container's cgroup values after %s", opStr))
				framework.ExpectNoError(e2epod.VerifyPodContainersCgroupValues(ctx, f, resizedPod, expectedContainers))

				ginkgo.By(fmt.Sprintf("verifying pod resources after %s", opStr))
				e2epod.VerifyPodResources(resizedPod, expectedContainers)

				// TODO make this dynamic depending on Policy Name, Resources input and topology of target
				// machine.
				// For the moment skip below if CPU Manager Policy is set to none
				if policy.name == string(cpumanager.PolicyStatic) {
					ginkgo.By(fmt.Sprintf("verifying pod Cpus allowed list value after %s", opStr))
					if isInPlacePodVerticalScalingExclusiveCPUsEnabled {
						gomega.Eventually(ctx, e2epod.VerifyPodContainersCPUsAllowedListValue, timeouts.PodStartShort, timeouts.Poll).
							WithArguments(f, resizedPod, expectedContainers).
							Should(gomega.Succeed(), "failed to verify Pod CPUsAllowedListValue for resizedPod with InPlacePodVerticalScalingExclusiveCPUs enabled")
					} else {
						gomega.Eventually(ctx, e2epod.VerifyPodContainersCPUsAllowedListValue, timeouts.PodStartShort, timeouts.Poll).
							WithArguments(f, resizedPod, initialContainers).
							Should(gomega.Succeed(), "failed to verify Pod CPUsAllowedListValue for resizedPod with InPlacePodVerticalScalingExclusiveCPUs disabled (default)")
					}
				}
			}

			patchAndVerify(tc.testPod1.patchString, tc.testPod1.expected, tc.testPod1.containers, "resize", false, newPod1)
			patchAndVerify(tc.testPod2.patchString, tc.testPod2.expected, tc.testPod2.containers, "resize", false, newPod2)

			rbPatchStr1, err1 := e2epod.ResizeContainerPatch(tc.testPod1.containers)
			framework.ExpectNoError(err1)
			rbPatchStr2, err2 := e2epod.ResizeContainerPatch(tc.testPod2.containers)
			framework.ExpectNoError(err2)
			// Resize has been actuated, test rollback
			patchAndVerify(rbPatchStr1, tc.testPod1.containers, tc.testPod1.expected, "rollback", true, newPod1)
			patchAndVerify(rbPatchStr2, tc.testPod2.containers, tc.testPod2.expected, "rollback", true, newPod2)

			ginkgo.By("deleting pod")
			deletePodSyncByName(ctx, f, newPod1.Name)
			deletePodSyncByName(ctx, f, newPod2.Name)
			// we need to wait for all containers to really be gone so cpumanager reconcile loop will not rewrite the cpu_manager_state.
			// this is in turn needed because we will have an unavoidable (in the current framework) race with the
			// reconcile loop which will make our attempt to delete the state file and to restore the old config go haywire
			waitForAllContainerRemoval(ctx, newPod1.Name, newPod1.Namespace)
			waitForAllContainerRemoval(ctx, newPod2.Name, newPod2.Namespace)
		})
	}

	ginkgo.AfterEach(func(ctx context.Context) {
		if oldCfg != nil {
			updateKubeletConfig(ctx, f, oldCfg, true)
		}
	})
}

var _ = SIGDescribe("Pod InPlace Resize Container Extended Cases", framework.WithSerial(), func() {

	policiesGeneralAvailability := []cpuManagerPolicyConfig{
		{
			name:  string(cpumanager.PolicyStatic),
			title: ", alongside CPU Manager Static Policy with no options",
			options: map[string]string{
				cpumanager.FullPCPUsOnlyOption:             "false",
				cpumanager.DistributeCPUsAcrossNUMAOption:  "false",
				cpumanager.AlignBySocketOption:             "false",
				cpumanager.DistributeCPUsAcrossCoresOption: "false",
			},
		},
		{
			name:  string(cpumanager.PolicyStatic),
			title: ", alongside CPU Manager Static Policy with FullPCPUsOnlyOption",
			options: map[string]string{
				cpumanager.FullPCPUsOnlyOption:             "true",
				cpumanager.DistributeCPUsAcrossNUMAOption:  "false",
				cpumanager.AlignBySocketOption:             "false",
				cpumanager.DistributeCPUsAcrossCoresOption: "false",
			},
		},
	}

	doPodResizeExtendTests(policiesGeneralAvailability[0], true, true)
	doPodResizeExtendTests(policiesGeneralAvailability[1], true, true)
	doMultiPodResizeTests(policiesGeneralAvailability[0], true, true)
})