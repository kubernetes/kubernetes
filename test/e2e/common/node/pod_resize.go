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

package node

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"go.opentelemetry.io/otel/trace/noop"
	"k8s.io/klog/v2"
	"net/http"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apimachinerytypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	clientset "k8s.io/client-go/kubernetes"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	remote "k8s.io/cri-client/pkg"
	"k8s.io/kubelet/pkg/types"
	"k8s.io/kubernetes/pkg/cluster/ports"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2enodekubelet "k8s.io/kubernetes/test/e2e_node/kubeletconfig"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/cpuset"
)

const (
	fakeExtendedResource = "dummy.com/dummy"
	// state files
	cpuManagerStateFile    = "/var/lib/kubelet/cpu_manager_state"
	memoryManagerStateFile = "/var/lib/kubelet/memory_manager_state"
)

var (
	kubeletHealthCheckURL = fmt.Sprintf("http://127.0.0.1:%d/healthz", ports.KubeletHealthzPort)
)

type cpuManagerKubeletArguments struct {
	policyName              string
	enableCPUManagerOptions bool
	reservedSystemCPUs      cpuset.CPUSet
	options                 map[string]string
}

func deleteStateFile(stateFileName string) {
	err := exec.Command("/bin/sh", "-c", fmt.Sprintf("rm -f %s", stateFileName)).Run()
	framework.ExpectNoError(err, "failed to delete the state file")
}

func waitForKubeletToStart(ctx context.Context, f *framework.Framework) {
	// wait until the kubelet health check will succeed
	gomega.Eventually(ctx, func() bool {
		return kubeletHealthCheck(kubeletHealthCheckURL)
	}, 2*time.Minute, 5*time.Second).Should(gomega.BeTrueBecause("expected kubelet to be in healthy state"))

	// Wait for the Kubelet to be ready.
	gomega.Eventually(ctx, func(ctx context.Context) bool {
		nodes, err := e2enode.TotalReady(ctx, f.ClientSet)
		framework.ExpectNoError(err)
		return nodes == 1
	}, time.Minute, time.Second).Should(gomega.BeTrueBecause("expected kubelet to be in ready state"))
}

func kubeletHealthCheck(url string) bool {
	insecureTransport := http.DefaultTransport.(*http.Transport).Clone()
	insecureTransport.TLSClientConfig = &tls.Config{InsecureSkipVerify: true}
	insecureHTTPClient := &http.Client{
		Transport: insecureTransport,
	}

	req, err := http.NewRequest(http.MethodHead, url, nil)
	if err != nil {
		return false
	}
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", framework.TestContext.BearerToken))
	resp, err := insecureHTTPClient.Do(req)
	if err != nil {
		klog.Warningf("Health check on %q failed, error=%v", url, err)
	} else if resp.StatusCode != http.StatusOK {
		klog.Warningf("Health check on %q failed, status=%d", url, resp.StatusCode)
	}
	return err == nil && resp.StatusCode == http.StatusOK
}

func updateKubeletConfig(ctx context.Context, f *framework.Framework, kubeletConfig *kubeletconfig.KubeletConfiguration, deleteStateFiles bool) {
	// Update the Kubelet configuration.
	ginkgo.By("Stopping the kubelet")
	startKubelet := stopKubelet()

	// wait until the kubelet health check will fail
	gomega.Eventually(ctx, func() bool {
		return kubeletHealthCheck(kubeletHealthCheckURL)
	}, time.Minute, time.Second).Should(gomega.BeFalseBecause("expected kubelet health check to be failed"))

	// Delete CPU and memory manager state files to be sure it will not prevent the kubelet restart
	if deleteStateFiles {
		deleteStateFile(cpuManagerStateFile)
		deleteStateFile(memoryManagerStateFile)
	}

	framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(kubeletConfig))

	ginkgo.By("Starting the kubelet")
	startKubelet()
	waitForKubeletToStart(ctx, f)
}

// findKubeletServiceName searches the unit name among the services known to systemd.
// if the `running` parameter is true, restricts the search among currently running services;
// otherwise, also stopped, failed, exited (non-running in general) services are also considered.
// TODO: Find a uniform way to deal with systemctl/initctl/service operations. #34494
func findKubeletServiceName(running bool) string {
	cmdLine := []string{
		"systemctl", "list-units", "*kubelet*",
	}
	if running {
		cmdLine = append(cmdLine, "--state=running")
	}
	stdout, err := exec.Command("sudo", cmdLine...).CombinedOutput()
	framework.ExpectNoError(err)
	regex := regexp.MustCompile(`(kubelet-\w+)`)
	matches := regex.FindStringSubmatch(string(stdout))
	gomega.Expect(matches).ToNot(gomega.BeEmpty(), "Found more than one kubelet service running: %q", stdout)
	kubeletServiceName := matches[0]
	framework.Logf("Get running kubelet with systemctl: %v, %v", string(stdout), kubeletServiceName)
	return kubeletServiceName
}

// stopKubelet will kill the running kubelet, and returns a func that will restart the process again
func stopKubelet() func() {
	kubeletServiceName := findKubeletServiceName(true)

	// reset the kubelet service start-limit-hit
	stdout, err := exec.Command("sudo", "systemctl", "reset-failed", kubeletServiceName).CombinedOutput()
	framework.ExpectNoError(err, "Failed to reset kubelet start-limit-hit with systemctl: %v, %s", err, string(stdout))

	stdout, err = exec.Command("sudo", "systemctl", "kill", kubeletServiceName).CombinedOutput()
	framework.ExpectNoError(err, "Failed to stop kubelet with systemctl: %v, %s", err, string(stdout))

	return func() {
		// we should restart service, otherwise the transient service start will fail
		stdout, err := exec.Command("sudo", "systemctl", "restart", kubeletServiceName).CombinedOutput()
		framework.ExpectNoError(err, "Failed to restart kubelet with systemctl: %v, %v", err, stdout)
	}
}

// getCRIClient connects CRI and returns CRI runtime service clients and image service client.
func getCRIClient() (internalapi.RuntimeService, internalapi.ImageManagerService, error) {
	// connection timeout for CRI service connection
	logger := klog.Background()
	const connectionTimeout = 2 * time.Minute
	runtimeEndpoint := framework.TestContext.ContainerRuntimeEndpoint
	r, err := remote.NewRemoteRuntimeService(runtimeEndpoint, connectionTimeout, noop.NewTracerProvider(), &logger)
	if err != nil {
		return nil, nil, err
	}
	imageManagerEndpoint := runtimeEndpoint
	if framework.TestContext.ImageServiceEndpoint != "" {
		// ImageServiceEndpoint is the same as ContainerRuntimeEndpoint if not
		// explicitly specified
		imageManagerEndpoint = framework.TestContext.ImageServiceEndpoint
	}
	i, err := remote.NewRemoteImageService(imageManagerEndpoint, connectionTimeout, noop.NewTracerProvider(), &logger)
	if err != nil {
		return nil, nil, err
	}
	return r, i, nil
}

func getLocalNodeCPUDetails(ctx context.Context, f *framework.Framework) (cpuCapVal int64, cpuAllocVal int64, cpuResVal int64) {
	localNodeCap := getLocalNode(ctx, f).Status.Capacity
	cpuCap := localNodeCap[v1.ResourceCPU]
	localNodeAlloc := getLocalNode(ctx, f).Status.Allocatable
	cpuAlloc := localNodeAlloc[v1.ResourceCPU]
	cpuRes := cpuCap.DeepCopy()
	cpuRes.Sub(cpuAlloc)

	// RoundUp reserved CPUs to get only integer cores.
	cpuRes.RoundUp(0)

	return cpuCap.Value(), cpuCap.Value() - cpuRes.Value(), cpuRes.Value()
}

func getLocalNode(ctx context.Context, f *framework.Framework) *v1.Node {
	nodeList, err := e2enode.GetReadySchedulableNodes(ctx, f.ClientSet)
	framework.ExpectNoError(err)
	gomega.Expect(nodeList.Items).Should(gomega.HaveLen(1), "Unexpected number of node objects for node e2e. Expects only one node.")
	return &nodeList.Items[0]
}

func getSMTLevel() int {
	cpuID := 0 // this is just the most likely cpu to be present in a random system. No special meaning besides this.
	out, err := exec.Command("/bin/sh", "-c", fmt.Sprintf("cat /sys/devices/system/cpu/cpu%d/topology/thread_siblings_list | tr -d \"\n\r\"", cpuID)).Output()
	framework.ExpectNoError(err)
	// how many thread sibling you have = SMT level
	// example: 2-way SMT means 2 threads sibling for each thread
	cpus, err := cpuset.Parse(strings.TrimSpace(string(out)))
	framework.ExpectNoError(err)
	return cpus.Size()
}

func configureCPUManagerInKubelet(oldCfg *kubeletconfig.KubeletConfiguration, kubeletArguments *cpuManagerKubeletArguments) *kubeletconfig.KubeletConfiguration {
	newCfg := oldCfg.DeepCopy()
	if newCfg.FeatureGates == nil {
		newCfg.FeatureGates = make(map[string]bool)
	}

	newCfg.FeatureGates["CPUManagerPolicyOptions"] = kubeletArguments.enableCPUManagerOptions
	newCfg.FeatureGates["CPUManagerPolicyBetaOptions"] = kubeletArguments.enableCPUManagerOptions
	newCfg.FeatureGates["CPUManagerPolicyAlphaOptions"] = kubeletArguments.enableCPUManagerOptions

	newCfg.CPUManagerPolicy = kubeletArguments.policyName
	newCfg.CPUManagerReconcilePeriod = metav1.Duration{Duration: 1 * time.Second}

	if kubeletArguments.options != nil {
		newCfg.CPUManagerPolicyOptions = kubeletArguments.options
	}

	if kubeletArguments.reservedSystemCPUs.Size() > 0 {
		cpus := kubeletArguments.reservedSystemCPUs.String()
		framework.Logf("configureCPUManagerInKubelet: using reservedSystemCPUs=%q", cpus)
		newCfg.ReservedSystemCPUs = cpus
	} else {
		// The Kubelet panics if either kube-reserved or system-reserved is not set
		// when CPU Manager is enabled. Set cpu in kube-reserved > 0 so that
		// kubelet doesn't panic.
		if newCfg.KubeReserved == nil {
			newCfg.KubeReserved = map[string]string{}
		}

		if _, ok := newCfg.KubeReserved["cpu"]; !ok {
			newCfg.KubeReserved["cpu"] = "200m"
		}
	}

	return newCfg
}

func patchNode(ctx context.Context, client clientset.Interface, old *v1.Node, new *v1.Node) error {
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
	_, err = client.CoreV1().Nodes().Patch(ctx, old.Name, apimachinerytypes.StrategicMergePatchType, patchBytes, metav1.PatchOptions{}, "status")
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
}

// waitForAllContainerRemoval waits until all the containers on a given pod are really gone.
// This is needed by the e2e tests which involve exclusive resource allocation (cpu, topology manager; podresources; etc.)
// In these cases, we need to make sure the tests clean up after themselves to make sure each test runs in
// a pristine environment. The only way known so far to do that is to introduce this wait.
// Worth noting, however, that this makes the test runtime much bigger.
func waitForAllContainerRemoval(ctx context.Context, podName, podNS string) {
	rs, _, err := getCRIClient()
	framework.ExpectNoError(err)
	gomega.Eventually(ctx, func(ctx context.Context) error {
		containers, err := rs.ListContainers(ctx, &runtimeapi.ContainerFilter{
			LabelSelector: map[string]string{
				types.KubernetesPodNameLabel:      podName,
				types.KubernetesPodNamespaceLabel: podNS,
			},
		})
		if err != nil {
			return fmt.Errorf("got error waiting for all containers to be removed from CRI: %w", err)
		}

		if len(containers) > 0 {
			return fmt.Errorf("expected all containers to be removed from CRI but %v containers still remain. Containers: %+v", len(containers), containers)
		}
		return nil
	}, 2*time.Minute, 1*time.Second).Should(gomega.Succeed())
}

func cpuManagerPolicyKubeletConfig(ctx context.Context, f *framework.Framework, oldCfg *kubeletconfig.KubeletConfiguration, cpuManagerPolicyName string, cpuManagerPolicyOptions map[string]string) {
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
			})
			updateKubeletConfig(ctx, f, newCfg, true)
		}
	}
}

type cpuManagerPolicyConfig struct {
	name    string
	title   string
	options map[string]string
}

func doPodResizeTests(f *framework.Framework, policy cpuManagerPolicyConfig) {
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var oldCfg *kubeletconfig.KubeletConfiguration
	ginkgo.BeforeEach(func(ctx context.Context) {
		var err error
		if oldCfg == nil {
			oldCfg, err = e2enodekubelet.GetCurrentKubeletConfig(ctx, framework.TestContext.NodeName, "", false, framework.TestContext.StandaloneMode)
			framework.ExpectNoError(err)
		}
	})

	type testCase struct {
		name                string
		containers          []e2epod.ResizableContainerInfo
		patchString         string
		expected            []e2epod.ResizableContainerInfo
		addExtendedResource bool
	}

	noRestart := v1.NotRequired
	doRestart := v1.RestartContainer
	tests := []testCase{
		{
			name: "Guaranteed QoS pod, one container - increase CPU & memory",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "200Mi", MemLim: "200Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"200m","memory":"400Mi"},"limits":{"cpu":"200m","memory":"400Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "400Mi", MemLim: "400Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
		},
		{
			name: "Guaranteed QoS pod, one container - decrease CPU & memory",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "300m", CPULim: "300m", MemReq: "500Mi", MemLim: "500Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"100m","memory":"250Mi"},"limits":{"cpu":"100m","memory":"250Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "250Mi", MemLim: "250Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
		},
		{
			name: "Guaranteed QoS pod, one container - increase CPU & decrease memory",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "200Mi", MemLim: "200Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"200m","memory":"100Mi"},"limits":{"cpu":"200m","memory":"100Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "100Mi", MemLim: "100Mi"},
				},
			},
		},
		{
			name: "Guaranteed QoS pod, one container - decrease CPU & increase memory",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "200Mi", MemLim: "200Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"50m","memory":"300Mi"},"limits":{"cpu":"50m","memory":"300Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "50m", CPULim: "50m", MemReq: "300Mi", MemLim: "300Mi"},
				},
			},
		},
		{
			name: "Guaranteed QoS pod, three containers (c1, c2, c3) - increase: CPU (c1,c3), memory (c2) ; decrease: CPU (c2), memory (c1,c3)",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "200Mi", MemLim: "200Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c3",
					Resources: &e2epod.ContainerResources{CPUReq: "300m", CPULim: "300m", MemReq: "300Mi", MemLim: "300Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"140m","memory":"50Mi"},"limits":{"cpu":"140m","memory":"50Mi"}}},
						{"name":"c2", "resources":{"requests":{"cpu":"150m","memory":"240Mi"},"limits":{"cpu":"150m","memory":"240Mi"}}},
						{"name":"c3", "resources":{"requests":{"cpu":"340m","memory":"250Mi"},"limits":{"cpu":"340m","memory":"250Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "140m", CPULim: "140m", MemReq: "50Mi", MemLim: "50Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &e2epod.ContainerResources{CPUReq: "150m", CPULim: "150m", MemReq: "240Mi", MemLim: "240Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c3",
					Resources: &e2epod.ContainerResources{CPUReq: "340m", CPULim: "340m", MemReq: "250Mi", MemLim: "250Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease memory requests only",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"200Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "200Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease memory limits only",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"limits":{"memory":"400Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "400Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase memory requests only",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"300Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "300Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase memory limits only",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"limits":{"memory":"600Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "600Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease CPU requests only",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"100m"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease CPU limits only",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"limits":{"cpu":"300m"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "300m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase CPU requests only",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"150m"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "150m", CPULim: "200m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase CPU limits only",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"limits":{"cpu":"500m"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "500m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease CPU requests and limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"100m"},"limits":{"cpu":"200m"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase CPU requests and limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"200m"},"limits":{"cpu":"400m"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease CPU requests and increase CPU limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"100m"},"limits":{"cpu":"500m"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "500m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase CPU requests and decrease CPU limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"200m"},"limits":{"cpu":"300m"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "300m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease memory requests and limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "200Mi", MemLim: "400Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"100Mi"},"limits":{"memory":"300Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "100Mi", MemLim: "300Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase memory requests and limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "200Mi", MemLim: "400Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"300Mi"},"limits":{"memory":"500Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "300Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease memory requests and increase memory limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "200Mi", MemLim: "400Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"100Mi"},"limits":{"memory":"500Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "100Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase memory requests and decrease memory limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "200Mi", MemLim: "400Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"300Mi"},"limits":{"memory":"300Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "300Mi", MemLim: "300Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease CPU requests and increase memory limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "200Mi", MemLim: "400Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"100m"},"limits":{"memory":"500Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "400m", MemReq: "200Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase CPU requests and decrease memory limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "400m", MemReq: "200Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"200m"},"limits":{"memory":"400Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "200Mi", MemLim: "400Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease memory requests and increase CPU limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "200Mi", MemLim: "400Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"100Mi"},"limits":{"cpu":"300m"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "300m", MemReq: "100Mi", MemLim: "400Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase memory requests and decrease CPU limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "200Mi", MemLim: "400Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"300Mi"},"limits":{"cpu":"300m"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "300m", MemReq: "300Mi", MemLim: "400Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests - decrease memory request",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", MemReq: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"400Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", MemReq: "400Mi"},
				},
			},
		},
		{
			name: "Guaranteed QoS pod, one container - increase CPU (NotRequired) & memory (RestartContainer)",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "200Mi", MemLim: "200Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &doRestart,
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"200m","memory":"400Mi"},"limits":{"cpu":"200m","memory":"400Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:         "c1",
					Resources:    &e2epod.ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "400Mi", MemLim: "400Mi"},
					CPUPolicy:    &noRestart,
					MemPolicy:    &doRestart,
					RestartCount: 1,
				},
			},
		},
		{
			name: "Burstable QoS pod, one container - decrease CPU (RestartContainer) & memory (NotRequired)",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "200Mi", MemLim: "400Mi"},
					CPUPolicy: &doRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"50m","memory":"100Mi"},"limits":{"cpu":"100m","memory":"200Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:         "c1",
					Resources:    &e2epod.ContainerResources{CPUReq: "50m", CPULim: "100m", MemReq: "100Mi", MemLim: "200Mi"},
					CPUPolicy:    &doRestart,
					MemPolicy:    &noRestart,
					RestartCount: 1,
				},
			},
		},
		{
			name: "Burstable QoS pod, three containers - increase c1 resources, no change for c2, decrease c3 resources (no net change for pod)",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "100Mi", MemLim: "200Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "300m", MemReq: "200Mi", MemLim: "300Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &doRestart,
				},
				{
					Name:      "c3",
					Resources: &e2epod.ContainerResources{CPUReq: "300m", CPULim: "400m", MemReq: "300Mi", MemLim: "400Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"150m","memory":"150Mi"},"limits":{"cpu":"250m","memory":"250Mi"}}},
						{"name":"c3", "resources":{"requests":{"cpu":"250m","memory":"250Mi"},"limits":{"cpu":"350m","memory":"350Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "150m", CPULim: "250m", MemReq: "150Mi", MemLim: "250Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "300m", MemReq: "200Mi", MemLim: "300Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &doRestart,
				},
				{
					Name:      "c3",
					Resources: &e2epod.ContainerResources{CPUReq: "250m", CPULim: "350m", MemReq: "250Mi", MemLim: "350Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
		},
		{
			name: "Burstable QoS pod, three containers - decrease c1 resources, increase c2 resources, no change for c3 (net increase for pod)",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "100Mi", MemLim: "200Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "300m", MemReq: "200Mi", MemLim: "300Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &doRestart,
				},
				{
					Name:      "c3",
					Resources: &e2epod.ContainerResources{CPUReq: "300m", CPULim: "400m", MemReq: "300Mi", MemLim: "400Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"50m","memory":"50Mi"},"limits":{"cpu":"150m","memory":"150Mi"}}},
						{"name":"c2", "resources":{"requests":{"cpu":"350m","memory":"350Mi"},"limits":{"cpu":"450m","memory":"450Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "50m", CPULim: "150m", MemReq: "50Mi", MemLim: "150Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:         "c2",
					Resources:    &e2epod.ContainerResources{CPUReq: "350m", CPULim: "450m", MemReq: "350Mi", MemLim: "450Mi"},
					CPUPolicy:    &noRestart,
					MemPolicy:    &doRestart,
					RestartCount: 1,
				},
				{
					Name:      "c3",
					Resources: &e2epod.ContainerResources{CPUReq: "300m", CPULim: "400m", MemReq: "300Mi", MemLim: "400Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
		},
		{
			name: "Burstable QoS pod, three containers - no change for c1, increase c2 resources, decrease c3 (net decrease for pod)",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "100Mi", MemLim: "200Mi"},
					CPUPolicy: &doRestart,
					MemPolicy: &doRestart,
				},
				{
					Name:      "c2",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "300m", MemReq: "200Mi", MemLim: "300Mi"},
					CPUPolicy: &doRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c3",
					Resources: &e2epod.ContainerResources{CPUReq: "300m", CPULim: "400m", MemReq: "300Mi", MemLim: "400Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &doRestart,
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c2", "resources":{"requests":{"cpu":"250m","memory":"250Mi"},"limits":{"cpu":"350m","memory":"350Mi"}}},
						{"name":"c3", "resources":{"requests":{"cpu":"100m","memory":"100Mi"},"limits":{"cpu":"200m","memory":"200Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "100Mi", MemLim: "200Mi"},
					CPUPolicy: &doRestart,
					MemPolicy: &doRestart,
				},
				{
					Name:         "c2",
					Resources:    &e2epod.ContainerResources{CPUReq: "250m", CPULim: "350m", MemReq: "250Mi", MemLim: "350Mi"},
					CPUPolicy:    &noRestart,
					MemPolicy:    &noRestart,
					RestartCount: 1,
				},
				{
					Name:         "c3",
					Resources:    &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "100Mi", MemLim: "200Mi"},
					CPUPolicy:    &doRestart,
					MemPolicy:    &doRestart,
					RestartCount: 1,
				},
			},
		},
		{
			name: "Guaranteed QoS pod, one container - increase CPU & memory with an extended resource",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name: "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "200Mi", MemLim: "200Mi",
						ExtendedResourceReq: "1", ExtendedResourceLim: "1"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: `{"spec":{"containers":[
					{"name":"c1", "resources":{"requests":{"cpu":"200m","memory":"400Mi"},"limits":{"cpu":"200m","memory":"400Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name: "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "400Mi", MemLim: "400Mi",
						ExtendedResourceReq: "1", ExtendedResourceLim: "1"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			addExtendedResource: true,
		},
	}

	timeouts := framework.NewTimeoutContext()

	for idx := range tests {
		tc := tests[idx]
		ginkgo.It(tc.name+policy.title, func(ctx context.Context) {
			cpuManagerPolicyKubeletConfig(ctx, f, oldCfg, policy.name, policy.options)

			podClient := e2epod.NewPodClient(f)
			var testPod, patchedPod *v1.Pod
			var pErr error

			tStamp := strconv.Itoa(time.Now().Nanosecond())
			e2epod.InitDefaultResizePolicy(tc.containers)
			e2epod.InitDefaultResizePolicy(tc.expected)
			testPod = e2epod.MakePodWithResizableContainers(f.Namespace.Name, "testpod", tStamp, tc.containers)
			testPod = e2epod.MustMixinRestrictedPodSecurity(testPod)

			if tc.addExtendedResource {
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
			}

			ginkgo.By("creating pod")
			newPod := podClient.CreateSync(ctx, testPod)

			ginkgo.By("verifying initial pod resources, allocations are as expected")
			e2epod.VerifyPodResources(newPod, tc.containers)
			ginkgo.By("verifying initial pod resize policy is as expected")
			e2epod.VerifyPodResizePolicy(newPod, tc.containers)

			ginkgo.By("verifying initial pod status resources are as expected")
			e2epod.VerifyPodStatusResources(newPod, tc.containers)
			ginkgo.By("verifying initial cgroup config are as expected")
			framework.ExpectNoError(e2epod.VerifyPodContainersCgroupValues(f, newPod, tc.containers))
			// TODO make this dynamic depending on Policy Name, Resources input and topology of target
			// machine.
			// For the moment skip below if CPU Manager Policy is set to none
			if policy.name == string(cpumanager.PolicyStatic) {
				ginkgo.By("verifying initial pod Cpus allowed list value")
				gomega.Eventually(ctx, e2epod.VerifyPodContainersCpusAllowedListValue, timeouts.PodStartShort, timeouts.Poll).
					WithArguments(f, newPod, tc.containers).
					Should(gomega.BeNil(), "failed to verify initial Pod CpusAllowedListValue")
			}

			patchAndVerify := func(patchString string, expectedContainers []e2epod.ResizableContainerInfo, initialContainers []e2epod.ResizableContainerInfo, opStr string, isRollback bool) {
				ginkgo.By(fmt.Sprintf("patching pod for %s", opStr))
				patchedPod, pErr = f.ClientSet.CoreV1().Pods(newPod.Namespace).Patch(context.TODO(), newPod.Name,
					apimachinerytypes.StrategicMergePatchType, []byte(patchString), metav1.PatchOptions{})
				framework.ExpectNoError(pErr, fmt.Sprintf("failed to patch pod for %s", opStr))

				ginkgo.By(fmt.Sprintf("verifying pod patched for %s", opStr))
				e2epod.VerifyPodResources(patchedPod, expectedContainers)
				gomega.Eventually(ctx, e2epod.VerifyPodAllocations, timeouts.PodStartShort, timeouts.Poll).
					WithArguments(patchedPod, initialContainers).
					Should(gomega.BeNil(), "failed to verify Pod allocations for patchedPod")

				ginkgo.By(fmt.Sprintf("waiting for %s to be actuated", opStr))
				resizedPod := e2epod.WaitForPodResizeActuation(ctx, f, podClient, newPod, patchedPod, expectedContainers, initialContainers, isRollback)

				// Check cgroup values only for containerd versions before 1.6.9
				ginkgo.By(fmt.Sprintf("verifying pod container's cgroup values after %s", opStr))
				framework.ExpectNoError(e2epod.VerifyPodContainersCgroupValues(f, resizedPod, expectedContainers))

				ginkgo.By(fmt.Sprintf("verifying pod resources after %s", opStr))
				e2epod.VerifyPodResources(resizedPod, expectedContainers)

				ginkgo.By(fmt.Sprintf("verifying pod allocations after %s", opStr))
				gomega.Eventually(ctx, e2epod.VerifyPodAllocations, timeouts.PodStartShort, timeouts.Poll).
					WithArguments(resizedPod, expectedContainers).
					Should(gomega.BeNil(), "failed to verify Pod allocations for resizedPod")
				// TODO make this dynamic depending on Policy Name, Resources input and topology of target
				// machine.
				// For the moment skip below if CPU Manager Policy is set to none
				if policy.name == string(cpumanager.PolicyStatic) {
					ginkgo.By("verifying pod Cpus allowed list value after resize")
					gomega.Eventually(ctx, e2epod.VerifyPodContainersCpusAllowedListValue, timeouts.PodStartShort, timeouts.Poll).
						WithArguments(f, resizedPod, tc.expected).
						Should(gomega.BeNil(), "failed to verify Pod CpusAllowedListValue for resizedPod")
				}
			}

			patchAndVerify(tc.patchString, tc.expected, tc.containers, "resize", false)

			rbPatchStr, err := e2epod.ResizeContainerPatch(tc.containers)
			framework.ExpectNoError(err)
			// Resize has been actuated, test rollback
			patchAndVerify(rbPatchStr, tc.containers, tc.expected, "rollback", true)

			ginkgo.By("deleting pod")
			podClient.DeleteSync(ctx, newPod.Name, metav1.DeleteOptions{}, timeouts.PodDelete)
		})
	}

	ginkgo.AfterEach(func(ctx context.Context) {
		updateKubeletConfig(ctx, f, oldCfg, true)
	})

}

func doPodResizeErrorTests(f *framework.Framework) {

	type testCase struct {
		name        string
		containers  []e2epod.ResizableContainerInfo
		patchString string
		patchError  string
		expected    []e2epod.ResizableContainerInfo
	}

	tests := []testCase{
		{
			name: "BestEffort QoS pod, one container - try requesting memory, expect error",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name: "c1",
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"400Mi"}}}
					]}}`,
			patchError: "Pod QoS is immutable",
			expected: []e2epod.ResizableContainerInfo{
				{
					Name: "c1",
				},
			},
		},
		{
			name: "BestEffort QoS pod, three containers - try requesting memory for c1, expect error",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name: "c1",
				},
				{
					Name: "c2",
				},
				{
					Name: "c3",
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"400Mi"}}}
					]}}`,
			patchError: "Pod QoS is immutable",
			expected: []e2epod.ResizableContainerInfo{
				{
					Name: "c1",
				},
				{
					Name: "c2",
				},
				{
					Name: "c3",
				},
			},
		},
	}

	timeouts := framework.NewTimeoutContext()

	for idx := range tests {
		tc := tests[idx]
		ginkgo.It(tc.name, func(ctx context.Context) {
			podClient := e2epod.NewPodClient(f)
			var testPod, patchedPod *v1.Pod
			var pErr error

			tStamp := strconv.Itoa(time.Now().Nanosecond())
			e2epod.InitDefaultResizePolicy(tc.containers)
			e2epod.InitDefaultResizePolicy(tc.expected)
			testPod = e2epod.MakePodWithResizableContainers(f.Namespace.Name, "testpod", tStamp, tc.containers)
			testPod = e2epod.MustMixinRestrictedPodSecurity(testPod)

			ginkgo.By("creating pod")
			newPod := podClient.CreateSync(ctx, testPod)

			ginkgo.By("verifying initial pod resources, allocations, and policy are as expected")
			e2epod.VerifyPodResources(newPod, tc.containers)
			e2epod.VerifyPodResizePolicy(newPod, tc.containers)

			ginkgo.By("verifying initial pod status resources and cgroup config are as expected")
			e2epod.VerifyPodStatusResources(newPod, tc.containers)

			ginkgo.By("patching pod for resize")
			patchedPod, pErr = f.ClientSet.CoreV1().Pods(newPod.Namespace).Patch(ctx, newPod.Name,
				apimachinerytypes.StrategicMergePatchType, []byte(tc.patchString), metav1.PatchOptions{})
			if tc.patchError == "" {
				framework.ExpectNoError(pErr, "failed to patch pod for resize")
			} else {
				gomega.Expect(pErr).To(gomega.HaveOccurred(), tc.patchError)
				patchedPod = newPod
			}

			ginkgo.By("verifying pod resources after patch")
			e2epod.VerifyPodResources(patchedPod, tc.expected)

			ginkgo.By("verifying pod allocations after patch")
			gomega.Eventually(ctx, e2epod.VerifyPodAllocations, timeouts.PodStartShort, timeouts.Poll).
				WithArguments(patchedPod, tc.expected).
				Should(gomega.BeNil(), "failed to verify Pod allocations for patchedPod")

			ginkgo.By("deleting pod")
			gp := int64(0)
			delOpts := metav1.DeleteOptions{
				GracePeriodSeconds: &gp,
			}
			podClient.DeleteSync(ctx, newPod.Name, delOpts, timeouts.PodDelete)
			// we need to wait for all containers to really be gone so cpumanager reconcile loop will not rewrite the cpu_manager_state.
			// this is in turn needed because we will have an unavoidable (in the current framework) race with the
			// reconcile loop which will make our attempt to delete the state file and to restore the old config go haywire
			waitForAllContainerRemoval(ctx, newPod.Name, newPod.Namespace)

		})
	}
}

// NOTE: Pod resize scheduler resource quota tests are out of scope in e2e_node tests,
//       because in e2e_node tests
//          a) scheduler and controller manager is not running by the Node e2e
//          b) api-server in services doesn't start with --enable-admission-plugins=ResourceQuota
//             and is not possible to start it from TEST_ARGS
//       Above tests are performed by doSheduletTests() and doPodResizeResourceQuotaTests()
//       in test/e2e/node/pod_resize.go

var _ = SIGDescribe("Pod InPlace Resize Container", framework.WithSerial(), feature.InPlacePodVerticalScaling, "[NodeAlphaFeature:InPlacePodVerticalScaling]", func() {
	f := framework.NewDefaultFramework("pod-resize-tests")

	ginkgo.BeforeEach(func(ctx context.Context) {
		node, err := e2enode.GetRandomReadySchedulableNode(ctx, f.ClientSet)
		framework.ExpectNoError(err)
		if !e2enode.NodeSupportsInPlacePodVerticalScaling(node) {
			e2eskipper.Skipf("runtime does not support InPlacePodVerticalScaling -- skipping")
		}
	})

	policiesGeneralAvailability := []cpuManagerPolicyConfig{
		{
			name:  string(cpumanager.PolicyNone),
			title: "",
		},
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
	}

	for idp := range policiesGeneralAvailability {
		doPodResizeTests(f, policiesGeneralAvailability[idp])
	}

	policiesBeta := []cpuManagerPolicyConfig{
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

	for idp := range policiesBeta {
		doPodResizeTests(f, policiesBeta[idp])
	}

	/*policiesAlpha := []cpuManagerPolicyConfig{
		{
			name:  string(cpumanager.PolicyStatic),
			title: ", alongside CPU Manager Static Policy with DistributeCPUsAcrossNUMAOption",
			options: map[string]string{
				cpumanager.FullPCPUsOnlyOption:             "false",
				cpumanager.DistributeCPUsAcrossNUMAOption:  "true",
				cpumanager.AlignBySocketOption:             "false",
				cpumanager.DistributeCPUsAcrossCoresOption: "false",
			},
		},
		{
			name:  string(cpumanager.PolicyStatic),
			title: ", alongside CPU Manager Static Policy with FullPCPUsOnlyOption, DistributeCPUsAcrossNUMAOption",
			options: map[string]string{
				cpumanager.FullPCPUsOnlyOption:             "true",
				cpumanager.DistributeCPUsAcrossNUMAOption:  "true",
				cpumanager.AlignBySocketOption:             "false",
				cpumanager.DistributeCPUsAcrossCoresOption: "false",
			},
		},
		{
			name:  string(cpumanager.PolicyStatic),
			title: ", alongside CPU Manager Static Policy with AlignBySocketOption",
			options: map[string]string{
				cpumanager.FullPCPUsOnlyOption:             "false",
				cpumanager.DistributeCPUsAcrossNUMAOption:  "false",
				cpumanager.AlignBySocketOption:             "true",
				cpumanager.DistributeCPUsAcrossCoresOption: "false",
			},
		},
		{
			name:  string(cpumanager.PolicyStatic),
			title: ", alongside CPU Manager Static Policy with FullPCPUsOnlyOption, AlignBySocketOption",
			options: map[string]string{
				cpumanager.FullPCPUsOnlyOption:             "true",
				cpumanager.DistributeCPUsAcrossNUMAOption:  "false",
				cpumanager.AlignBySocketOption:             "true",
				cpumanager.DistributeCPUsAcrossCoresOption: "false",
			},
		},
		{
			name:  string(cpumanager.PolicyStatic),
			title: ", alongside CPU Manager Static Policy with DistributeCPUsAcrossNUMAOption, AlignBySocketOption",
			options: map[string]string{
				cpumanager.FullPCPUsOnlyOption:             "false",
				cpumanager.DistributeCPUsAcrossNUMAOption:  "true",
				cpumanager.AlignBySocketOption:             "true",
				cpumanager.DistributeCPUsAcrossCoresOption: "false",
			},
		},
		{
			name:  string(cpumanager.PolicyStatic),
			title: ", alongside CPU Manager Static Policy with FullPCPUsOnlyOption, DistributeCPUsAcrossNUMAOption, AlignBySocketOption",
			options: map[string]string{
				cpumanager.FullPCPUsOnlyOption:             "true",
				cpumanager.DistributeCPUsAcrossNUMAOption:  "true",
				cpumanager.AlignBySocketOption:             "true",
				cpumanager.DistributeCPUsAcrossCoresOption: "false",
			},
		},
		{
			name:  string(cpumanager.PolicyStatic),
			title: ", alongside CPU Manager Static Policy with DistributeCPUsAcrossCoresOption",
			options: map[string]string{
				cpumanager.FullPCPUsOnlyOption:             "false",
				cpumanager.DistributeCPUsAcrossNUMAOption:  "false",
				cpumanager.AlignBySocketOption:             "false",
				cpumanager.DistributeCPUsAcrossCoresOption: "true",
			},
		},
		{
			name:  string(cpumanager.PolicyStatic),
			title: ", alongside CPU Manager Static Policy with DistributeCPUsAcrossCoresOption, AlignBySocketOption",
			options: map[string]string{
				cpumanager.FullPCPUsOnlyOption:             "false",
				cpumanager.DistributeCPUsAcrossNUMAOption:  "false",
				cpumanager.AlignBySocketOption:             "true",
				cpumanager.DistributeCPUsAcrossCoresOption: "true",
			},
		},
	}

	for idp := range policiesAlpha {
		doPodResizeTests(f,policiesAlpha[idp])
	}*/

	doPodResizeErrorTests(f)
})
