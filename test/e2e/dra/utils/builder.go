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

package utils

import (
	"context"
	_ "embed"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	resourcealphaapi "k8s.io/api/resource/v1alpha3"
	resourcev1beta1 "k8s.io/api/resource/v1beta1"
	resourcev1beta2 "k8s.io/api/resource/v1beta2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	cgoresource "k8s.io/client-go/kubernetes/typed/resource/v1"
	draclient "k8s.io/dynamic-resource-allocation/client"
	"k8s.io/dynamic-resource-allocation/resourceslice"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

// ExtendedResourceName returns hard coded extended resource name with a variable
// suffix from the input integer when it's greater than or equal to 0.
// "example.com/resource" is not special, any valid extended resource name can be used
// instead, except when using example device plugin in the test, which hard coded it,
// see test/e2e/dra/deploy_device_plugin.go.
// i == -1 is special, the extended resource name has no extra suffix, it is
// used in the test where a cluster has both DRA driver and device plugin.
func ExtendedResourceName(i int) string {
	suffix := ""
	if i >= 0 {
		suffix = strconv.Itoa(i)
	}
	return "example.com/resource" + suffix
}

// Builder contains a running counter to make objects unique within thir
// namespace.
type Builder struct {
	f                       *framework.Framework
	driver                  *Driver
	UseExtendedResourceName bool

	podCounter      int
	claimCounter    int
	ClassParameters string // JSON
}

// ClassName returns the default device class name.
func (b *Builder) ClassName() string {
	return b.f.UniqueName + b.driver.NameSuffix + "-class"
}

// Class returns the device Class that the builder's other objects
// reference.
// The input i is used to pick the extended resource name whose suffix has the
// same i for the device class.
// i == -1 is special, the extended resource name has no extra suffix, it is
// used in the test where a cluster has both DRA driver and device plugin.
func (b *Builder) Class(i int) *resourceapi.DeviceClass {
	ern := ExtendedResourceName(i)
	name := b.ClassName()
	if i >= 0 {
		name = b.ClassName() + strconv.Itoa(i)
	}
	class := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
	}
	if b.UseExtendedResourceName {
		class.Spec = resourceapi.DeviceClassSpec{
			ExtendedResourceName: &ern,
		}
	}
	class.Spec.Selectors = []resourceapi.DeviceSelector{{
		CEL: &resourceapi.CELDeviceSelector{
			Expression: fmt.Sprintf(`device.driver == "%s"`, b.driver.Name),
		},
	}}
	if b.ClassParameters != "" {
		class.Spec.Config = []resourceapi.DeviceClassConfiguration{{
			DeviceConfiguration: resourceapi.DeviceConfiguration{
				Opaque: &resourceapi.OpaqueDeviceConfiguration{
					Driver:     b.driver.Name,
					Parameters: runtime.RawExtension{Raw: []byte(b.ClassParameters)},
				},
			},
		}}
	}
	return class
}

// ExternalClaim returns external resource claim
// that test pods can reference
func (b *Builder) ExternalClaim() *resourceapi.ResourceClaim {
	b.claimCounter++
	name := "external-claim" + b.driver.NameSuffix // This is what podExternal expects.
	if b.claimCounter > 1 {
		name += fmt.Sprintf("-%d", b.claimCounter)
	}
	return &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: b.ClaimSpec(),
	}
}

// claimSpec returns the device request for a claim or claim template
// with the associated config using the v1beta1 API.
func (b *Builder) claimSpecWithV1beta1() resourcev1beta1.ResourceClaimSpec {
	parameters, _ := b.ParametersEnv()
	spec := resourcev1beta1.ResourceClaimSpec{
		Devices: resourcev1beta1.DeviceClaim{
			Requests: []resourcev1beta1.DeviceRequest{{
				Name:            "my-request",
				DeviceClassName: b.ClassName(),
			}},
			Config: []resourcev1beta1.DeviceClaimConfiguration{{
				DeviceConfiguration: resourcev1beta1.DeviceConfiguration{
					Opaque: &resourcev1beta1.OpaqueDeviceConfiguration{
						Driver: b.driver.Name,
						Parameters: runtime.RawExtension{
							Raw: []byte(parameters),
						},
					},
				},
			}},
		},
	}

	return spec
}

// claimSpec returns the device request for a claim or claim template
// with the associated config using the v1beta1 API.
func (b *Builder) claimSpecWithV1beta2() resourcev1beta2.ResourceClaimSpec {
	parameters, _ := b.ParametersEnv()
	spec := resourcev1beta2.ResourceClaimSpec{
		Devices: resourcev1beta2.DeviceClaim{
			Requests: []resourcev1beta2.DeviceRequest{{
				Name: "my-request",
				Exactly: &resourcev1beta2.ExactDeviceRequest{
					DeviceClassName: b.ClassName(),
				},
			}},
			Config: []resourcev1beta2.DeviceClaimConfiguration{{
				DeviceConfiguration: resourcev1beta2.DeviceConfiguration{
					Opaque: &resourcev1beta2.OpaqueDeviceConfiguration{
						Driver: b.driver.Name,
						Parameters: runtime.RawExtension{
							Raw: []byte(parameters),
						},
					},
				},
			}},
		},
	}

	return spec
}

// claimSpecWithV1beta2 returns the device request for a claim or claim template
// with the associated config using the latest API.
func (b *Builder) ClaimSpec() resourceapi.ResourceClaimSpec {
	parameters, _ := b.ParametersEnv()
	spec := resourceapi.ResourceClaimSpec{
		Devices: resourceapi.DeviceClaim{
			Requests: []resourceapi.DeviceRequest{{
				Name: "my-request",
				Exactly: &resourceapi.ExactDeviceRequest{
					DeviceClassName: b.ClassName(),
				},
			}},
			Config: []resourceapi.DeviceClaimConfiguration{{
				DeviceConfiguration: resourceapi.DeviceConfiguration{
					Opaque: &resourceapi.OpaqueDeviceConfiguration{
						Driver: b.driver.Name,
						Parameters: runtime.RawExtension{
							Raw: []byte(parameters),
						},
					},
				},
			}},
		},
	}

	return spec
}

// ParametersEnv returns the default user env variables as JSON (config) and key/value list (pod env).
func (b *Builder) ParametersEnv() (string, []string) {
	return `{"a":"b"}`,
		[]string{"user_a", "b"}
}

// makePod returns a simple Pod with no resource claims.
// The Pod prints its env and waits.
func (b *Builder) Pod() *v1.Pod {
	// The e2epod.InfiniteSleepCommand was changed so that it reacts to SIGTERM,
	// causing the pod to shut down immediately. This is better than the previous approach
	// with `terminationGraceperiodseconds: 1` because that still caused a one second delay.
	//
	// It is tempting to use `terminationGraceperiodSeconds: 0`, but that is a very bad
	// idea because it removes the pod before the kubelet had a chance to react (https://github.com/kubernetes/kubernetes/issues/120671).
	pod := e2epod.MakePod(b.f.Namespace.Name, nil, nil, admissionapi.LevelRestricted, "" /* no command = pause */)
	pod.Labels = make(map[string]string)
	pod.Spec.RestartPolicy = v1.RestartPolicyNever
	pod.GenerateName = ""
	b.podCounter++
	pod.Name = fmt.Sprintf("tester%s-%d", b.driver.NameSuffix, b.podCounter)
	return pod
}

// makePodInline adds an inline resource claim with default class name and parameters.
func (b *Builder) PodInline() (*v1.Pod, *resourceapi.ResourceClaimTemplate) {
	pod := b.Pod()
	pod.Spec.Containers[0].Name = "with-resource"
	podClaimName := "my-inline-claim"
	pod.Spec.Containers[0].Resources.Claims = []v1.ResourceClaim{{Name: podClaimName}}
	pod.Spec.ResourceClaims = []v1.PodResourceClaim{
		{
			Name:                      podClaimName,
			ResourceClaimTemplateName: ptr.To(pod.Name),
		},
	}
	template := &resourceapi.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name,
			Namespace: pod.Namespace,
		},
		Spec: resourceapi.ResourceClaimTemplateSpec{
			Spec: b.ClaimSpec(),
		},
	}
	return pod, template
}

func (b *Builder) PodInlineWithV1beta1() (*v1.Pod, *resourcev1beta1.ResourceClaimTemplate) {
	pod, _ := b.PodInline()
	template := &resourcev1beta1.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name,
			Namespace: pod.Namespace,
		},
		Spec: resourcev1beta1.ResourceClaimTemplateSpec{
			Spec: b.claimSpecWithV1beta1(),
		},
	}
	return pod, template
}

func (b *Builder) PodInlineWithV1beta2() (*v1.Pod, *resourcev1beta2.ResourceClaimTemplate) {
	pod, _ := b.PodInline()
	template := &resourcev1beta2.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name,
			Namespace: pod.Namespace,
		},
		Spec: resourcev1beta2.ResourceClaimTemplateSpec{
			Spec: b.claimSpecWithV1beta2(),
		},
	}
	return pod, template
}

func (b *Builder) PodInlineMultiple() (*v1.Pod, *resourceapi.ResourceClaimTemplate) {
	pod, template := b.PodInline()
	pod.Spec.Containers = append(pod.Spec.Containers, *pod.Spec.Containers[0].DeepCopy(), *pod.Spec.Containers[0].DeepCopy())
	pod.Spec.Containers[1].Name += "-1"
	pod.Spec.Containers[2].Name += "-2"
	return pod, template
}

// PodExternal adds a pod that references external resource claim with default class name and parameters.
func (b *Builder) PodExternal() *v1.Pod {
	pod := b.Pod()
	pod.Spec.Containers[0].Name = "with-resource"
	podClaimName := "resource-claim"
	externalClaimName := "external-claim" + b.driver.NameSuffix
	pod.Spec.ResourceClaims = []v1.PodResourceClaim{
		{
			Name:              podClaimName,
			ResourceClaimName: &externalClaimName,
		},
	}
	pod.Spec.Containers[0].Resources.Claims = []v1.ResourceClaim{{Name: podClaimName}}
	return pod
}

// podShared returns a pod with 3 containers that reference external resource claim with default class name and parameters.
func (b *Builder) PodExternalMultiple() *v1.Pod {
	pod := b.PodExternal()
	pod.Spec.Containers = append(pod.Spec.Containers, *pod.Spec.Containers[0].DeepCopy(), *pod.Spec.Containers[0].DeepCopy())
	pod.Spec.Containers[1].Name += "-1"
	pod.Spec.Containers[2].Name += "-2"
	return pod
}

// Create takes a bunch of objects and calls their Create function.
func (b *Builder) Create(ctx context.Context, objs ...klog.KMetadata) []klog.KMetadata {
	var createdObjs []klog.KMetadata
	for _, obj := range objs {
		ginkgo.By(fmt.Sprintf("creating %T %s", obj, obj.GetName()))
		var err error
		var createdObj klog.KMetadata
		switch obj := obj.(type) {
		case *resourceapi.DeviceClass:
			createdObj, err = b.ClientV1().DeviceClasses().Create(ctx, obj, metav1.CreateOptions{})
			ginkgo.DeferCleanup(func(ctx context.Context) {
				err := b.ClientV1().DeviceClasses().Delete(ctx, createdObj.GetName(), metav1.DeleteOptions{})
				framework.ExpectNoError(err, "delete device class")
			})
		case *v1.Pod:
			createdObj, err = b.f.ClientSet.CoreV1().Pods(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
		case *v1.ConfigMap:
			createdObj, err = b.f.ClientSet.CoreV1().ConfigMaps(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
		case *resourceapi.ResourceClaim:
			createdObj, err = b.ClientV1().ResourceClaims(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
		case *resourcev1beta1.ResourceClaim:
			createdObj, err = b.f.ClientSet.ResourceV1beta1().ResourceClaims(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
		case *resourcev1beta2.ResourceClaim:
			createdObj, err = b.f.ClientSet.ResourceV1beta2().ResourceClaims(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
		case *resourceapi.ResourceClaimTemplate:
			createdObj, err = b.ClientV1().ResourceClaimTemplates(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
		case *resourcev1beta1.ResourceClaimTemplate:
			createdObj, err = b.f.ClientSet.ResourceV1beta1().ResourceClaimTemplates(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
		case *resourcev1beta2.ResourceClaimTemplate:
			createdObj, err = b.f.ClientSet.ResourceV1beta2().ResourceClaimTemplates(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
		case *resourceapi.ResourceSlice:
			createdObj, err = b.ClientV1().ResourceSlices().Create(ctx, obj, metav1.CreateOptions{})
			ginkgo.DeferCleanup(func(ctx context.Context) {
				err := b.ClientV1().ResourceSlices().Delete(ctx, createdObj.GetName(), metav1.DeleteOptions{})
				framework.ExpectNoError(err, "delete node resource slice")
			})
		case *resourcealphaapi.DeviceTaintRule:
			createdObj, err = b.f.ClientSet.ResourceV1alpha3().DeviceTaintRules().Create(ctx, obj, metav1.CreateOptions{})
			ginkgo.DeferCleanup(func(ctx context.Context) {
				err := b.f.ClientSet.ResourceV1alpha3().DeviceTaintRules().Delete(ctx, createdObj.GetName(), metav1.DeleteOptions{})
				framework.ExpectNoError(err, "delete DeviceTaintRule")
			})
		case *appsv1.DaemonSet:
			createdObj, err = b.f.ClientSet.AppsV1().DaemonSets(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
			// Cleanup not really needed, but speeds up namespace shutdown.
			ginkgo.DeferCleanup(func(ctx context.Context) {
				err := b.f.ClientSet.AppsV1().DaemonSets(b.f.Namespace.Name).Delete(ctx, obj.Name, metav1.DeleteOptions{})
				framework.ExpectNoError(err, "delete daemonset")
			})
		default:
			framework.Fail(fmt.Sprintf("internal error, unsupported type %T", obj), 1)
		}
		framework.ExpectNoErrorWithOffset(1, err, "create %T", obj)
		createdObjs = append(createdObjs, createdObj)
	}
	return createdObjs
}

func (b *Builder) DeletePodAndWaitForNotFound(ctx context.Context, pod *v1.Pod) {
	err := b.f.ClientSet.CoreV1().Pods(b.f.Namespace.Name).Delete(ctx, pod.Name, metav1.DeleteOptions{})
	framework.ExpectNoErrorWithOffset(1, err, "delete %T", pod)
	err = e2epod.WaitForPodNotFoundInNamespace(ctx, b.f.ClientSet, pod.Name, pod.Namespace, b.f.Timeouts.PodDelete)
	framework.ExpectNoErrorWithOffset(1, err, "terminate %T", pod)
}

// TestPod runs pod and checks if container logs contain expected environment variables
func (b *Builder) TestPod(ctx context.Context, f *framework.Framework, pod *v1.Pod, env ...string) {
	ginkgo.GinkgoHelper()

	if !b.driver.WithKubelet {
		// Less testing when we cannot rely on the kubelet to actually run the pod.
		err := e2epod.WaitForPodScheduled(ctx, f.ClientSet, pod.Namespace, pod.Name)
		framework.ExpectNoError(err, "schedule pod")
		return
	}

	err := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
	framework.ExpectNoError(err, "start pod")

	if len(env) == 0 {
		_, env = b.ParametersEnv()
	}
	for _, container := range pod.Spec.Containers {
		TestContainerEnv(ctx, f, pod, container.Name, false, env...)
	}
}

// envLineRE matches env output with variables set by test/e2e/dra/test-driver.
var envLineRE = regexp.MustCompile(`^(?:admin|user|claim)_[a-zA-Z0-9_]*=.*$`)

func TestContainerEnv(ctx context.Context, f *framework.Framework, pod *v1.Pod, containerName string, fullMatch bool, env ...string) {
	ginkgo.GinkgoHelper()
	stdout, stderr, err := e2epod.ExecWithOptionsContext(ctx, f, e2epod.ExecOptions{
		Command:       []string{"env"},
		Namespace:     pod.Namespace,
		PodName:       pod.Name,
		ContainerName: containerName,
		CaptureStdout: true,
		CaptureStderr: true,
		Quiet:         true,
	})
	framework.ExpectNoError(err, fmt.Sprintf("get env output for container %s", containerName))
	gomega.Expect(stderr).To(gomega.BeEmpty(), fmt.Sprintf("env stderr for container %s", containerName))
	if fullMatch {
		// Find all env variables set by the test driver.
		var actualEnv, expectEnv []string
		for _, line := range strings.Split(stdout, "\n") {
			if envLineRE.MatchString(line) {
				actualEnv = append(actualEnv, line)
			}
		}
		for i := 0; i < len(env); i += 2 {
			expectEnv = append(expectEnv, env[i]+"="+env[i+1])
		}
		sort.Strings(actualEnv)
		sort.Strings(expectEnv)
		gomega.Expect(actualEnv).To(gomega.Equal(expectEnv), fmt.Sprintf("container %s env output:\n%s", containerName, stdout))
	} else {
		for i := 0; i < len(env); i += 2 {
			envStr := fmt.Sprintf("%s=%s\n", env[i], env[i+1])
			gomega.Expect(stdout).To(gomega.ContainSubstring(envStr), fmt.Sprintf("container %s env variables", containerName))
		}
	}
}

func NewBuilder(f *framework.Framework, driver *Driver) *Builder {
	b := &Builder{f: f, driver: driver}
	ginkgo.BeforeEach(b.setUp)
	return b
}

func NewBuilderNow(ctx context.Context, f *framework.Framework, driver *Driver) *Builder {
	b := &Builder{f: f, driver: driver}
	b.setUp(ctx)
	return b
}

func (b *Builder) setUp(ctx context.Context) {
	b.podCounter = 0
	b.claimCounter = 0
	for i := -1; i < 6; i++ {
		b.Create(ctx, b.Class(i))
	}
	ginkgo.DeferCleanup(b.tearDown)
}

// ClientV1 returns a wrapper for client-go which provides the V1 API on top of whatever is enabled in the cluster.
func (b *Builder) ClientV1() cgoresource.ResourceV1Interface {
	return draclient.New(b.f.ClientSet)
}

func (b *Builder) tearDown(ctx context.Context) {
	// Before we allow the namespace and all objects in it do be deleted by
	// the framework, we must ensure that test pods and the claims that
	// they use are deleted. Otherwise the driver might get deleted first,
	// in which case deleting the claims won't work anymore.
	ginkgo.By("delete pods and claims")
	pods, err := b.listTestPods(ctx)
	framework.ExpectNoError(err, "list pods")
	for _, pod := range pods {
		if pod.DeletionTimestamp != nil {
			continue
		}
		ginkgo.By(fmt.Sprintf("deleting %T %s", &pod, klog.KObj(&pod)))
		err := b.f.ClientSet.CoreV1().Pods(b.f.Namespace.Name).Delete(ctx, pod.Name, metav1.DeleteOptions{})
		if !apierrors.IsNotFound(err) {
			framework.ExpectNoError(err, "delete pod")
		}
	}
	gomega.Eventually(func() ([]v1.Pod, error) {
		return b.listTestPods(ctx)
	}).WithTimeout(time.Minute).Should(gomega.BeEmpty(), "remaining pods despite deletion")

	claims, err := b.ClientV1().ResourceClaims(b.f.Namespace.Name).List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err, "get resource claims")
	for _, claim := range claims.Items {
		if claim.DeletionTimestamp != nil {
			continue
		}
		ginkgo.By(fmt.Sprintf("deleting %T %s", &claim, klog.KObj(&claim)))
		err := b.ClientV1().ResourceClaims(b.f.Namespace.Name).Delete(ctx, claim.Name, metav1.DeleteOptions{})
		if !apierrors.IsNotFound(err) {
			framework.ExpectNoError(err, "delete claim")
		}
	}

	for host, plugin := range b.driver.Nodes {
		ginkgo.By(fmt.Sprintf("waiting for resources on %s to be unprepared", host))
		gomega.Eventually(plugin.GetPreparedResources).WithTimeout(time.Minute).Should(gomega.BeEmpty(), "prepared claims on host %s", host)
	}

	ginkgo.By("waiting for claims to be deallocated and deleted")
	gomega.Eventually(func() ([]resourceapi.ResourceClaim, error) {
		claims, err := b.ClientV1().ResourceClaims(b.f.Namespace.Name).List(ctx, metav1.ListOptions{})
		if err != nil {
			return nil, err
		}
		return claims.Items, nil
	}).WithTimeout(time.Minute).Should(gomega.BeEmpty(), "claims in the namespaces")
}

func (b *Builder) listTestPods(ctx context.Context) ([]v1.Pod, error) {
	pods, err := b.f.ClientSet.CoreV1().Pods(b.f.Namespace.Name).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}

	var testPods []v1.Pod
	for _, pod := range pods.Items {
		if pod.Labels["app.kubernetes.io/part-of"] == "dra-test-driver" {
			continue
		}
		testPods = append(testPods, pod)
	}
	return testPods, nil
}

func TaintAllDevices(taints ...resourceapi.DeviceTaint) driverResourcesMutatorFunc {
	return func(resources map[string]resourceslice.DriverResources) {
		for i := range resources {
			for j := range resources[i].Pools {
				for k := range resources[i].Pools[j].Slices {
					for l := range resources[i].Pools[j].Slices[k].Devices {
						resources[i].Pools[j].Slices[k].Devices[l].Taints = append(resources[i].Pools[j].Slices[k].Devices[l].Taints, taints...)
					}
				}
			}
		}
	}
}

func NetworkResources(maxAllocations int, tainted bool) driverResourcesGenFunc {
	return func(nodes *Nodes) map[string]resourceslice.DriverResources {
		driverResources := make(map[string]resourceslice.DriverResources)
		devices := make([]resourceapi.Device, 0)
		for i := 0; i < maxAllocations; i++ {
			device := resourceapi.Device{
				Name: fmt.Sprintf("device-%d", i),
			}
			if tainted {
				device.Taints = []resourceapi.DeviceTaint{{
					Key:    "example.com/taint",
					Value:  "tainted",
					Effect: resourceapi.DeviceTaintEffectNoSchedule,
				}}
			}
			devices = append(devices, device)
		}
		driverResources[multiHostDriverResources] = resourceslice.DriverResources{
			Pools: map[string]resourceslice.Pool{
				"network": {
					Slices: []resourceslice.Slice{{
						Devices: devices,
					}},
					NodeSelector: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{{
							// MatchExpressions allow multiple values,
							// MatchFields don't.
							MatchExpressions: []v1.NodeSelectorRequirement{{
								Key:      "kubernetes.io/hostname",
								Operator: v1.NodeSelectorOpIn,
								Values:   nodes.NodeNames,
							}},
						}},
					},
					Generation: 1,
				},
			},
		}
		return driverResources
	}
}

func DriverResources(maxAllocations int, devicesPerNode ...map[string]map[resourceapi.QualifiedName]resourceapi.DeviceAttribute) driverResourcesGenFunc {
	return func(nodes *Nodes) map[string]resourceslice.DriverResources {
		return DriverResourcesNow(nodes, maxAllocations, devicesPerNode...)
	}
}

func DriverResourcesNow(nodes *Nodes, maxAllocations int, devicesPerNode ...map[string]map[resourceapi.QualifiedName]resourceapi.DeviceAttribute) map[string]resourceslice.DriverResources {
	driverResources := make(map[string]resourceslice.DriverResources)
	for i, nodename := range nodes.NodeNames {
		if i < len(devicesPerNode) {
			devices := make([]resourceapi.Device, 0)
			for deviceName, attributes := range devicesPerNode[i] {
				devices = append(devices, resourceapi.Device{
					Name:       deviceName,
					Attributes: attributes,
				})
			}
			driverResources[nodename] = resourceslice.DriverResources{
				Pools: map[string]resourceslice.Pool{
					nodename: {
						Slices: []resourceslice.Slice{{
							Devices: devices,
						}},
					},
				},
			}
		} else if maxAllocations >= 0 {
			devices := make([]resourceapi.Device, maxAllocations)
			for i := 0; i < maxAllocations; i++ {
				devices[i] = resourceapi.Device{
					Name: fmt.Sprintf("device-%02d", i),
				}
			}
			driverResources[nodename] = resourceslice.DriverResources{
				Pools: map[string]resourceslice.Pool{
					nodename: {
						Slices: []resourceslice.Slice{{
							Devices: devices,
						}},
					},
				},
			}
		}
	}
	return driverResources
}

func ToDriverResources(counters []resourceapi.CounterSet, devices ...resourceapi.Device) driverResourcesGenFunc {
	return func(nodes *Nodes) map[string]resourceslice.DriverResources {
		nodename := nodes.NodeNames[0]
		return map[string]resourceslice.DriverResources{
			nodename: {
				Pools: map[string]resourceslice.Pool{
					nodename: {
						Slices: []resourceslice.Slice{
							{
								SharedCounters: counters,
								Devices:        devices,
							},
						},
					},
				},
			},
		}
	}
}
