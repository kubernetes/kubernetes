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
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	cgoresource "k8s.io/client-go/kubernetes/typed/resource/v1"
	draclient "k8s.io/dynamic-resource-allocation/client"
	"k8s.io/dynamic-resource-allocation/resourceslice"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/e2e/dra/test-driver/app"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/utils/client-go/ktesting"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

// ExtendedResourceName returns extended resource name with a variable suffix.
// Example: b.ExtendedResourceName("gpu") returns "driver-name/resource-gpu"
func (b *Builder) ExtendedResourceName(suffix string) string {
	return b.Driver.Name + "/resource-" + suffix
}

// Builder contains a running counter to make objects unique within their
// namespace.
type Builder struct {
	namespace               string
	Driver                  *Driver
	UseExtendedResourceName bool

	podCounter      int
	workloadCounter int
	podGroupCounter int
	claimCounter    int
	ClassParameters string // JSON
	SkipCleanup     bool
}

// DeviceClassWrapper is a wrapper around DeviceClass that allows
// adding builder-style functions that modify the class before creation.
type DeviceClassWrapper struct {
	*resourceapi.DeviceClass
}

// ClassName returns the default device class name.
func (b *Builder) ClassName() string {
	return b.namespace + b.Driver.NameSuffix + "-class"
}

// DriverName returns the default device driver name.
func (b *Builder) DriverName() string {
	return b.Driver.Name
}

// Class returns the device Class that the builder's other objects
// reference.
func (b *Builder) Class() *DeviceClassWrapper {
	name := b.ClassName()
	class := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
	}
	class.Spec.Selectors = []resourceapi.DeviceSelector{{
		CEL: &resourceapi.CELDeviceSelector{
			Expression: fmt.Sprintf(`device.driver == "%s"`, b.Driver.Name),
		},
	}}
	if b.ClassParameters != "" {
		class.Spec.Config = []resourceapi.DeviceClassConfiguration{{
			DeviceConfiguration: resourceapi.DeviceConfiguration{
				Opaque: &resourceapi.OpaqueDeviceConfiguration{
					Driver:     b.Driver.Name,
					Parameters: runtime.RawExtension{Raw: []byte(b.ClassParameters)},
				},
			},
		}}
	}
	return &DeviceClassWrapper{DeviceClass: class}
}

// ClassWithExtendedResource returns a device class with the extended resource name set to the provided value.
// The class name is suffixed with the last part of the extended resource name to make it unique.
func (b *Builder) ClassWithExtendedResource(extendedResource string) *resourceapi.DeviceClass {
	suffix := extendedResource[strings.LastIndex(extendedResource, "/")+1:]
	return b.Class().WithName(b.ClassName() + "-" + suffix).WithExtendedResource(extendedResource).DeviceClass
}

// WithName sets the name of the device class.
func (dcw *DeviceClassWrapper) WithName(name string) *DeviceClassWrapper {
	dcw.ObjectMeta.Name = name
	return dcw
}

// WithExtendedResource sets the extended resource name of the device class.
func (dcw *DeviceClassWrapper) WithExtendedResource(extendedResourceName string) *DeviceClassWrapper {
	dcw.Spec.ExtendedResourceName = &extendedResourceName
	return dcw
}

// ExternalClaim returns external resource claim
// that test pods can reference
func (b *Builder) ExternalClaim() *resourceapi.ResourceClaim {
	b.claimCounter++
	name := "external-claim" + b.Driver.NameSuffix // This is what podExternal expects.
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
						Driver: b.Driver.Name,
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
						Driver: b.Driver.Name,
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
						Driver: b.Driver.Name,
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
	pod := e2epod.MakePod(b.namespace, nil, nil, admissionapi.LevelRestricted, "" /* no command = pause */)
	pod.Labels = make(map[string]string)
	pod.Spec.RestartPolicy = v1.RestartPolicyNever
	pod.GenerateName = ""
	b.podCounter++
	pod.Name = fmt.Sprintf("tester%s-%d", b.Driver.NameSuffix, b.podCounter)
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

// PodExternal adds a pod that references the named resource claim.
func (b *Builder) PodExternal(externalClaimName string) *v1.Pod {
	pod := b.Pod()
	pod.Spec.Containers[0].Name = "with-resource"
	podClaimName := "resource-claim"
	pod.Spec.ResourceClaims = []v1.PodResourceClaim{
		{
			Name:              podClaimName,
			ResourceClaimName: &externalClaimName,
		},
	}
	pod.Spec.Containers[0].Resources.Claims = []v1.ResourceClaim{{Name: podClaimName}}
	return pod
}

// podShared returns a pod with 3 containers that reference the named external resource claim.
func (b *Builder) PodExternalMultiple(externalClaimName string) *v1.Pod {
	pod := b.PodExternal(externalClaimName)
	pod.Spec.Containers = append(pod.Spec.Containers, *pod.Spec.Containers[0].DeepCopy(), *pod.Spec.Containers[0].DeepCopy())
	pod.Spec.Containers[1].Name += "-1"
	pod.Spec.Containers[2].Name += "-2"
	return pod
}

// GroupedPodWithClaims returns a pod that is a member of the given PodGroup.
func (b *Builder) GroupedPodWithClaims(podGroup *schedulingv1beta1.PodGroup) *v1.Pod {
	pod := b.Pod()
	pod.Spec.SchedulingGroup = &v1.PodSchedulingGroup{
		PodGroupName: &podGroup.Name,
	}
	for _, claim := range podGroup.Spec.ResourceClaims {
		pod.Spec.ResourceClaims = append(pod.Spec.ResourceClaims, v1.PodResourceClaim{
			Name:                      claim.Name,
			ResourceClaimName:         claim.ResourceClaimName,
			ResourceClaimTemplateName: claim.ResourceClaimTemplateName,
		})
		pod.Spec.Containers[0].Resources.Claims = append(pod.Spec.Containers[0].Resources.Claims, v1.ResourceClaim{Name: claim.Name})
	}
	return pod
}

// Workload creates a Workload with one PodGroupTemplate and no ResourceClaims.
func (b *Builder) Workload() *schedulingv1beta1.Workload {
	workload := &schedulingv1beta1.Workload{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: b.namespace,
			Name:      fmt.Sprintf("tester%s-%d", b.Driver.NameSuffix, b.workloadCounter),
		},
		Spec: schedulingv1beta1.WorkloadSpec{
			PodGroupTemplates: []schedulingv1beta1.PodGroupTemplate{
				{
					Name: "group",
					SchedulingPolicy: schedulingv1beta1.PodGroupSchedulingPolicy{
						Basic: &schedulingv1beta1.BasicSchedulingPolicy{},
					},
				},
			},
		},
	}
	b.workloadCounter++
	return workload
}

// WorkloadExternal creates a Workload with one PodGroupTemplate that refers to
// one ResourceClaim with the given name.
func (b *Builder) WorkloadExternal(externalClaimName string) *schedulingv1beta1.Workload {
	workload := b.Workload()
	workload.Spec.PodGroupTemplates[0].ResourceClaims = []schedulingv1beta1.PodGroupResourceClaim{
		{
			Name:              "resource-claim",
			ResourceClaimName: &externalClaimName,
		},
	}
	return workload
}

// WorkloadInline creates a ResourceClaimTemplate and a Workload with one
// PodGroupTemplate that refers to that ResourceClaimTemplate.
func (b *Builder) WorkloadInline() (*schedulingv1beta1.Workload, *resourceapi.ResourceClaimTemplate) {
	workload := b.Workload()
	podGroupClaimName := "my-inline-claim"
	workload.Spec.PodGroupTemplates[0].ResourceClaims = []schedulingv1beta1.PodGroupResourceClaim{
		{
			Name:                      podGroupClaimName,
			ResourceClaimTemplateName: new(workload.Name),
		},
	}
	template := &resourceapi.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      workload.Name,
			Namespace: workload.Namespace,
		},
		Spec: resourceapi.ResourceClaimTemplateSpec{
			Spec: b.ClaimSpec(),
		},
	}
	return workload, template
}

// PodGroup returns a simple PodGroup owned by the given Workload with no
// resource claims.
func (b *Builder) PodGroup(workload *schedulingv1beta1.Workload, template schedulingv1beta1.PodGroupTemplate) *schedulingv1beta1.PodGroup {
	podGroup := &schedulingv1beta1.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: b.namespace,
			Name:      fmt.Sprintf("%s-%s-%d", workload.Name, template.Name, b.podGroupCounter),
		},
		Spec: schedulingv1beta1.PodGroupSpec{
			WorkloadRef: &schedulingv1beta1.WorkloadReference{
				WorkloadName: workload.Name,
				TemplateName: template.Name,
			},
			SchedulingPolicy: template.SchedulingPolicy,
			ResourceClaims:   template.ResourceClaims,
		},
	}
	b.podGroupCounter++
	return podGroup
}

// Create takes a bunch of objects and calls their Create function.
func (b *Builder) Create(tCtx ktesting.TContext, objs ...klog.KMetadata) []klog.KMetadata {
	tCtx.Helper()
	cleanupCtx := tCtx.CleanupCtx
	if b.SkipCleanup {
		cleanupCtx = func(func(tCtx ktesting.TContext)) {}
	}

	var createdObjs []klog.KMetadata
	for _, obj := range objs {
		tCtx.Logf("Creating %T %s", obj, obj.GetName())
		var err error
		var createdObj klog.KMetadata
		switch obj := obj.(type) {
		case *resourceapi.DeviceClass:
			createdObj, err = b.ClientV1(tCtx).DeviceClasses().Create(tCtx, obj, metav1.CreateOptions{})
			cleanupCtx(func(tCtx ktesting.TContext) {
				err := b.ClientV1(tCtx).DeviceClasses().Delete(tCtx, createdObj.GetName(), metav1.DeleteOptions{})
				tCtx.ExpectNoError(err, "delete device class")
			})
		case *v1.Pod:
			createdObj, err = tCtx.Client().CoreV1().Pods(b.namespace).Create(tCtx, obj, metav1.CreateOptions{})
		case *v1.ResourceQuota:
			createdObj, err = tCtx.Client().CoreV1().ResourceQuotas(b.namespace).Create(tCtx, obj, metav1.CreateOptions{})
		case *v1.ConfigMap:
			createdObj, err = tCtx.Client().CoreV1().ConfigMaps(b.namespace).Create(tCtx, obj, metav1.CreateOptions{})
		case *resourceapi.ResourceClaim:
			createdObj, err = b.ClientV1(tCtx).ResourceClaims(b.namespace).Create(tCtx, obj, metav1.CreateOptions{})
		case *resourcev1beta1.ResourceClaim:
			createdObj, err = tCtx.Client().ResourceV1beta1().ResourceClaims(b.namespace).Create(tCtx, obj, metav1.CreateOptions{})
		case *resourcev1beta2.ResourceClaim:
			createdObj, err = tCtx.Client().ResourceV1beta2().ResourceClaims(b.namespace).Create(tCtx, obj, metav1.CreateOptions{})
		case *resourceapi.ResourceClaimTemplate:
			createdObj, err = b.ClientV1(tCtx).ResourceClaimTemplates(b.namespace).Create(tCtx, obj, metav1.CreateOptions{})
		case *resourcev1beta1.ResourceClaimTemplate:
			createdObj, err = tCtx.Client().ResourceV1beta1().ResourceClaimTemplates(b.namespace).Create(tCtx, obj, metav1.CreateOptions{})
		case *resourcev1beta2.ResourceClaimTemplate:
			createdObj, err = tCtx.Client().ResourceV1beta2().ResourceClaimTemplates(b.namespace).Create(tCtx, obj, metav1.CreateOptions{})
		case *resourceapi.ResourceSlice:
			createdObj, err = b.ClientV1(tCtx).ResourceSlices().Create(tCtx, obj, metav1.CreateOptions{})
			cleanupCtx(func(tCtx ktesting.TContext) {
				err := b.ClientV1(tCtx).ResourceSlices().Delete(tCtx, createdObj.GetName(), metav1.DeleteOptions{})
				tCtx.ExpectNoError(err, "delete node resource slice")
			})
		case *resourcealphaapi.DeviceTaintRule:
			createdObj, err = tCtx.Client().ResourceV1alpha3().DeviceTaintRules().Create(tCtx, obj, metav1.CreateOptions{})
			cleanupCtx(func(tCtx ktesting.TContext) {
				err := tCtx.Client().ResourceV1alpha3().DeviceTaintRules().Delete(tCtx, createdObj.GetName(), metav1.DeleteOptions{})
				tCtx.ExpectNoError(err, "delete DeviceTaintRule")
			})
		case *resourcev1beta2.DeviceTaintRule:
			createdObj, err = tCtx.Client().ResourceV1beta2().DeviceTaintRules().Create(tCtx, obj, metav1.CreateOptions{})
			cleanupCtx(func(tCtx ktesting.TContext) {
				err := tCtx.Client().ResourceV1beta2().DeviceTaintRules().Delete(tCtx, createdObj.GetName(), metav1.DeleteOptions{})
				tCtx.ExpectNoError(err, "delete DeviceTaintRule")
			})
		case *resourceapi.DeviceTaintRule:
			createdObj, err = tCtx.Client().ResourceV1().DeviceTaintRules().Create(tCtx, obj, metav1.CreateOptions{})
			cleanupCtx(func(tCtx ktesting.TContext) {
				err := tCtx.Client().ResourceV1().DeviceTaintRules().Delete(tCtx, createdObj.GetName(), metav1.DeleteOptions{})
				tCtx.ExpectNoError(err, "delete DeviceTaintRule")
			})
		case *appsv1.DaemonSet:
			createdObj, err = tCtx.Client().AppsV1().DaemonSets(b.namespace).Create(tCtx, obj, metav1.CreateOptions{})
			// Cleanup not really needed, but speeds up namespace shutdown.
			cleanupCtx(func(tCtx ktesting.TContext) {
				err := tCtx.Client().AppsV1().DaemonSets(b.namespace).Delete(tCtx, obj.Name, metav1.DeleteOptions{})
				tCtx.ExpectNoError(err, "delete daemonset")
			})
		case *schedulingv1beta1.Workload:
			createdObj, err = tCtx.Client().SchedulingV1beta1().Workloads(b.namespace).Create(tCtx, obj, metav1.CreateOptions{})
		case *schedulingv1beta1.PodGroup:
			createdObj, err = tCtx.Client().SchedulingV1beta1().PodGroups(b.namespace).Create(tCtx, obj, metav1.CreateOptions{})
		default:
			tCtx.Fatalf("internal error, unsupported type %T", obj)
		}
		tCtx.ExpectNoError(err, "create %T", obj)
		createdObjs = append(createdObjs, createdObj)
	}
	return createdObjs
}

func (b *Builder) DeletePodAndWaitForNotFound(tCtx ktesting.TContext, pod *v1.Pod) {
	tCtx.Helper()
	err := tCtx.Client().CoreV1().Pods(b.namespace).Delete(tCtx, pod.Name, metav1.DeleteOptions{})
	tCtx.ExpectNoError(err, "delete %T", pod)
	/* TODO: add timeouts to TContext? */
	err = e2epod.WaitForPodNotFoundInNamespace(tCtx, tCtx.Client(), pod.Name, pod.Namespace, 5*time.Minute /* former b.f.Timeouts.PodDelete */)
	tCtx.ExpectNoError(err, "terminate %T", pod)
}

// TestPod runs pod and checks if container logs contain expected environment variables
func (b *Builder) TestPod(tCtx ktesting.TContext, pod *v1.Pod, env ...string) {
	tCtx.Helper()

	if !b.Driver.WithKubelet {
		// Less testing when we cannot rely on the kubelet to actually run the pod.
		err := e2epod.WaitForPodScheduled(tCtx, tCtx.Client(), pod.Namespace, pod.Name)
		tCtx.ExpectNoError(err, "schedule pod")
		return
	}

	err := e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), pod)
	tCtx.ExpectNoError(err, "start pod")

	if len(env) == 0 {
		_, env = b.ParametersEnv()
	}
	for _, container := range pod.Spec.Containers {
		TestContainerEnv(tCtx, pod, container.Name, false, env...)
	}
}

// envLineRE matches env output with variables set by test/e2e/dra/test-driver.
var envLineRE = regexp.MustCompile(`^(?:admin|user|claim)_[a-zA-Z0-9_]*=.*$`)

func TestContainerEnv(tCtx ktesting.TContext, pod *v1.Pod, containerName string, fullMatch bool, env ...string) {
	tCtx.Helper()
	stdout, stderr, err := e2epod.Exec(tCtx, e2epod.ExecOptions{
		Command:       []string{"env"},
		Namespace:     pod.Namespace,
		PodName:       pod.Name,
		ContainerName: containerName,
		CaptureStdout: true,
		CaptureStderr: true,
		Quiet:         true,
	})
	tCtx.ExpectNoError(err, fmt.Sprintf("get env output for container %s", containerName))
	tCtx.Expect(stderr).To(gomega.BeEmpty(), fmt.Sprintf("env stderr for container %s", containerName))
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
		tCtx.Expect(actualEnv).To(gomega.Equal(expectEnv), fmt.Sprintf("container %s env output:\n%s", containerName, stdout))
	} else {
		for i := 0; i < len(env); i += 2 {
			envStr := fmt.Sprintf("%s=%s\n", env[i], env[i+1])
			tCtx.Expect(stdout).To(gomega.ContainSubstring(envStr), fmt.Sprintf("container %s env variables", containerName))
		}
	}
}

func NewBuilder(f *framework.Framework, driver *Driver) *Builder {
	b := &Builder{Driver: driver}
	ginkgo.BeforeEach(func() {
		b.setUp(f.TContext(context.Background()))
	})
	return b
}

func NewBuilderNow(tCtx ktesting.TContext, driver *Driver) *Builder {
	b := &Builder{Driver: driver}
	b.setUp(tCtx)
	return b
}

func (b *Builder) setUp(tCtx ktesting.TContext) {
	b.namespace = tCtx.Namespace()
	b.podCounter = 0
	b.workloadCounter = 0
	b.podGroupCounter = 0
	b.claimCounter = 0
	b.Create(tCtx, b.Class().DeviceClass)
	tCtx.CleanupCtx(b.tearDown)
}

// ClientV1 returns a wrapper for client-go which provides the V1 API on top of whatever is enabled in the cluster.
func (b *Builder) ClientV1(tCtx ktesting.TContext) cgoresource.ResourceV1Interface {
	return draclient.New(tCtx.Client())
}

func (b *Builder) tearDown(tCtx ktesting.TContext) {
	client := b.ClientV1(tCtx)

	// Before we allow the namespace and all objects in it do be deleted by
	// the framework, we must ensure that test pods and the claims that
	// they use are deleted. Otherwise the driver might get deleted first,
	// in which case deleting the claims won't work anymore.
	tCtx.Log("delete pods, podgroups, and claims")
	pods, err := b.listTestPods(tCtx)
	tCtx.ExpectNoError(err, "list pods")
	for _, pod := range pods {
		if pod.DeletionTimestamp != nil {
			continue
		}
		tCtx.Logf("Deleting %T %s", &pod, klog.KObj(&pod))
		options := metav1.DeleteOptions{}
		if !b.Driver.WithRealNodes {
			// Force-delete, no kubelet.
			options.GracePeriodSeconds = ptr.To(int64(0))
		}
		err := tCtx.Client().CoreV1().Pods(b.namespace).Delete(tCtx, pod.Name, options)
		if !apierrors.IsNotFound(err) {
			tCtx.ExpectNoError(err, "delete pod")
		}
	}
	tCtx.Eventually(func(tCtx ktesting.TContext) ([]v1.Pod, error) {
		return b.listTestPods(tCtx)
	}).WithTimeout(time.Minute).Should(gomega.BeEmpty(), "remaining pods despite deletion")

	// Clean up PodGroups to release claims allocated for them.
	podGroups, err := b.listTestPodGroups(tCtx)
	tCtx.ExpectNoError(err, "list podgroups")
	for _, podGroup := range podGroups {
		if podGroup.DeletionTimestamp != nil {
			continue
		}
		tCtx.Logf("Deleting %T %s", &podGroup, klog.KObj(&podGroup))
		err := tCtx.Client().SchedulingV1beta1().PodGroups(b.namespace).Delete(tCtx, podGroup.Name, metav1.DeleteOptions{})
		if !apierrors.IsNotFound(err) {
			tCtx.ExpectNoError(err, "delete podgroup")
		}
	}
	tCtx.Eventually(func(tCtx ktesting.TContext) ([]schedulingv1beta1.PodGroup, error) {
		return b.listTestPodGroups(tCtx)
	}).WithTimeout(time.Minute).Should(gomega.BeEmpty(), "remaining podgroups despite deletion")

	claims, err := b.ClientV1(tCtx).ResourceClaims(b.namespace).List(tCtx, metav1.ListOptions{})
	tCtx.ExpectNoError(err, "get resource claims")
	for _, claim := range claims.Items {
		if claim.DeletionTimestamp != nil {
			continue
		}
		tCtx.Logf("Deleting %T %s", &claim, klog.KObj(&claim))
		err := client.ResourceClaims(b.namespace).Delete(tCtx, claim.Name, metav1.DeleteOptions{})
		if !apierrors.IsNotFound(err) {
			tCtx.ExpectNoError(err, "delete claim")
		}
	}

	for host, plugin := range b.Driver.Nodes {
		tCtx.Logf("Waiting for resources on %s to be unprepared", host)
		tCtx.Eventually(func(ktesting.TContext) []app.ClaimID { return plugin.GetPreparedResources() }).WithTimeout(time.Minute).Should(gomega.BeEmpty(), "prepared claims on host %s", host)
	}

	tCtx.Log("waiting for claims to be deallocated and deleted")
	tCtx.Eventually(func(tCtx ktesting.TContext) (*resourceapi.ResourceClaimList, error) {
		return client.ResourceClaims(tCtx.Namespace()).List(tCtx, metav1.ListOptions{})
	}).WithTimeout(time.Minute).Should(gomega.HaveField("Items", gomega.BeEmpty()), "claims in the namespaces")
}

func (b *Builder) listTestPods(tCtx ktesting.TContext) ([]v1.Pod, error) {
	pods, err := tCtx.Client().CoreV1().Pods(b.namespace).List(tCtx, metav1.ListOptions{})
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

func (b *Builder) listTestPodGroups(tCtx ktesting.TContext) ([]schedulingv1beta1.PodGroup, error) {
	podGroups, err := tCtx.Client().SchedulingV1beta1().PodGroups(b.namespace).List(tCtx, metav1.ListOptions{})
	if apierrors.IsNotFound(err) {
		// API is disabled
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return podGroups.Items, nil
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
		for i := range maxAllocations {
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

// PartitionProfileAttribute is the fully qualified device attribute whose value
// labels each device's partition type in PartitionableResources.
const PartitionProfileAttribute = resourceapi.FullyQualifiedName("dra.e2e.example.com/profile")

// PartitionableResources publishes one pool "partitioned" split into a
// shared-counter slice and a device slice whose devices consume those counters.
// Each device carries the PartitionProfileAttribute ("Full"/"Half"). When
// withPartitionType is true the device slice declares PartitionTypeAttribute,
// which opts the pool into the typed partitionSummary view; otherwise the pool
// falls back to the counterSets view. The attribute goes only on the device
// slice: it may not be set on a slice without counter-consuming devices.
// Node-selected for control-plane use.
func PartitionableResources(withPartitionType bool) driverResourcesGenFunc {
	return func(nodes *Nodes) map[string]resourceslice.DriverResources {
		full := "Full"
		half := "Half"
		devices := []resourceapi.Device{
			partitionDevice("full", full, "8"),
			partitionDevice("half-a", half, "4"),
			partitionDevice("half-b", half, "4"),
		}
		counterSlice := resourceslice.Slice{
			SharedCounters: []resourceapi.CounterSet{{
				Name:     "gpu-0",
				Counters: map[string]resourceapi.SharedCounter{"memory": {Value: resource.NewQuantity(8, resource.BinarySI)}},
			}},
		}
		deviceSlice := resourceslice.Slice{Devices: devices}
		if withPartitionType {
			attr := PartitionProfileAttribute
			deviceSlice.PartitionTypeAttribute = &attr
		}
		return map[string]resourceslice.DriverResources{
			multiHostDriverResources: {
				Pools: map[string]resourceslice.Pool{
					"partitioned": {
						Slices:       []resourceslice.Slice{counterSlice, deviceSlice},
						NodeSelector: hostnameSelector(nodes),
						Generation:   1,
					},
				},
			},
		}
	}
}

// partitionDevice builds a device that consumes "memory" counters from gpu-0 and
// carries the partition-type attribute.
func partitionDevice(name, profile, memory string) resourceapi.Device {
	return resourceapi.Device{
		Name:       name,
		Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{resourceapi.QualifiedName(PartitionProfileAttribute): {StringValue: &profile}},
		ConsumesCounters: []resourceapi.DeviceCounterConsumption{{
			CounterSet: "gpu-0",
			Counters:   map[string]resourceapi.ConsumeCounter{"memory": {Value: new(resource.MustParse(memory))}},
		}},
	}
}

// ShareableResources publishes one pool "shareable" with count shareable devices
// (AllowMultipleAllocations), each carrying "memory" capacity, exercising the
// shareableSummary view. Node-selected for control-plane use.
func ShareableResources(count int) driverResourcesGenFunc {
	return func(nodes *Nodes) map[string]resourceslice.DriverResources {
		devices := make([]resourceapi.Device, count)
		for i := range count {
			devices[i] = resourceapi.Device{
				Name:                     fmt.Sprintf("shared-%d", i),
				AllowMultipleAllocations: new(true),
				Capacity:                 map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{"memory": {Value: resource.MustParse("16")}},
			}
		}
		return map[string]resourceslice.DriverResources{
			multiHostDriverResources: {
				Pools: map[string]resourceslice.Pool{
					"shareable": {
						Slices:       []resourceslice.Slice{{Devices: devices}},
						NodeSelector: hostnameSelector(nodes),
						Generation:   1,
					},
				},
			},
		}
	}
}

// hostnameSelector selects all of the test's nodes by hostname.
func hostnameSelector(nodes *Nodes) *v1.NodeSelector {
	return &v1.NodeSelector{
		NodeSelectorTerms: []v1.NodeSelectorTerm{{
			MatchExpressions: []v1.NodeSelectorRequirement{{
				Key:      "kubernetes.io/hostname",
				Operator: v1.NodeSelectorOpIn,
				Values:   nodes.NodeNames,
			}},
		}},
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
			for i := range maxAllocations {
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
							},
							{
								Devices: devices,
							},
						},
					},
				},
			},
		}
	}
}
