/*
Copyright 2019 The Kubernetes Authors.

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

package testsuites

import (
	"context"
	"fmt"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	admissionapi "k8s.io/pod-security-admission/api"
)

type ephemeralTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

// InitCustomEphemeralTestSuite returns ephemeralTestSuite that implements TestSuite interface
// using custom test patterns
func InitCustomEphemeralTestSuite(patterns []storageframework.TestPattern) storageframework.TestSuite {
	return &ephemeralTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name:         "ephemeral",
			TestPatterns: patterns,
		},
	}
}

// GenericEphemeralTestPatterns returns the test patterns for
// generic ephemeral inline volumes.
func GenericEphemeralTestPatterns() []storageframework.TestPattern {
	genericLateBinding := storageframework.DefaultFsGenericEphemeralVolume
	genericLateBinding.Name += " (late-binding)"
	genericLateBinding.BindingMode = storagev1.VolumeBindingWaitForFirstConsumer

	genericImmediateBinding := storageframework.DefaultFsGenericEphemeralVolume
	genericImmediateBinding.Name += " (immediate-binding)"
	genericImmediateBinding.BindingMode = storagev1.VolumeBindingImmediate

	return []storageframework.TestPattern{
		genericLateBinding,
		genericImmediateBinding,
		storageframework.BlockVolModeGenericEphemeralVolume,
	}
}

// CSIEphemeralTestPatterns returns the test patterns for
// CSI ephemeral inline volumes.
func CSIEphemeralTestPatterns() []storageframework.TestPattern {
	return []storageframework.TestPattern{
		storageframework.DefaultFsCSIEphemeralVolume,
	}
}

// AllEphemeralTestPatterns returns all pre-defined test patterns for
// generic and CSI ephemeral inline volumes.
func AllEphemeralTestPatterns() []storageframework.TestPattern {
	return append(GenericEphemeralTestPatterns(), CSIEphemeralTestPatterns()...)
}

// InitEphemeralTestSuite returns ephemeralTestSuite that implements TestSuite interface
// using test suite default patterns
func InitEphemeralTestSuite() storageframework.TestSuite {
	return InitCustomEphemeralTestSuite(AllEphemeralTestPatterns())
}

func (p *ephemeralTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return p.tsInfo
}

func (p *ephemeralTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	if pattern.VolMode == v1.PersistentVolumeBlock {
		skipTestIfBlockNotSupported(driver)
	}
}

func (p *ephemeralTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	type local struct {
		config *storageframework.PerTestConfig

		testCase *EphemeralTest
		resource *storageframework.VolumeResource
	}
	var (
		dInfo   = driver.GetDriverInfo()
		eDriver storageframework.EphemeralTestDriver
		l       local
	)

	// Beware that it also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewFrameworkWithCustomTimeouts("ephemeral", storageframework.GetDriverTimeouts(driver))
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	init := func(ctx context.Context) {
		if pattern.VolType == storageframework.CSIInlineVolume {
			eDriver, _ = driver.(storageframework.EphemeralTestDriver)
		}
		if pattern.VolType == storageframework.GenericEphemeralVolume {
			// The GenericEphemeralVolume feature is GA, but
			// perhaps this test is run against an older Kubernetes
			// where the feature might be disabled.
			enabled, err := GenericEphemeralVolumesEnabled(ctx, f.ClientSet, f.Timeouts, f.Namespace.Name)
			framework.ExpectNoError(err, "check GenericEphemeralVolume feature")
			if !enabled {
				e2eskipper.Skipf("Cluster doesn't support %q volumes -- skipping", pattern.VolType)
			}
		}
		// A driver might support the Topology capability which is incompatible with the VolumeBindingMode immediate because
		// volumes might be provisioned immediately in a different zone to where the workload is located.
		if pattern.BindingMode == storagev1.VolumeBindingImmediate && len(dInfo.TopologyKeys) > 0 {
			e2eskipper.Skipf("VolumeBindingMode immediate is not compatible with a multi-topology environment.")
		}

		l = local{}

		if !driver.GetDriverInfo().Capabilities[storageframework.CapOnlineExpansion] {
			pattern.AllowExpansion = false
		}

		// Now do the more expensive test initialization.
		l.config = driver.PrepareTest(ctx, f)
		l.resource = storageframework.CreateVolumeResource(ctx, driver, l.config, pattern, e2evolume.SizeRange{})

		switch pattern.VolType {
		case storageframework.CSIInlineVolume:
			l.testCase = &EphemeralTest{
				Client:     l.config.Framework.ClientSet,
				Timeouts:   f.Timeouts,
				Namespace:  f.Namespace.Name,
				DriverName: eDriver.GetCSIDriverName(l.config),
				Node:       l.config.ClientNodeSelection,
				GetVolume: func(volumeNumber int) (map[string]string, bool, bool) {
					return eDriver.GetVolume(l.config, volumeNumber)
				},
			}
		case storageframework.GenericEphemeralVolume:
			l.testCase = &EphemeralTest{
				Client:    l.config.Framework.ClientSet,
				Timeouts:  f.Timeouts,
				Namespace: f.Namespace.Name,
				Node:      l.config.ClientNodeSelection,
				VolSource: l.resource.VolSource,
			}
		}
	}

	cleanup := func(ctx context.Context) {
		var cleanUpErrs []error
		cleanUpErrs = append(cleanUpErrs, l.resource.CleanupResource(ctx))
		err := utilerrors.NewAggregate(cleanUpErrs)
		framework.ExpectNoError(err, "while cleaning up")
	}

	ginkgo.It("should create read-only inline ephemeral volume", func(ctx context.Context) {
		if pattern.VolMode == v1.PersistentVolumeBlock {
			e2eskipper.Skipf("raw block volumes cannot be read-only")
		}

		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		l.testCase.ReadOnly = true
		l.testCase.RunningPodCheck = func(ctx context.Context, pod *v1.Pod) interface{} {
			command := "mount | grep /mnt/test | grep ro,"
			if framework.NodeOSDistroIs("windows") {
				// attempt to create a dummy file and expect for it not to be created
				command = "ls /mnt/test* && (touch /mnt/test-0/hello-world || true) && [ ! -f /mnt/test-0/hello-world ]"
			}
			e2epod.VerifyExecInPodSucceed(ctx, f, pod, command)
			return nil
		}
		l.testCase.TestEphemeral(ctx)
	})

	ginkgo.It("should create read/write inline ephemeral volume", func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		l.testCase.ReadOnly = false
		l.testCase.RunningPodCheck = func(ctx context.Context, pod *v1.Pod) interface{} {
			command := "mount | grep /mnt/test | grep rw,"
			if framework.NodeOSDistroIs("windows") {
				// attempt to create a dummy file and expect for it to be created
				command = "ls /mnt/test* && touch /mnt/test-0/hello-world && [ -f /mnt/test-0/hello-world ]"
			}
			if pattern.VolMode == v1.PersistentVolumeBlock {
				command = "if ! [ -b /mnt/test-0 ]; then echo /mnt/test-0 is not a block device; exit 1; fi"
			}
			e2epod.VerifyExecInPodSucceed(ctx, f, pod, command)
			return nil
		}
		l.testCase.TestEphemeral(ctx)
	})

	ginkgo.It("should support expansion of pvcs created for ephemeral pvcs", func(ctx context.Context) {
		if pattern.VolType != storageframework.GenericEphemeralVolume {
			e2eskipper.Skipf("Skipping %s test for expansion", pattern.VolType)
		}

		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		if !driver.GetDriverInfo().Capabilities[storageframework.CapOnlineExpansion] {
			e2eskipper.Skipf("Driver %q does not support online volume expansion - skipping", driver.GetDriverInfo().Name)
		}

		l.testCase.ReadOnly = false
		l.testCase.RunningPodCheck = func(ctx context.Context, pod *v1.Pod) interface{} {
			podName := pod.Name
			framework.Logf("Running volume expansion checks %s", podName)

			outerPodVolumeSpecName := ""
			for i := range pod.Spec.Volumes {
				volume := pod.Spec.Volumes[i]
				if volume.Ephemeral != nil {
					outerPodVolumeSpecName = volume.Name
					break
				}
			}
			pvcName := fmt.Sprintf("%s-%s", podName, outerPodVolumeSpecName)
			pvc, err := f.ClientSet.CoreV1().PersistentVolumeClaims(pod.Namespace).Get(ctx, pvcName, metav1.GetOptions{})
			framework.ExpectNoError(err, "error getting ephemeral pvc")

			ginkgo.By("Expanding current pvc")
			currentPvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
			newSize := currentPvcSize.DeepCopy()
			newSize.Add(resource.MustParse("1Gi"))
			framework.Logf("currentPvcSize %s, requested new size %s", currentPvcSize.String(), newSize.String())

			newPVC, err := ExpandPVCSize(ctx, pvc, newSize, f.ClientSet)
			framework.ExpectNoError(err, "While updating pvc for more size")
			pvc = newPVC
			gomega.Expect(pvc).NotTo(gomega.BeNil())

			pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
			if pvcSize.Cmp(newSize) != 0 {
				framework.Failf("error updating pvc %s from %s to %s size", pvc.Name, currentPvcSize.String(), newSize.String())
			}

			ginkgo.By("Waiting for cloudprovider resize to finish")
			err = WaitForControllerVolumeResize(ctx, pvc, f.ClientSet, totalResizeWaitPeriod)
			framework.ExpectNoError(err, "While waiting for pvc resize to finish")

			ginkgo.By("Waiting for file system resize to finish")
			pvc, err = WaitForFSResize(ctx, pvc, f.ClientSet)
			framework.ExpectNoError(err, "while waiting for fs resize to finish")

			pvcConditions := pvc.Status.Conditions
			gomega.Expect(pvcConditions).To(gomega.BeEmpty(), "pvc should not have conditions")
			return nil
		}
		l.testCase.TestEphemeral(ctx)

	})

	ginkgo.It("should support two pods which have the same volume definition", func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		// We test in read-only mode if that is all that the driver supports,
		// otherwise read/write. For PVC, both are assumed to be false.
		shared := false
		readOnly := false
		if eDriver != nil {
			_, shared, readOnly = eDriver.GetVolume(l.config, 0)
		}

		l.testCase.RunningPodCheck = func(ctx context.Context, pod *v1.Pod) interface{} {
			// Create another pod with the same inline volume attributes.
			pod2 := StartInPodWithInlineVolume(ctx, f.ClientSet, f.Namespace.Name, "inline-volume-tester2", e2epod.InfiniteSleepCommand,
				[]v1.VolumeSource{pod.Spec.Volumes[0].VolumeSource},
				readOnly,
				l.testCase.Node)
			framework.ExpectNoError(e2epod.WaitTimeoutForPodRunningInNamespace(ctx, f.ClientSet, pod2.Name, pod2.Namespace, f.Timeouts.PodStartSlow), "waiting for second pod with inline volume")

			// If (and only if) we were able to mount
			// read/write and volume data is not shared
			// between pods, then we can check whether
			// data written in one pod is really not
			// visible in the other.
			if pattern.VolMode != v1.PersistentVolumeBlock && !readOnly && !shared {
				ginkgo.By("writing data in one pod and checking the second does not see it (it should get its own volume)")
				e2epod.VerifyExecInPodSucceed(ctx, f, pod, "touch /mnt/test-0/hello-world")
				e2epod.VerifyExecInPodSucceed(ctx, f, pod2, "[ ! -f /mnt/test-0/hello-world ]")
			}

			// TestEphemeral expects the pod to be fully deleted
			// when this function returns, so don't delay this
			// cleanup.
			StopPodAndDependents(ctx, f.ClientSet, f.Timeouts, pod2)

			return nil
		}

		l.testCase.TestEphemeral(ctx)
	})

	ginkgo.It("should support multiple inline ephemeral volumes", func(ctx context.Context) {
		if pattern.BindingMode == storagev1.VolumeBindingImmediate &&
			pattern.VolType == storageframework.GenericEphemeralVolume {
			e2eskipper.Skipf("Multiple generic ephemeral volumes with immediate binding may cause pod startup failures when the volumes get created in separate topology segments.")
		}

		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		l.testCase.NumInlineVolumes = 2
		l.testCase.TestEphemeral(ctx)
	})
}

// EphemeralTest represents parameters to be used by tests for inline volumes.
// Not all parameters are used by all tests.
type EphemeralTest struct {
	Client     clientset.Interface
	Timeouts   *framework.TimeoutContext
	Namespace  string
	DriverName string
	VolSource  *v1.VolumeSource
	Node       e2epod.NodeSelection

	// GetVolume returns the volume attributes for a
	// certain inline ephemeral volume, enumerated starting with
	// #0. Some tests might require more than one volume. They can
	// all be the same or different, depending what the driver supports
	// and/or wants to test.
	//
	// For each volume, the test driver can specify the
	// attributes, whether two pods using those attributes will
	// end up sharing the same backend storage (i.e. changes made
	// in one pod will be visible in the other), and whether
	// the volume can be mounted read/write or only read-only.
	GetVolume func(volumeNumber int) (attributes map[string]string, shared bool, readOnly bool)

	// RunningPodCheck is invoked while a pod using an inline volume is running.
	// It can execute additional checks on the pod and its volume(s). Any data
	// returned by it is passed to StoppedPodCheck.
	RunningPodCheck func(ctx context.Context, pod *v1.Pod) interface{}

	// StoppedPodCheck is invoked after ensuring that the pod is gone.
	// It is passed the data gather by RunningPodCheck or nil if that
	// isn't defined and then can do additional checks on the node,
	// like for example verifying that the ephemeral volume was really
	// removed. How to do such a check is driver-specific and not
	// covered by the generic storage test suite.
	StoppedPodCheck func(ctx context.Context, nodeName string, runningPodData interface{})

	// NumInlineVolumes sets the number of ephemeral inline volumes per pod.
	// Unset (= zero) is the same as one.
	NumInlineVolumes int

	// ReadOnly limits mounting to read-only.
	ReadOnly bool
}

// TestEphemeral tests pod creation with one ephemeral volume.
func (t EphemeralTest) TestEphemeral(ctx context.Context) {
	client := t.Client
	gomega.Expect(client).NotTo(gomega.BeNil(), "EphemeralTest.Client is required")

	ginkgo.By(fmt.Sprintf("checking the requested inline volume exists in the pod running on node %+v", t.Node))

	var volumes []v1.VolumeSource
	numVolumes := t.NumInlineVolumes
	if numVolumes == 0 {
		numVolumes = 1
	}
	for i := 0; i < numVolumes; i++ {
		var volume v1.VolumeSource
		switch {
		case t.GetVolume != nil:
			attributes, _, readOnly := t.GetVolume(i)
			if readOnly && !t.ReadOnly {
				e2eskipper.Skipf("inline ephemeral volume #%d is read-only, but the test needs a read/write volume", i)
			}
			volume = v1.VolumeSource{
				CSI: &v1.CSIVolumeSource{
					Driver:           t.DriverName,
					VolumeAttributes: attributes,
				},
			}
		case t.VolSource != nil:
			volume = *t.VolSource
		default:
			framework.Failf("EphemeralTest has neither GetVolume nor VolSource")
		}
		volumes = append(volumes, volume)
	}
	pod := StartInPodWithInlineVolume(ctx, client, t.Namespace, "inline-volume-tester", e2epod.InfiniteSleepCommand, volumes, t.ReadOnly, t.Node)
	defer func() {
		// pod might be nil now.
		StopPodAndDependents(ctx, client, t.Timeouts, pod)
	}()
	framework.ExpectNoError(e2epod.WaitTimeoutForPodRunningInNamespace(ctx, client, pod.Name, pod.Namespace, t.Timeouts.PodStartSlow), "waiting for pod with inline volume")
	runningPod, err := client.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "get pod")
	actualNodeName := runningPod.Spec.NodeName

	// Run the checker of the running pod.
	var runningPodData interface{}
	if t.RunningPodCheck != nil {
		runningPodData = t.RunningPodCheck(ctx, pod)
	}

	StopPodAndDependents(ctx, client, t.Timeouts, pod)
	pod = nil // Don't stop twice.

	// There should be no dangling PVCs in the namespace now. There might be for
	// generic ephemeral volumes, if something went wrong...
	pvcs, err := client.CoreV1().PersistentVolumeClaims(t.Namespace).List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err, "list PVCs")
	gomega.Expect(pvcs.Items).Should(gomega.BeEmpty(), "no dangling PVCs")

	if t.StoppedPodCheck != nil {
		t.StoppedPodCheck(ctx, actualNodeName, runningPodData)
	}
}

// StartInPodWithInlineVolume starts a command in a pod with given volume(s) mounted to /mnt/test-<number> directory.
// The caller is responsible for checking the pod and deleting it.
func StartInPodWithInlineVolume(ctx context.Context, c clientset.Interface, ns, podName, command string, volumes []v1.VolumeSource, readOnly bool, node e2epod.NodeSelection) *v1.Pod {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: podName + "-",
			Labels: map[string]string{
				"app": podName,
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "csi-volume-tester",
					Image: e2epod.GetDefaultTestImage(),
					// NOTE: /bin/sh works on both agnhost and busybox
					Command: []string{"/bin/sh", "-c", command},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
	e2epod.SetNodeSelection(&pod.Spec, node)

	for i, volume := range volumes {
		name := fmt.Sprintf("my-volume-%d", i)
		path := fmt.Sprintf("/mnt/test-%d", i)
		if volume.Ephemeral != nil && volume.Ephemeral.VolumeClaimTemplate.Spec.VolumeMode != nil &&
			*volume.Ephemeral.VolumeClaimTemplate.Spec.VolumeMode == v1.PersistentVolumeBlock {
			pod.Spec.Containers[0].VolumeDevices = append(pod.Spec.Containers[0].VolumeDevices,
				v1.VolumeDevice{
					Name:       name,
					DevicePath: path,
				})
		} else {
			pod.Spec.Containers[0].VolumeMounts = append(pod.Spec.Containers[0].VolumeMounts,
				v1.VolumeMount{
					Name:      name,
					MountPath: path,
					ReadOnly:  readOnly,
				})
		}
		pod.Spec.Volumes = append(pod.Spec.Volumes,
			v1.Volume{
				Name:         name,
				VolumeSource: volume,
			})
	}

	pod, err := c.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create pod")
	return pod
}

// CSIInlineVolumesEnabled checks whether the running cluster has the CSIInlineVolumes feature gate enabled.
// It does that by trying to create a pod that uses that feature.
func CSIInlineVolumesEnabled(ctx context.Context, c clientset.Interface, t *framework.TimeoutContext, ns string) (bool, error) {
	return VolumeSourceEnabled(ctx, c, t, ns, v1.VolumeSource{
		CSI: &v1.CSIVolumeSource{
			Driver: "no-such-driver.example.com",
		},
	})
}

// GenericEphemeralVolumesEnabled checks whether the running cluster has the GenericEphemeralVolume feature gate enabled.
// It does that by trying to create a pod that uses that feature.
func GenericEphemeralVolumesEnabled(ctx context.Context, c clientset.Interface, t *framework.TimeoutContext, ns string) (bool, error) {
	storageClassName := "no-such-storage-class"
	return VolumeSourceEnabled(ctx, c, t, ns, v1.VolumeSource{
		Ephemeral: &v1.EphemeralVolumeSource{
			VolumeClaimTemplate: &v1.PersistentVolumeClaimTemplate{
				Spec: v1.PersistentVolumeClaimSpec{
					StorageClassName: &storageClassName,
					AccessModes:      []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
					Resources: v1.VolumeResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceStorage: resource.MustParse("1Gi"),
						},
					},
				},
			},
		},
	})
}

// VolumeSourceEnabled checks whether a certain kind of volume source is enabled by trying
// to create a pod that uses it.
func VolumeSourceEnabled(ctx context.Context, c clientset.Interface, t *framework.TimeoutContext, ns string, volume v1.VolumeSource) (bool, error) {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "inline-volume-",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "volume-tester",
					Image: "no-such-registry/no-such-image",
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "my-volume",
							MountPath: "/mnt/test",
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			Volumes: []v1.Volume{
				{
					Name:         "my-volume",
					VolumeSource: volume,
				},
			},
		},
	}

	pod, err := c.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{})

	switch {
	case err == nil:
		// Pod was created, feature supported.
		StopPodAndDependents(ctx, c, t, pod)
		return true, nil
	case apierrors.IsInvalid(err):
		// "Invalid" because it uses a feature that isn't supported.
		return false, nil
	default:
		// Unexpected error.
		return false, err
	}
}
