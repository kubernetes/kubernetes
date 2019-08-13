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
	"fmt"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/framework/volume"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
)

type ephemeralTestSuite struct {
	tsInfo TestSuiteInfo
}

var _ TestSuite = &ephemeralTestSuite{}

// InitEphemeralTestSuite returns ephemeralTestSuite that implements TestSuite interface
func InitEphemeralTestSuite() TestSuite {
	return &ephemeralTestSuite{
		tsInfo: TestSuiteInfo{
			name: "ephemeral [Feature:CSIInlineVolume]",
			testPatterns: []testpatterns.TestPattern{
				{
					Name:    "inline ephemeral CSI volume",
					VolType: testpatterns.CSIInlineVolume,
				},
			},
		},
	}
}

func (p *ephemeralTestSuite) getTestSuiteInfo() TestSuiteInfo {
	return p.tsInfo
}

func (p *ephemeralTestSuite) skipRedundantSuite(driver TestDriver, pattern testpatterns.TestPattern) {
}

func (p *ephemeralTestSuite) defineTests(driver TestDriver, pattern testpatterns.TestPattern) {
	type local struct {
		config      *PerTestConfig
		testCleanup func()

		testCase *EphemeralTest
	}
	var (
		dInfo   = driver.GetDriverInfo()
		eDriver EphemeralTestDriver
		l       local
	)

	ginkgo.BeforeEach(func() {
		ok := false
		eDriver, ok = driver.(EphemeralTestDriver)
		if !ok {
			framework.Skipf("Driver %s doesn't support ephemeral inline volumes -- skipping", dInfo.Name)
		}
	})

	// This intentionally comes after checking the preconditions because it
	// registers its own BeforeEach which creates the namespace. Beware that it
	// also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewDefaultFramework("ephemeral")

	init := func() {
		l = local{}

		// Now do the more expensive test initialization.
		l.config, l.testCleanup = driver.PrepareTest(f)
		l.testCase = &EphemeralTest{
			Client:     l.config.Framework.ClientSet,
			Namespace:  f.Namespace.Name,
			DriverName: eDriver.GetCSIDriverName(l.config),
			Node:       framework.NodeSelection{Name: l.config.ClientNodeName},
			GetVolumeAttributes: func(volumeNumber int) map[string]string {
				return eDriver.GetVolumeAttributes(l.config, volumeNumber)
			},
		}
	}

	cleanup := func() {
		if l.testCleanup != nil {
			l.testCleanup()
			l.testCleanup = nil
		}
	}

	ginkgo.It("should create inline ephemeral volume", func() {
		init()
		defer cleanup()

		l.testCase.TestEphemeral()
	})
}

// EphemeralTest represents parameters to be used by tests for inline volumes.
// Not all parameters are used by all tests.
type EphemeralTest struct {
	Client     clientset.Interface
	Namespace  string
	DriverName string
	Node       framework.NodeSelection

	// GetVolumeAttributes returns the volume attributes for a
	// certain inline ephemeral volume, enumerated starting with
	// #0. Some tests might require more than one volume. They can
	// all be the same or different, depending what the driver supports
	// and/or wants to test.
	GetVolumeAttributes func(volumeNumber int) map[string]string

	// RunningPodCheck is invoked while a pod using an inline volume is running.
	// It can execute additional checks on the pod and its volume(s). Any data
	// returned by it is passed to StoppedPodCheck.
	RunningPodCheck func(pod *v1.Pod) interface{}

	// StoppedPodCheck is invoked after ensuring that the pod is gone.
	// It is passed the data gather by RunningPodCheck or nil if that
	// isn't defined and then can do additional checks on the node,
	// like for example verifying that the ephemeral volume was really
	// removed. How to do such a check is driver-specific and not
	// covered by the generic storage test suite.
	StoppedPodCheck func(nodeName string, runningPodData interface{})
}

// TestEphemeral tests pod creation with one ephemeral volume.
func (t EphemeralTest) TestEphemeral() {
	client := t.Client
	gomega.Expect(client).NotTo(gomega.BeNil(), "EphemeralTest.Client is required")
	gomega.Expect(t.GetVolumeAttributes).NotTo(gomega.BeNil(), "EphemeralTest.GetVolumeAttributes is required")
	gomega.Expect(t.DriverName).NotTo(gomega.BeEmpty(), "EphemeralTest.DriverName is required")

	ginkgo.By(fmt.Sprintf("checking the requested inline volume exists in the pod running on node %+v", t.Node))
	command := "mount | grep /mnt/test"
	pod := StartInPodWithInlineVolume(client, t.Namespace, "inline-volume-tester", command,
		v1.CSIVolumeSource{
			Driver:           t.DriverName,
			VolumeAttributes: t.GetVolumeAttributes(0),
		},
		t.Node)
	defer func() {
		// pod might be nil now.
		StopPod(client, pod)
	}()
	framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespaceSlow(client, pod.Name, pod.Namespace), "waiting for pod with inline volume")
	runningPod, err := client.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "get pod")
	actualNodeName := runningPod.Spec.NodeName

	// Run the checker of the running pod.
	var runningPodData interface{}
	if t.RunningPodCheck != nil {
		runningPodData = t.RunningPodCheck(pod)
	}

	StopPod(client, pod)
	pod = nil // Don't stop twice.

	if t.StoppedPodCheck != nil {
		t.StoppedPodCheck(actualNodeName, runningPodData)
	}
}

// StartInPodWithInlineVolume starts a command in a pod with given volume mounted to /mnt/test directory.
// The caller is responsible for checking the pod and deleting it.
func StartInPodWithInlineVolume(c clientset.Interface, ns, podName, command string, csiVolume v1.CSIVolumeSource, node framework.NodeSelection) *v1.Pod {
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
			NodeName:     node.Name,
			NodeSelector: node.Selector,
			Affinity:     node.Affinity,
			Containers: []v1.Container{
				{
					Name:    "csi-volume-tester",
					Image:   volume.GetTestImage(framework.BusyBoxImage),
					Command: volume.GenerateScriptCmd(command),
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
					Name: "my-volume",
					VolumeSource: v1.VolumeSource{
						CSI: &csiVolume,
					},
				},
			},
		},
	}

	pod, err := c.CoreV1().Pods(ns).Create(pod)
	framework.ExpectNoError(err, "failed to create pod")
	return pod
}

// CSIInlineVolumesEnabled checks whether the running cluster has the CSIInlineVolumes feature gate enabled.
// It does that by trying to create a pod that uses that feature.
func CSIInlineVolumesEnabled(c clientset.Interface, ns string) (bool, error) {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "csi-inline-volume-",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "csi-volume-tester",
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
					Name: "my-volume",
					VolumeSource: v1.VolumeSource{
						CSI: &v1.CSIVolumeSource{
							Driver: "no-such-driver.example.com",
						},
					},
				},
			},
		},
	}

	pod, err := c.CoreV1().Pods(ns).Create(pod)

	switch {
	case err == nil:
		// Pod was created, feature supported.
		StopPod(c, pod)
		return true, nil
	case errors.IsInvalid(err):
		// "Invalid" because it uses a feature that isn't supported.
		return false, nil
	default:
		// Unexpected error.
		return false, err
	}
}
