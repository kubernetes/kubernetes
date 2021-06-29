// +build linux

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

package e2enode

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"k8s.io/apimachinery/pkg/fields"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubelettypes "k8s.io/kubernetes/pkg/kubelet/types"
	testutils "k8s.io/kubernetes/test/utils"
)

var _ = SIGDescribe("GracefulNodeShutdown [Serial] [NodeAlphaFeature:GracefulNodeShutdown]", func() {
	f := framework.NewDefaultFramework("graceful-node-shutdown")
	ginkgo.Context("when gracefully shutting down", func() {

		const (
			pollInterval                        = 1 * time.Second
			podStatusUpdateTimeout              = 5 * time.Second
			nodeStatusUpdateTimeout             = 10 * time.Second
			nodeShutdownGracePeriod             = 20 * time.Second
			nodeShutdownGracePeriodCriticalPods = 10 * time.Second
		)

		tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.FeatureGates = map[string]bool{
				string(features.GracefulNodeShutdown): true,
			}
			initialConfig.ShutdownGracePeriod = metav1.Duration{Duration: nodeShutdownGracePeriod}
			initialConfig.ShutdownGracePeriodCriticalPods = metav1.Duration{Duration: nodeShutdownGracePeriodCriticalPods}
		})

		ginkgo.BeforeEach(func() {
			ginkgo.By("Wait for the node to be ready")
			waitForNodeReady()
		})

		ginkgo.AfterEach(func() {
			ginkgo.By("Emitting Shutdown false signal; cancelling the shutdown")
			err := emitSignalPrepareForShutdown(false)
			framework.ExpectNoError(err)
		})

		ginkgo.It("should be able to gracefully shutdown pods with various grace periods", func() {
			nodeName := getNodeName(f)
			nodeSelector := fields.Set{
				"spec.nodeName": nodeName,
			}.AsSelector().String()

			// Define test pods
			pods := []*v1.Pod{
				getGracePeriodOverrideTestPod("period-120", nodeName, 120, false),
				getGracePeriodOverrideTestPod("period-5", nodeName, 5, false),
				getGracePeriodOverrideTestPod("period-critical-120", nodeName, 120, true),
				getGracePeriodOverrideTestPod("period-critical-5", nodeName, 5, true),
			}

			ginkgo.By("Creating batch pods")
			f.PodClient().CreateBatch(pods)

			list, err := f.PodClient().List(context.TODO(), metav1.ListOptions{
				FieldSelector: nodeSelector,
			})
			framework.ExpectNoError(err)
			framework.ExpectEqual(len(list.Items), len(pods), "the number of pods is not as expected")

			ginkgo.By("Verifying batch pods are running")
			for _, pod := range list.Items {
				if podReady, err := testutils.PodRunningReady(&pod); err != nil || !podReady {
					framework.Failf("Failed to start batch pod: %v", pod.Name)
				}
			}

			ginkgo.By("Emitting shutdown signal")
			err = emitSignalPrepareForShutdown(true)
			framework.ExpectNoError(err)

			ginkgo.By("Verifying that non-critical pods are shutdown")
			// Not critical pod should be shutdown
			gomega.Eventually(func() error {
				list, err = f.PodClient().List(context.TODO(), metav1.ListOptions{
					FieldSelector: nodeSelector,
				})
				if err != nil {
					return err
				}
				framework.ExpectEqual(len(list.Items), len(pods), "the number of pods is not as expected")

				for _, pod := range list.Items {
					if kubelettypes.IsCriticalPod(&pod) {
						if pod.Status.Phase != v1.PodRunning {
							framework.Logf("Expecting critcal pod to be running, but it's not currently. Pod: %q, Pod Status Phase: %q, Pod Status Reason: %q", pod.Name, pod.Status.Phase, pod.Status.Reason)
							return fmt.Errorf("critical pod should not be shutdown, phase: %s", pod.Status.Phase)
						}
					} else {
						if pod.Status.Phase != v1.PodFailed || pod.Status.Reason != "Shutdown" {
							framework.Logf("Expecting non-critcal pod to be shutdown, but it's not currently. Pod: %q, Pod Status Phase: %q, Pod Status Reason: %q", pod.Name, pod.Status.Phase, pod.Status.Reason)
							return fmt.Errorf("pod should be shutdown, phase: %s", pod.Status.Phase)
						}
					}
				}
				return nil
			}, podStatusUpdateTimeout, pollInterval).Should(gomega.BeNil())

			ginkgo.By("Verifying that all pods are shutdown")
			// All pod should be shutdown
			gomega.Eventually(func() error {
				list, err = f.PodClient().List(context.TODO(), metav1.ListOptions{
					FieldSelector: nodeSelector,
				})
				if err != nil {
					return err
				}
				framework.ExpectEqual(len(list.Items), len(pods), "the number of pods is not as expected")

				for _, pod := range list.Items {
					if pod.Status.Phase != v1.PodFailed || pod.Status.Reason != "Shutdown" {
						framework.Logf("Expecting pod to be shutdown, but it's not currently: Pod: %q, Pod Status Phase: %q, Pod Status Reason: %q", pod.Name, pod.Status.Phase, pod.Status.Reason)
						return fmt.Errorf("pod should be shutdown, phase: %s", pod.Status.Phase)
					}
				}
				return nil
			},
				// Critical pod starts shutdown after (nodeShutdownGracePeriod-nodeShutdownGracePeriodCriticalPods)
				podStatusUpdateTimeout+(nodeShutdownGracePeriod-nodeShutdownGracePeriodCriticalPods),
				pollInterval).Should(gomega.BeNil())
		})

		ginkgo.It("should be able to handle a cancelled shutdown", func() {
			ginkgo.By("Emitting Shutdown signal")
			err := emitSignalPrepareForShutdown(true)
			framework.ExpectNoError(err)
			gomega.Eventually(func() error {
				isReady := getNodeReadyStatus(f)
				if isReady {
					return fmt.Errorf("node did not become shutdown as expected")
				}
				return nil
			}, nodeStatusUpdateTimeout, pollInterval).Should(gomega.BeNil())

			ginkgo.By("Emitting Shutdown false signal; cancelling the shutdown")
			err = emitSignalPrepareForShutdown(false)
			framework.ExpectNoError(err)
			gomega.Eventually(func() error {
				isReady := getNodeReadyStatus(f)
				if !isReady {
					return fmt.Errorf("node did not recover as expected")
				}
				return nil
			}, nodeStatusUpdateTimeout, pollInterval).Should(gomega.BeNil())
		})

		ginkgo.It("after restart dbus, should be able to gracefully shutdown", func() {
			// allows manual restart of dbus to work in Ubuntu.
			err := overlayDbusConfig()
			framework.ExpectNoError(err)
			defer func() {
				err := restoreDbusConfig()
				framework.ExpectNoError(err)
			}()

			ginkgo.By("Restart Dbus")
			err = restartDbus()
			framework.ExpectNoError(err)

			ginkgo.By("Emitting Shutdown signal")
			err = emitSignalPrepareForShutdown(true)
			framework.ExpectNoError(err)

			gomega.Eventually(func() error {
				isReady := getNodeReadyStatus(f)
				if isReady {
					return fmt.Errorf("node did not become shutdown as expected")
				}
				return nil
			}, nodeStatusUpdateTimeout, pollInterval).Should(gomega.BeNil())
		})
	})
})

func getGracePeriodOverrideTestPod(name string, node string, gracePeriod int64, critical bool) *v1.Pod {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    name,
					Image:   busyboxImage,
					Command: []string{"sh", "-c"},
					Args: []string{`
_term() {
	echo "Caught SIGTERM signal!"
	while true; do sleep 5; done
}
trap _term SIGTERM
while true; do sleep 5; done
`},
				},
			},
			TerminationGracePeriodSeconds: &gracePeriod,
			NodeName:                      node,
		},
	}
	if critical {
		pod.ObjectMeta.Annotations = map[string]string{
			kubelettypes.ConfigSourceAnnotationKey: kubelettypes.FileSource,
		}
		pod.Spec.PriorityClassName = scheduling.SystemNodeCritical

		framework.ExpectEqual(kubelettypes.IsCriticalPod(pod), true, "pod should be a critical pod")
	} else {
		framework.ExpectEqual(kubelettypes.IsCriticalPod(pod), false, "pod should not be a critical pod")
	}
	return pod
}

// Emits a fake PrepareForShutdown dbus message on system dbus. Will cause kubelet to react to an active shutdown event.
func emitSignalPrepareForShutdown(b bool) error {
	cmd := "dbus-send --system /org/freedesktop/login1 org.freedesktop.login1.Manager.PrepareForShutdown boolean:" + strconv.FormatBool(b)
	_, err := runCommand("sh", "-c", cmd)
	return err
}

func getNodeReadyStatus(f *framework.Framework) bool {
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
	framework.ExpectNoError(err)
	// Assuming that there is only one node, because this is a node e2e test.
	framework.ExpectEqual(len(nodeList.Items), 1)
	return isNodeReady(&nodeList.Items[0])
}

func restartDbus() error {
	cmd := "systemctl restart dbus"
	_, err := runCommand("sh", "-c", cmd)
	return err
}

func systemctlDaemonReload() error {
	cmd := "systemctl daemon-reload"
	_, err := runCommand("sh", "-c", cmd)
	return err
}

var (
	dbusConfPath = "/etc/systemd/system/dbus.service.d/k8s-graceful-node-shutdown-e2e.conf"
	dbusConf     = `
[Unit]
RefuseManualStart=no
RefuseManualStop=no
[Service]
KillMode=control-group
ExecStop=
`
)

func overlayDbusConfig() error {
	err := os.MkdirAll(filepath.Dir(dbusConf), 0755)
	if err != nil {
		return err
	}
	err = os.WriteFile(dbusConfPath, []byte(dbusConf), 0644)
	if err != nil {
		return err
	}
	return systemctlDaemonReload()
}

func restoreDbusConfig() error {
	err := os.Remove(dbusConf)
	if err != nil {
		return err
	}
	return systemctlDaemonReload()
}
