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

package lifecyclehooks

import (
    "context"
    "fmt"

    v1 "k8s.io/api/core/v1"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/kubernetes/test/e2e/framework"
    "k8s.io/kubernetes/test/e2e/framework/ssh"
    e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
    "k8s.io/kubernetes/test/e2e_node/testdoc"
    admissionapi "k8s.io/pod-security-admission/api"
    imageutils "k8s.io/kubernetes/test/utils/image"
    "github.com/onsi/ginkgo/v2"
    "github.com/onsi/gomega"
    "k8s.io/utils/ptr"
)

var _ =  SIGDescribe(framework.WithNodeConformance(),"PreStop Hook Test Suite", func() {
    f := framework.NewDefaultFramework("prestop-hook-test-suite")
    f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

    ginkgo.It("should run prestop basic with the grace period", func(ctx context.Context) {
        testdoc.TestName("Hooks:prestop_basic_execution_test")

	testdoc.TestStep("When you have a PreStop hook defined on your container, it will execute before the container is terminated.")
        const gracePeriod int64 = 30
        client := e2epod.NewPodClient(f)
        const volumeName = "host-data"
        const hostPath = "/tmp/prestop-hook-test.log"
        ginkgo.By("creating a pod with a termination grace period and a long-running PreStop hook")
        pod := &v1.Pod{
            ObjectMeta: metav1.ObjectMeta{
                Name: "pod-termination-grace-period",
            },
            Spec: v1.PodSpec{
                Containers: []v1.Container{
                    {
                        Name:  "busybox",
                        Image: imageutils.GetE2EImage(imageutils.BusyBox),
                        Command: []string{
                            "sleep",
                            "10000",
                        },
                    },
                },
                TerminationGracePeriodSeconds: ptr.To(gracePeriod),
            },
        }

        // Add the PreStop hook to write a message to the hostPath
        pod.Spec.Containers[0].Lifecycle = &v1.Lifecycle{
            PreStop: &v1.LifecycleHandler{
                Exec: &v1.ExecAction{
                    Command: []string{"sh", "-c", fmt.Sprintf("echo 'PreStop Hook Executed' > %s", hostPath)},
                },
            },
        }

        // Define the hostPath volume
        pod.Spec.Volumes = append(pod.Spec.Volumes, v1.Volume{
            Name: volumeName,
            VolumeSource: v1.VolumeSource{
                HostPath: &v1.HostPathVolumeSource{
                    Path: hostPath,
                    Type: ptr.To(v1.HostPathFileOrCreate),
                },
            },
        })

        // Mount the hostPath volume to the container
        pod.Spec.Containers[0].VolumeMounts = []v1.VolumeMount{
            {Name: volumeName, MountPath: hostPath},
        }
        testdoc.TestStep("Imagine You Have a Pod with the Following Specification:")
        testdoc.PodSpec(pod)
        testdoc.TestStep("This Pod should start successfully and run for some time")
        createdPod := client.CreateSync(ctx, pod)

        testdoc.TestStep("When the container is terminated, the PreStop hook will be triggered within the grace period")

        // Delete pod to trigger PreStop hook
		client.DeleteSync(ctx, createdPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

        // Verify PreStop hook output on the host via SSH
        node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, createdPod.Spec.NodeName, metav1.GetOptions{})
        framework.ExpectNoError(err, "Failed to get node details")

        cmd := fmt.Sprintf("cat %s", hostPath)
        result, err := ssh.IssueSSHCommandWithResult(ctx, cmd, framework.TestContext.Provider, node)
        framework.ExpectNoError(err, "SSH command failed")
        gomega.Expect(result).ToNot(gomega.BeNil(), "SSH command returned nil result")
        gomega.Expect(result.Code).To(gomega.Equal(0), "SSH command failed with error code")

        testdoc.TestStep("The following log output confirms the successful execution of the PreStop hook:")
	    testdoc.TestLog(fmt.Sprintf("%s", result.Stdout))
		gomega.Expect(result.Stdout).To(gomega.Equal("PreStop Hook Executed\n"))

        testdoc.TestStep("Once the PreStop hook is successfully executed, the container terminates successfully.")
    })
})
