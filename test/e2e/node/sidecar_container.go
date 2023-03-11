/*
Copyright 2023 The Kubernetes Authors.

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
	"bufio"
	"bytes"
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("[Feature:SidecarContainers] Containers Lifecycle ", func() {
	f := framework.NewDefaultFramework("containers-lifecycle-test")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	ginkgo.It("should launch init container and sidecar containers serially before a regular container", func() {

		init1 := "init-1"
		sidecar1 := "sidecar-1"
		init2 := "init-2"
		sidecar2 := "sidecar-2"
		init3 := "init-3"
		regular1 := "regular-1"

		alwaysPolicy := v1.ContainerRestartPolicyAlways

		podSpec := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "sidecar-containers-start-serially",
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyNever,
				InitContainers: []v1.Container{
					{
						Name:  init1,
						Image: framework.BusyBoxImage,
						Command: ExecCommand(init1, execCommand{
							Delay:    1,
							ExitCode: 0,
						}),
					},
					{
						Name:  sidecar1,
						Image: framework.BusyBoxImage,
						Command: ExecCommand(sidecar1, execCommand{
							Delay:    600, // replacing to 1 will make test fail
							ExitCode: 0,
						}),
						RestartPolicy: &alwaysPolicy,
					},
					{
						Name:  init2,
						Image: framework.BusyBoxImage,
						Command: ExecCommand(init2, execCommand{
							Delay:    1,
							ExitCode: 0,
						}),
					},
					{
						Name:  sidecar2,
						Image: framework.BusyBoxImage,
						Command: ExecCommand(sidecar2, execCommand{
							Delay:    600,
							ExitCode: 0,
						}),
						RestartPolicy: &alwaysPolicy,
					},
					{
						Name:  init3,
						Image: framework.BusyBoxImage,
						Command: ExecCommand(init3, execCommand{
							Delay:    1,
							ExitCode: 0,
						}),
					},
				},
				Containers: []v1.Container{
					{
						Name:  regular1,
						Image: framework.BusyBoxImage,
						Command: ExecCommand(regular1, execCommand{
							Delay:    1,
							ExitCode: 0,
						}),
					},
				},
			},
		}

		preparePodForSidecarContainers(podSpec)

		client := e2epod.NewPodClient(f)
		podSpec = client.Create(context.TODO(), podSpec)
		ginkgo.By("Waiting for the pod to finish")
		err := e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(context.TODO(), f.ClientSet, podSpec.Name, podSpec.Namespace, 5*time.Minute)
		framework.ExpectNoError(err)

		ginkgo.By("Parsing results")
		podSpec, err = client.Get(context.Background(), podSpec.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		results := parseOutput(podSpec)

		// which we then use to make assertions regarding container ordering
		ginkgo.By("Analyzing results")
		framework.ExpectNoError(results.StartsBefore(init1, sidecar1))
		framework.ExpectNoError(results.ExitsBefore(init1, sidecar1))

		framework.ExpectNoError(results.StartsBefore(sidecar1, init2))
		framework.ExpectNoError(results.RunTogether(sidecar1, init2))

		framework.ExpectNoError(results.StartsBefore(init2, sidecar2))
		framework.ExpectNoError(results.ExitsBefore(init2, sidecar2))

		framework.ExpectNoError(results.StartsBefore(sidecar2, init3))
		framework.ExpectNoError(results.RunTogether(sidecar2, sidecar1))
		framework.ExpectNoError(results.RunTogether(sidecar1, init3))
		framework.ExpectNoError(results.RunTogether(sidecar2, init3))

		framework.ExpectNoError(results.StartsBefore(init3, regular1))
		framework.ExpectNoError(results.ExitsBefore(init3, regular1))
		framework.ExpectNoError(results.RunTogether(sidecar1, regular1))
		framework.ExpectNoError(results.RunTogether(sidecar2, regular1))
	})
})

type execCommand struct {
	ExitCode int
	Delay    int
}

func ExecCommand(name string, c execCommand) []string {
	var cmd bytes.Buffer
	// all outputs are in the format of:
	// time-since-boot timestamp container-name message

	// The busybox time command doesn't support sub-second display. uptime displays in hundredths of a second, so we
	// include both and use time since boot for relative ordering
	timeCmd := "`date +%s` `cat /proc/uptime | awk '{print $1}'`"
	containerLog := fmt.Sprintf("/persistent/%s.log", name)

	fmt.Fprintf(&cmd, "touch %s; ", containerLog)
	fmt.Fprintf(&cmd, "cat %s >> /dev/termination-log; ", containerLog)

	fmt.Fprintf(&cmd, "echo %s '%s Starting' | tee -a %s >> /dev/termination-log; ", timeCmd, name, containerLog)
	fmt.Fprintf(&cmd, "echo %s '%s Delaying %d' | tee -a %s >> /dev/termination-log; ", timeCmd, name, c.Delay, containerLog)
	if c.Delay != 0 {
		fmt.Fprintf(&cmd, "sleep %d; ", c.Delay)
	}
	fmt.Fprintf(&cmd, "echo %s '%s Exiting'  | tee -a %s >> /dev/termination-log; ", timeCmd, name, containerLog)
	fmt.Fprintf(&cmd, "exit %d", c.ExitCode)
	return []string{"sh", "-c", cmd.String()}
}

// WaitForPodContainerRestartCount waits for the given Pod container to achieve at least a given restartCount
func WaitForPodContainerRestartCount(ctx context.Context, c clientset.Interface, namespace, podName string, containerIndex int, desiredRestartCount int32, timeout time.Duration) error {
	conditionDesc := fmt.Sprintf("container %d started", containerIndex)
	return e2epod.WaitForPodCondition(ctx, c, namespace, podName, conditionDesc, timeout, func(pod *v1.Pod) (bool, error) {
		if containerIndex > len(pod.Status.ContainerStatuses)-1 {
			return false, nil
		}
		containerStatus := pod.Status.ContainerStatuses[containerIndex]
		return containerStatus.RestartCount >= desiredRestartCount, nil
	})
}

const (
	PostStartPrefix = "PostStart"
)

func prefixedName(namePrefix string, name string) string {
	return fmt.Sprintf("%s-%s", namePrefix, name)
}

type containerOutput struct {
	// time the message was seen to the nearest second
	timestamp time.Time
	// time the message was seen since the host booted, to the nearest hundredth of a second
	timeSinceBoot float64
	containerName string
	command       string
}
type containerOutputList []containerOutput

func (o containerOutputList) String() string {
	var b bytes.Buffer
	for _, v := range o {
		fmt.Fprintf(&b, "%s %f %s %s\n", v.timestamp, v.timeSinceBoot, v.containerName, v.command)
	}
	return b.String()
}

// RunTogether returns an error the lhs and rhs run together
func (o containerOutputList) RunTogether(lhs, rhs string) error {
	lhsStart := o.findIndex(lhs, "Starting", 0)
	rhsStart := o.findIndex(rhs, "Starting", 0)

	lhsFinish := o.findIndex(lhs, "Finishing", 0)
	rhsFinish := o.findIndex(rhs, "Finishing", 0)

	if lhsStart == -1 {
		return fmt.Errorf("couldn't find that %s ever started, got %v", lhs, o)
	}
	if rhsStart == -1 {
		return fmt.Errorf("couldn't find that %s ever started, got %v", rhs, o)
	}

	if lhsFinish != -1 && rhsStart > lhsFinish {
		return fmt.Errorf("expected %s to start before finishing %s, got %v", rhs, lhs, o)
	}

	if rhsFinish != -1 && lhsStart > rhsFinish {
		return fmt.Errorf("expected %s to start before finishing %s, got %v", lhs, rhs, o)
	}

	return nil
}

// StartsBefore returns an error if lhs did not start before rhs
func (o containerOutputList) StartsBefore(lhs, rhs string) error {
	lhsStart := o.findIndex(lhs, "Starting", 0)

	if lhsStart == -1 {
		return fmt.Errorf("couldn't find that %s ever started, got %v", lhs, o)
	}

	// this works even for the same names (restart case)
	rhsStart := o.findIndex(rhs, "Starting", lhsStart+1)

	if rhsStart == -1 {
		return fmt.Errorf("couldn't find that %s started after %s, got %v", rhs, lhs, o)
	}
	return nil
}

// ExitsBefore returns an error if lhs did not end before rhs
func (o containerOutputList) ExitsBefore(lhs, rhs string) error {
	lhsExit := o.findIndex(lhs, "Exiting", 0)

	if lhsExit == -1 {
		return fmt.Errorf("couldn't find that %s ever exited, got %v", lhs, o)
	}

	// this works even for the same names (restart case)
	rhsExit := o.findIndex(rhs, "Starting", lhsExit+1)

	if rhsExit == -1 {
		return fmt.Errorf("couldn't find that %s starting before %s exited, got %v", rhs, lhs, o)
	}
	return nil
}

// Starts returns an error if the container was not found to have started
func (o containerOutputList) Starts(name string) error {
	if idx := o.findIndex(name, "Starting", 0); idx == -1 {
		return fmt.Errorf("couldn't find that %s ever started, got %v", name, o)
	}
	return nil
}

// DoesntStart returns an error if the container was found to have started
func (o containerOutputList) DoesntStart(name string) error {
	if idx := o.findIndex(name, "Starting", 0); idx != -1 {
		return fmt.Errorf("find %s started, but didn't expect to, got %v", name, o)
	}
	return nil
}

// Exits returns an error if the container was not found to have exited
func (o containerOutputList) Exits(name string) error {
	if idx := o.findIndex(name, "Exiting", 0); idx == -1 {
		return fmt.Errorf("couldn't find that %s ever exited, got %v", name, o)
	}
	return nil
}

func (o containerOutputList) findIndex(name string, command string, startIdx int) int {
	for i, v := range o {
		if i < startIdx {
			continue
		}
		if v.containerName == name && v.command == command {
			return i
		}
	}
	return -1
}

// parseOutput combines the termination log from all of the init and regular containers and parses/sorts the outputs to
// produce an execution log
func parseOutput(pod *v1.Pod) containerOutputList {
	// accumulate all of our statuses
	var statuses []v1.ContainerStatus
	statuses = append(statuses, pod.Status.InitContainerStatuses...)
	statuses = append(statuses, pod.Status.ContainerStatuses...)
	var buf bytes.Buffer
	for _, cs := range statuses {
		if cs.State.Terminated != nil {
			buf.WriteString(cs.State.Terminated.Message)
		} else if cs.LastTerminationState.Terminated != nil {
			buf.WriteString(cs.LastTerminationState.Terminated.Message)
		}
	}

	// parse
	sc := bufio.NewScanner(&buf)
	var res containerOutputList
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) < 4 {
			framework.ExpectNoError(fmt.Errorf("%v should have at least length 3", fields))
		}
		timestamp, err := strconv.ParseInt(fields[0], 10, 64)
		framework.ExpectNoError(err)
		timeSinceBoot, err := strconv.ParseFloat(fields[1], 64)
		framework.ExpectNoError(err)
		res = append(res, containerOutput{
			timestamp:     time.Unix(timestamp, 0),
			timeSinceBoot: timeSinceBoot,
			containerName: fields[2],
			command:       fields[3],
		})
	}

	// sort using the timeSinceBoot since it has more precision
	sort.Slice(res, func(i, j int) bool {
		return res[i].timeSinceBoot < res[j].timeSinceBoot
	})
	return res
}

func preparePodForSidecarContainers(pod *v1.Pod) {
	var defaultResourceRequirements v1.ResourceRequirements = v1.ResourceRequirements{
		Requests: v1.ResourceList{
			v1.ResourceMemory: resource.MustParse("15Mi"),
		},
		Limits: v1.ResourceList{
			v1.ResourceMemory: resource.MustParse("15Mi"),
		},
	}

	for i := range pod.Spec.Containers {
		c := &pod.Spec.Containers[i]
		c.Resources = defaultResourceRequirements
		c.VolumeMounts = []v1.VolumeMount{
			{
				Name:      "persistent",
				MountPath: "/persistent",
			},
		}
	}
	for i := range pod.Spec.InitContainers {
		c := &pod.Spec.InitContainers[i]
		c.Resources = defaultResourceRequirements
		c.VolumeMounts = []v1.VolumeMount{
			{
				Name:      "persistent",
				MountPath: "/persistent",
			},
		}
	}

	pod.Spec.Volumes = []v1.Volume{
		{
			Name: "persistent",
			VolumeSource: v1.VolumeSource{
				EmptyDir: &v1.EmptyDirVolumeSource{},
			},
		},
	}
}
