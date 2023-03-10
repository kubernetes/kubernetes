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

package e2enode

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
	admissionapi "k8s.io/pod-security-admission/api"

	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
)

type containerType byte

const (
	containerTypeInvalid containerType = iota
	containerTypeInit
	containerTypeRegular
)

type containerTestConfig struct {
	Name     string
	Type     containerType
	ExitCode int
	Delay    int
}

func (c containerTestConfig) Command() []string {
	var cmd bytes.Buffer
	// all outputs are in the format of:
	// time-since-boot timestamp container-name message

	// The busybox time command doesn't support sub-second display. uptime displays in hundredths of a second, so we
	// include both and use time since boot for relative ordering
	timeCmd := "`date +%s` `cat /proc/uptime | awk '{print $1}'`"
	fmt.Fprintf(&cmd, "echo %s '%s Starting' >> /dev/termination-log; ", timeCmd, c.Name)
	fmt.Fprintf(&cmd, "echo %s '%s Delaying %d' >> /dev/termination-log; ", timeCmd, c.Name, c.Delay)
	if c.Delay != 0 {
		fmt.Fprintf(&cmd, "sleep %d; ", c.Delay)
	}
	fmt.Fprintf(&cmd, "echo %s '%s Exiting' >> /dev/termination-log; ", timeCmd, c.Name)
	fmt.Fprintf(&cmd, "exit %d", c.ExitCode)
	return []string{"sh", "-c", cmd.String()}
}

var _ = SIGDescribe("InitContainers [NodeConformance]", func() {
	f := framework.NewDefaultFramework("initcontainers-test")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	ginkgo.It("should launch init container serially before a regular container", func() {
		init1 := containerTestConfig{
			Name:     "init-1",
			Type:     containerTypeInit,
			Delay:    1,
			ExitCode: 0,
		}
		init2 := containerTestConfig{
			Name:     "init-2",
			Type:     containerTypeInit,
			Delay:    1,
			ExitCode: 0,
		}
		init3 := containerTestConfig{
			Name:     "init-3",
			Type:     containerTypeInit,
			Delay:    1,
			ExitCode: 0,
		}
		regular1 := containerTestConfig{
			Name:     "regular-1",
			Type:     containerTypeRegular,
			Delay:    1,
			ExitCode: 0,
		}

		/// generates an out file output like:
		//
		// 1678337827 45930.43 init-1 Starting
		// 1678337827 45930.43 init-1 Delaying 1
		// 1678337828 45931.43 init-1 Exiting
		// 1678337829 45932.52 init-2 Starting
		// 1678337829 45932.53 init-2 Delaying 1
		// 1678337830 45933.53 init-2 Exiting
		// 1678337831 45934.47 init-3 Starting
		// 1678337831 45934.47 init-3 Delaying 1
		// 1678337832 45935.47 init-3 Exiting
		// 1678337833 45936.58 regular-1 Starting
		// 1678337833 45936.58 regular-1 Delaying 1
		// 1678337834 45937.58 regular-1 Exiting

		podSpec := getContainerOrderingPod("initcontainer-test-pod",
			init1, init2, init3, regular1)

		client := e2epod.NewPodClient(f)
		podSpec = client.Create(context.TODO(), podSpec)
		ginkgo.By("Waiting for the pod to finish")
		err := e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(context.TODO(), f.ClientSet, podSpec.Name, podSpec.Namespace, 1*time.Minute)
		framework.ExpectNoError(err)

		ginkgo.By("Parsing results")
		podSpec, err = client.Get(context.Background(), podSpec.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		results := parseOutput(podSpec)

		// which we then use to make assertions regarding container ordering
		ginkgo.By("Analyzing results")
		framework.ExpectNoError(results.StartsBefore(init1, init2))
		framework.ExpectNoError(results.ExitsBefore(init1, init2))

		framework.ExpectNoError(results.StartsBefore(init2, init3))
		framework.ExpectNoError(results.ExitsBefore(init2, init3))

		framework.ExpectNoError(results.StartsBefore(init3, regular1))
		framework.ExpectNoError(results.ExitsBefore(init3, regular1))
	})

	ginkgo.It("should not launch regular containers if an init container fails", func() {
		init1 := containerTestConfig{
			Name:     "init-1",
			Type:     containerTypeInit,
			Delay:    1,
			ExitCode: 1,
		}
		regular1 := containerTestConfig{
			Name:     "regular-1",
			Type:     containerTypeRegular,
			Delay:    1,
			ExitCode: 0,
		}

		podSpec := getContainerOrderingPod("initcontainer-test-pod-failure",
			init1, regular1)

		client := e2epod.NewPodClient(f)
		podSpec = client.Create(context.TODO(), podSpec)
		ginkgo.By("Waiting for the pod to fail")
		err := e2epod.WaitForPodFailedReason(context.TODO(), f.ClientSet, podSpec, "", 1*time.Minute)
		framework.ExpectNoError(err)

		ginkgo.By("Parsing results")
		podSpec, err = client.Get(context.Background(), podSpec.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		results := parseOutput(podSpec)

		ginkgo.By("Analyzing results")
		// init container should start and exit with an error, and the regular container should never start
		framework.ExpectNoError(results.Starts(init1))
		framework.ExpectNoError(results.Exits(init1))
		framework.ExpectNoError(results.DoesntStart(regular1))
	})

})

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

// StartsBefore returns an error if lhs did not start before rhs
func (o containerOutputList) StartsBefore(lhs, rhs containerTestConfig) error {
	lhsStart := o.findIndex(lhs.Name, "Starting")
	rhsStart := o.findIndex(rhs.Name, "Starting")

	if lhsStart == -1 {
		return fmt.Errorf("couldn't find that container %s ever started, got %s", lhs.Name, o)
	}
	if rhsStart == -1 {
		return fmt.Errorf("couldn't find that container %s ever started, got %s", rhs.Name, o)
	}
	if lhsStart >= rhsStart {
		return fmt.Errorf("expected container %s to start before %s, got %s", lhs.Name, rhs.Name, o)
	}
	return nil
}

// ExitsBefore returns an error if lhs did not end before rhs
func (o containerOutputList) ExitsBefore(lhs, rhs containerTestConfig) error {
	lhsExit := o.findIndex(lhs.Name, "Exiting")
	rhsExit := o.findIndex(rhs.Name, "Exiting")

	if lhsExit == -1 {
		return fmt.Errorf("couldn't find that container %s ever exited, got %s", lhs.Name, o)
	}
	if rhsExit == -1 {
		return fmt.Errorf("couldn't find that container %s ever exited, got %s", rhs.Name, o)
	}
	if lhsExit >= rhsExit {
		return fmt.Errorf("expected container %s to exit before %s, got %s", lhs.Name, rhs.Name, o)
	}
	return nil
}

// Starts returns an error if the container was not found to have started
func (o containerOutputList) Starts(c containerTestConfig) error {
	if idx := o.findIndex(c.Name, "Starting"); idx == -1 {
		return fmt.Errorf("couldn't find that container %s ever started, got %s", c.Name, o)
	}
	return nil
}

// DoesntStart returns an error if the container was found to have started
func (o containerOutputList) DoesntStart(c containerTestConfig) error {
	if idx := o.findIndex(c.Name, "Starting"); idx != -1 {
		return fmt.Errorf("find container %s started, but didn't expect to, got %s", c.Name, o)
	}
	return nil
}

// Exits returns an error if the container was not found to have exited
func (o containerOutputList) Exits(c containerTestConfig) error {
	if idx := o.findIndex(c.Name, "Exiting"); idx == -1 {
		return fmt.Errorf("couldn't find that container %s ever exited, got %s", c.Name, o)
	}
	return nil
}

func (o containerOutputList) findIndex(name string, command string) int {
	for i, v := range o {
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

func getContainerOrderingPod(podName string, containerConfigs ...containerTestConfig) *v1.Pod {
	p := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
		},
	}

	for _, cc := range containerConfigs {
		if cc.Name == "" {
			framework.Failf("expected container config to have a name, found %#v", cc)
		}
		c := v1.Container{
			Name:    cc.Name,
			Image:   busyboxImage,
			Command: cc.Command(),
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("15Mi"),
				},
				Limits: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("15Mi"),
				},
			},
		}

		switch cc.Type {
		case containerTypeInit:
			p.Spec.InitContainers = append(p.Spec.InitContainers, c)
		case containerTypeRegular:
			p.Spec.Containers = append(p.Spec.Containers, c)
		default:
			framework.Failf("expected container config to have a valid type, found %#v", cc)
		}
	}

	return p
}
