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
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"

	"k8s.io/kubernetes/test/e2e/framework"
)

type execCommand struct {
	// ExitCode is the exit status of the container
	ExitCode int
	// StartDelay is how long the container should delay before starting
	StartDelay int
	// Delay is how long the container should delay before exiting
	Delay int
}

// ExecCommand returns the command to execute in the container that implements execCommand and logs activities to a container
// specific log that persists across container restarts.  The final log is written to /dev/termination-log so it can
// be retrieved by the test harness after the container execution.
func ExecCommand(name string, c execCommand) []string {
	var cmd bytes.Buffer
	// all outputs are in the format of:
	// time-since-boot timestamp container-name message

	// The busybox time command doesn't support sub-second display. uptime displays in hundredths of a second, so we
	// include both and use time since boot for relative ordering of file entries
	timeCmd := "`date +%s` `cat /proc/uptime | awk '{print $1}'`"
	containerLog := fmt.Sprintf("/persistent/%s.log", name)

	fmt.Fprintf(&cmd, "touch %s; ", containerLog)
	fmt.Fprintf(&cmd, "cat %s >> /dev/termination-log; ", containerLog)

	fmt.Fprintf(&cmd, "echo %s '%s Starting %d' | tee -a %s >> /dev/termination-log; ", timeCmd, name, c.StartDelay, containerLog)
	if c.StartDelay != 0 {
		fmt.Fprintf(&cmd, "sleep %d; ", c.StartDelay)
	}
	// You can check started file to see if the container has started
	fmt.Fprintf(&cmd, "touch started; ")
	fmt.Fprintf(&cmd, "echo %s '%s Started' | tee -a %s >> /dev/termination-log; ", timeCmd, name, containerLog)
	fmt.Fprintf(&cmd, "echo %s '%s Delaying %d' | tee -a %s >> /dev/termination-log; ", timeCmd, name, c.Delay, containerLog)
	if c.Delay != 0 {
		fmt.Fprintf(&cmd, "sleep %d; ", c.Delay)
	}
	fmt.Fprintf(&cmd, "echo %s '%s Exiting'  | tee -a %s >> /dev/termination-log; ", timeCmd, name, containerLog)
	fmt.Fprintf(&cmd, "exit %d", c.ExitCode)
	return []string{"sh", "-c", cmd.String()}
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
	lhsStart := o.findIndex(lhs, "Started", 0)
	rhsStart := o.findIndex(rhs, "Started", 0)

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
	lhsStart := o.findIndex(lhs, "Started", 0)

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

// DoesntStartAfter returns an error if lhs started after rhs
func (o containerOutputList) DoesntStartAfter(lhs, rhs string) error {
	rhsStart := o.findIndex(rhs, "Starting", 0)

	if rhsStart == -1 {
		return fmt.Errorf("couldn't find that %s ever started, got %v", rhs, o)
	}

	// this works even for the same names (restart case)
	lhsStart := o.findIndex(lhs, "Started", rhsStart+1)

	if lhsStart != -1 {
		return fmt.Errorf("expected %s to not start after %s, got %v", lhs, rhs, o)
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
	if idx := o.findIndex(name, "Started", 0); idx == -1 {
		return fmt.Errorf("couldn't find that %s ever started, got %v", name, o)
	}
	return nil
}

// DoesntStart returns an error if the container was found to have started
func (o containerOutputList) DoesntStart(name string) error {
	if idx := o.findIndex(name, "Started", 0); idx != -1 {
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

// HasRestarted returns an error if the container was not found to have restarted
func (o containerOutputList) HasRestarted(name string) error {
	idx := o.findIndex(name, "Starting", 0)
	if idx == -1 {
		return fmt.Errorf("couldn't find that %s ever started, got %v", name, o)
	}

	idx = o.findIndex(name, "Starting", idx+1)

	if idx == -1 {
		return fmt.Errorf("couldn't find that %s ever restarted, got %v", name, o)
	}

	return nil
}

// HasNotRestarted returns an error if the container was found to have restarted
func (o containerOutputList) HasNotRestarted(name string) error {
	idx := o.findIndex(name, "Starting", 0)
	if idx == -1 {
		return fmt.Errorf("couldn't find that %s ever started, got %v", name, o)
	}

	idx = o.findIndex(name, "Starting", idx+1)

	if idx != -1 {
		return fmt.Errorf("found that %s restarted but wasn't expected to, got %v", name, o)
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
		// If the container is terminated but the reason is ContainerStatusUnknown,
		// it means that the kubelet has overwritten the termination message. Read
		// the LastTerminationState instead.
		if cs.State.Terminated != nil && cs.State.Terminated.Reason != "ContainerStatusUnknown" {
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

// preparePod adds an empty dir volume and mounts it to each container at /persistent
func preparePod(pod *v1.Pod) {
	var defaultResourceRequirements = v1.ResourceRequirements{
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
