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
	"os"
	"path/filepath"
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
	// timestamp container-name message
	fmt.Fprintf(&cmd, "echo `date +%%s` '%s Starting' >> /shared/output; ", c.Name)
	fmt.Fprintf(&cmd, "echo `date +%%s` '%s Delaying %d' >> /shared/output; ", c.Name, c.Delay)
	if c.Delay != 0 {
		fmt.Fprintf(&cmd, "sleep %d; ", c.Delay)
	}
	fmt.Fprintf(&cmd, "echo `date +%%s` '%s Exiting' >> /shared/output; ", c.Name)
	fmt.Fprintf(&cmd, "exit %d", c.ExitCode)
	return []string{"sh", "-c", cmd.String()}
}

var _ = SIGDescribe("InitContainers [NodeConformance]", func() {
	f := framework.NewDefaultFramework("initcontainers-test")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	var tmpDir string
	ginkgo.BeforeEach(func() {
		var err error
		tmpDir, err = os.MkdirTemp("", "init-container-*")
		framework.ExpectNoError(err, "creating temp directory")

	})
	ginkgo.AfterEach(func() {
		os.RemoveAll(tmpDir)
	})

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
		//  1677116487 init-1 Starting
		//  1677116487 init-1 Delaying 1
		//  1677116488 init-1 Exiting
		//  1677116489 init-2 Starting
		//  1677116489 init-2 Delaying 1
		//  1677116490 init-2 Exiting
		//  1677116491 init-3 Starting
		//  1677116491 init-3 Delaying 1
		//  1677116492 init-3 Exiting
		//  1677116493 regular-1 Starting
		//  1677116493 regular-1 Delaying 1
		//  1677116494 regular-1 Exiting

		podSpec := getContainerOrderingPod("initcontainer-test-pod",
			tmpDir, init1, init2, init3, regular1)

		podSpec = e2epod.NewPodClient(f).Create(context.TODO(), podSpec)
		ginkgo.By("Waiting for the pod to finish")
		err := e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(context.TODO(), f.ClientSet, podSpec.Name, podSpec.Namespace, 1*time.Minute)
		framework.ExpectNoError(err)

		ginkgo.By("Parsing results")
		results := parseOutput(tmpDir)

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
			tmpDir, init1, regular1)

		podSpec = e2epod.NewPodClient(f).Create(context.TODO(), podSpec)
		ginkgo.By("Waiting for the pod to fail")
		err := e2epod.WaitForPodFailedReason(context.TODO(), f.ClientSet, podSpec, "", 1*time.Minute)
		framework.ExpectNoError(err)

		ginkgo.By("Parsing results")
		results := parseOutput(tmpDir)

		ginkgo.By("Analyzing results")
		// init container should start and exit with an error, and the regular container should never start
		framework.ExpectNoError(results.Starts(init1))
		framework.ExpectNoError(results.Exits(init1))
		framework.ExpectNoError(results.DoesntStart(regular1))
	})

})

type containerOutput struct {
	line          int
	timestamp     string
	containerName string
	command       string
}
type containerOutputList []containerOutput

func (o containerOutputList) String() string {
	var b bytes.Buffer
	for _, v := range o {
		fmt.Fprintf(&b, "%d %s %s %s\n", v.line, v.timestamp, v.containerName, v.command)
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
	for _, v := range o {
		if v.containerName == name && v.command == command {
			return v.line
		}
	}
	return -1
}

func parseOutput(dir string) containerOutputList {
	contents, err := os.ReadFile(filepath.Join(dir, "output"))
	framework.ExpectNoError(err, "reading output file")

	sc := bufio.NewScanner(bytes.NewReader(contents))
	var res containerOutputList
	lineNo := 0
	for sc.Scan() {
		lineNo++
		fields := strings.Fields(sc.Text())
		if len(fields) < 3 {
			framework.ExpectNoError(fmt.Errorf("%v should have at least length 3", fields))
		}
		res = append(res, containerOutput{
			line:          lineNo,
			timestamp:     fields[0],
			containerName: fields[1],
			command:       fields[2],
		})
	}
	return res
}

func getContainerOrderingPod(podName string, hostDir string, containerConfigs ...containerTestConfig) *v1.Pod {
	// all the pods share the given host directory
	hostPathDirectory := v1.HostPathDirectory
	p := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Volumes: []v1.Volume{
				{
					Name: "shared",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: hostDir,
							Type: &hostPathDirectory,
						},
					},
				},
			},
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
			VolumeMounts: []v1.VolumeMount{
				{
					Name:      "shared",
					MountPath: "/shared",
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
