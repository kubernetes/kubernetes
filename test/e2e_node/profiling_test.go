/*
Copyright 2018 The Kubernetes Authors.

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

package e2e_node

import (
	"path/filepath"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

type Workload struct {
	// The name of the workload. The profiling result filenames will match the workload name.
	name string
	// The docker image of the workload.
	image string
	// The commands executed when launching the workload. Leave it nil to use the default commands of the Docker image.
	cmd []string
}

func (w Workload) Name() string {
	return w.name
}

func (w Workload) Image() string {
	return w.image
}

func (w Workload) Cmd() []string {
	return w.cmd
}

// add desired workload here
var workloads = []Workload{
	// workload name cannot have duplicate, otherwise result file will be overridden
	{
		name:  "dd1024",
		image: imageutils.GetE2EImage(imageutils.ProfilingToolbox),
		cmd: []string{
			"sh",
			"-c",
			"dd if=/dev/zero of=/dev/zero bs=16M count=1024",
		},
	},
	{
		name:  "dd4096",
		image: imageutils.GetE2EImage(imageutils.ProfilingToolbox),
		cmd: []string{
			"sh",
			"-c",
			"dd if=/dev/zero of=/dev/zero bs=16M count=4096",
		},
	},
}

var (
	f                *framework.Framework
	profilingToolbox *framework.ProfilingToolbox
	workloadPod      *v1.Pod
)

// This test runs a series of workloads in pods and gathers cpu event profiling data.
// To run it:
// make test-e2e-node REMOTE=true FOCUS=WorkloadProfilingTest SKIP=
var _ = framework.KubeDescribe("WorkloadProfilingTest [Slow] [Serial]", func() {
	f = framework.NewDefaultFramework("profiling-test")

	BeforeEach(func() {
		// currently profiling test only works with Docker runtime
		framework.RunIfContainerRuntimeIs("docker")
		// currently profiling test only works with COS image
		framework.SkipIfClusterNotRunningCOS(f)
	})

	for _, workload := range workloads {
		// we need a function wrapper to pass workload so that workload arg won't be overridden by Ginkgo running in parallel
		runIt(workload)
	}
})

func runIt(workload Workload) {
	It("starts a workload pod and perf records the data", func() {
		By("creating profiling toolbox")
		profilingToolbox = f.ProfilingToolbox()
		profilingToolbox.StartProfilingPod()

		By("creating and run workload pod: " + workload.Name())
		workloadPod = makeWorkloadPod(workload.Name(), workload.Image(), workload.Cmd())
		workloadPod = f.PodClient().CreateSync(workloadPod)
		err := f.WaitForPodRunning(workloadPod.Name)
		Expect(err).NotTo(HaveOccurred(), "failed to get workload pod running")

		By("starting perf record command on workload cgroup in profiling pod")
		profilingToolbox.StartPerf(workloadPod)

		// wait for workload pod exiting
		f.WaitForPodNoLongerRunning(workloadPod.Name)

		// send a SIGINT to perf record
		By("stopping perf command")
		profilingToolbox.StopPerf()

		By("creating and starting a graphing pod")
		profilingToolbox.StartGraphingPod(workloadPod)
		profilingToolbox.DeleteProfilingPod()

		By("generating flame graph by perf.data on graphing pod")
		profilingToolbox.GenerateFlamegraph()

		By("copying the generated flame graph back to the host")
		reportDir := framework.TestContext.ReportDir
		profilingToolbox.CopyFlamegraph(filepath.Join(reportDir, "perf-"+workloadPod.Name+".svg"))

		By("removing the workload and graphing pods")
		f.PodClient().DeleteSync(workloadPod.Name, &metav1.DeleteOptions{}, 3*time.Minute)
		profilingToolbox.DeleteGraphingPod()
	})
}

func makeWorkloadPod(name, image string, cmd []string) *v1.Pod {
	container := v1.Container{
		Name:  name,
		Image: image,
	}
	if cmd != nil {
		container.Command = cmd
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				container,
			},
		},
	}
	return pod
}
