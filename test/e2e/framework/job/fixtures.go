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

package job

import (
	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
	"k8s.io/utils/ptr"
)

// NewTestJob returns a Job which does one of several testing behaviors. notTerminate starts a Job that will run
// effectively forever. fail starts a Job that will fail immediately. succeed starts a Job that will succeed
// immediately. randomlySucceedOrFail starts a Job that will succeed or fail randomly. failOnce fails the Job the
// first time it is run and succeeds subsequently. name is the Name of the Job. RestartPolicy indicates the restart
// policy of the containers in which the Pod is running. Parallelism is the Job's parallelism, and completions is the
// Job's required number of completions.
func NewTestJob(behavior, name string, rPol v1.RestartPolicy, parallelism, completions int32, activeDeadlineSeconds *int64, backoffLimit int32) *batchv1.Job {
	anyNode := ""
	return NewTestJobOnNode(behavior, name, rPol, parallelism, completions, activeDeadlineSeconds, backoffLimit, anyNode)
}

// NewTestJobOnNode is similar to NewTestJob but supports specifying a Node on which the Job's Pods will run.
// Empty nodeName means no node selection constraints.
func NewTestJobOnNode(behavior, name string, rPol v1.RestartPolicy, parallelism, completions int32, activeDeadlineSeconds *int64, backoffLimit int32, nodeName string) *batchv1.Job {
	manualSelector := false
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		TypeMeta: metav1.TypeMeta{
			Kind: "Job",
		},
		Spec: batchv1.JobSpec{
			ActiveDeadlineSeconds: activeDeadlineSeconds,
			Parallelism:           &parallelism,
			Completions:           &completions,
			BackoffLimit:          &backoffLimit,
			ManualSelector:        &manualSelector,
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{JobSelectorKey: name},
				},
				Spec: v1.PodSpec{
					RestartPolicy: rPol,
					Volumes: []v1.Volume{
						{
							Name: "data",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{},
							},
						},
					},
					Containers: []v1.Container{
						{
							Name:    "c",
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{},
							VolumeMounts: []v1.VolumeMount{
								{
									MountPath: "/data",
									Name:      "data",
								},
							},
							SecurityContext: &v1.SecurityContext{},
						},
					},
				},
			},
		},
	}
	if len(nodeName) > 0 {
		job.Spec.Template.Spec.NodeSelector = map[string]string{
			"kubernetes.io/hostname": nodeName,
		}
	}
	switch behavior {
	case "neverTerminate":
		// this job is being used in an upgrade job see test/e2e/upgrades/apps/job.go
		// it should never be optimized, as it always has to restart during an upgrade
		// and continue running
		job.Spec.Template.Spec.Containers[0].Command = []string{"sleep", "1000000"}
		job.Spec.Template.Spec.TerminationGracePeriodSeconds = ptr.To(int64(1))
	case "notTerminate":
		job.Spec.Template.Spec.Containers[0].Image = imageutils.GetPauseImageName()
	case "fail":
		job.Spec.Template.Spec.Containers[0].Command = []string{"/bin/sh", "-c", "exit 1"}
	case "failOddSucceedEven":
		job.Spec.Template.Spec.Containers[0].Command = []string{"sh", "-c"}
		job.Spec.Template.Spec.Containers[0].Args = []string{`
			if [ $(expr ${JOB_COMPLETION_INDEX} % 2) -ne 0 ]; then
				exit 1
			else
				exit 0
			fi
			`,
		}
	case "succeed":
		job.Spec.Template.Spec.Containers[0].Command = []string{"/bin/sh", "-c", "exit 0"}
	case "randomlySucceedOrFail":
		// Bash's $RANDOM generates pseudorandom int in range 0 - 32767.
		// Dividing by 16384 gives roughly 50/50 chance of success.
		job.Spec.Template.Spec.Containers[0].Command = []string{"/bin/sh", "-c", "exit $(( $RANDOM / 16384 ))"}
	case "failOnce":
		// Fail the first the container of the pod is run, and
		// succeed the second time. Checks for file on a data volume.
		// If present, succeed.  If not, create but fail.
		// If RestartPolicy is Never, the nodeName should be set to
		// ensure all job pods run on a single node and the volume
		// will be mounted from a hostPath instead.
		setupHostPathDirectory(job)
		job.Spec.Template.Spec.Containers[0].Command = []string{"/bin/sh", "-c"}
		job.Spec.Template.Spec.Containers[0].Args = []string{`
			if [[ -r /data/foo ]]
			then
				exit 0
			else
				touch /data/foo
				exit 1
			fi
		`}
	case "notTerminateOnce":
		// Do not terminate the 0-indexed pod in the first run and
		// succeed the second time. Fail the non-0-indexed pods until
		// the marker file is created by the 0-indexed pod. The fact that
		// the non-0-indexed pods are succeeded is used to determine that the
		// 0th indexed pod already created the marker file.
		setupHostPathDirectory(job)
		job.Spec.Template.Spec.TerminationGracePeriodSeconds = ptr.To(int64(1))
		job.Spec.Template.Spec.Containers[0].Command = []string{"/bin/sh", "-c"}
		job.Spec.Template.Spec.Containers[0].Args = []string{`
			if [[ -r /data/foo ]]
			then
				exit 0
			elif [[ $JOB_COMPLETION_INDEX -eq 0 ]]
			then
				touch /data/foo
				sleep 1000000
			else
				exit 1
			fi
		`}
	}
	return job
}

// setup host path directory to pass information between pod restarts
func setupHostPathDirectory(job *batchv1.Job) {
	if _, nodeNameSpecified := job.Spec.Template.Spec.NodeSelector["kubernetes.io/hostname"]; nodeNameSpecified {
		randomDir := "/tmp/job-e2e/" + rand.String(10)
		hostPathType := v1.HostPathDirectoryOrCreate
		job.Spec.Template.Spec.Volumes[0].VolumeSource = v1.VolumeSource{HostPath: &v1.HostPathVolumeSource{Path: randomDir, Type: &hostPathType}}
		// Tests involving r/w operations on hostPath volume needs to run in
		// privileged mode for SELinux enabled distro, while Windows platform
		// neither supports nor needs privileged mode.
		privileged := !framework.NodeOSDistroIs("windows")
		job.Spec.Template.Spec.Containers[0].SecurityContext.Privileged = &privileged
	}
}
