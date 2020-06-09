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
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/kubernetes/test/e2e/framework"
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
							Image:   framework.BusyBoxImage,
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
					NodeName: nodeName,
				},
			},
		},
	}
	switch behavior {
	case "notTerminate":
		job.Spec.Template.Spec.Containers[0].Command = []string{"sleep", "1000000"}
	case "fail":
		job.Spec.Template.Spec.Containers[0].Command = []string{"/bin/sh", "-c", "exit 1"}
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
		if len(nodeName) > 0 {
			randomDir := "/tmp/job-e2e/" + rand.String(10)
			hostPathType := v1.HostPathDirectoryOrCreate
			job.Spec.Template.Spec.Volumes[0].VolumeSource = v1.VolumeSource{HostPath: &v1.HostPathVolumeSource{Path: randomDir, Type: &hostPathType}}
			// Tests involving r/w operations on hostPath volume needs to run in
			// privileged mode for SELinux enabled distro, while Windows platform
			// neither supports nor needs privileged mode.
			privileged := !framework.NodeOSDistroIs("windows")
			job.Spec.Template.Spec.Containers[0].SecurityContext.Privileged = &privileged
		}
		job.Spec.Template.Spec.Containers[0].Command = []string{"/bin/sh", "-c", "if [[ -r /data/foo ]] ; then exit 0 ; else touch /data/foo ; exit 1 ; fi"}
	}
	return job
}
