package build

import (
	"errors"
	"fmt"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

type BuildController struct {
	etcdClient tools.EtcdClient
	kubeClient client.Interface
	syncTime   <-chan time.Time
}

type BuildTypeDelegate func(jobSpec api.Job) (*api.Pod, error)

func dockerfileBuildJobFor(job api.Job) (*api.Pod, error) {
	var envVars []api.EnvVar
	for k, v := range job.Context {
		envVars = append(envVars, api.EnvVar{Name: k, Value: v})
	}

	pod := &api.Pod{
		Labels: map[string]string{
			"podType": "job",
			"jobType": "build",
		},
		DesiredState: api.PodState{
			Manifest: api.ContainerManifest{
				Version: "v1beta1",
				Containers: []api.Container{
					{
						Name:          "build-job-" + job.ID,
						Image:         "ironcladlou/openshift-docker-builder",
						Privileged:    true,
						RestartPolicy: "runOnce",
						Env:           envVars,
					},
				},
			},
		},
	}
	return pod, nil
}

func stiBuildJobFor(job api.Job) (*api.Pod, error) {
	return nil, nil
}

func MakeBuildController(etcdClient tools.EtcdClient, kubeClient client.Interface) *BuildController {
	rm := &BuildController{
		kubeClient: kubeClient,
		etcdClient: etcdClient,
		podControl: RealPodControl{
			kubeClient: kubeClient,
			typeDelegates: map[string]BuildTypeDelegate{
				"dockerfile": dockerfileBuildJobFor,
				"sti":        stiBuildJobFor,
			},
		},
	}

	return rm
}

// Run begins watching and syncing.
func (rm *BuildController) Run(period time.Duration) {
	rm.syncTime = time.Tick(period)
	go util.Forever(func() { rm.synchronize() }, period)
}

// Sync loop implementation, pointed to by rm.syncHandler
// TODO: improve handling of illegal state transitions
func (rm *BuildController) syncBuildState(job api.Job) error {
	glog.Infof("Syncing job state for job ID %s", job.ID)
	if job.State == api.JobNew {
		return rm.podControl.runJob(job)
	} else if job.State == api.JobPending && len(job.PodID) == 0 {
		return nil
	}

	glog.Infof("Retrieving info for pod ID %s for job ID %s", job.PodID, job.ID)
	jobPod, err := rm.kubeClient.GetPod(job.PodID)
	if err != nil {
		return err
	}

	var (
		podStatus = jobPod.CurrentState.Status
		podInfo   = jobPod.CurrentState.Info
		jobState  = job.State
		update    = false
	)

	glog.Infof("Status for pod ID %s: %s", job.PodID, podStatus)
	switch podStatus {
	case api.PodRunning:
		switch jobState {
		case api.JobNew, api.JobPending:
			glog.Infof("Setting job state to running")
			job.State = api.JobRunning
			update = true
		case api.JobComplete:
			return errors.New("Illegal state transition")
		}
	case api.PodPending:
		switch jobState {
		case api.JobNew:
			job.State = api.JobPending
			update = true
		case api.JobComplete:
			return errors.New("Illegal state transition")
		}
	case api.PodStopped:
		if jobState == api.JobComplete {
			glog.Infof("Job status is already complete - no-op")
			return nil
		}

		// TODO: better way of evaluating job completion
		glog.Infof("Setting job state to complete")
		job.State = api.JobComplete

		infoKeys := make([]string, len(podInfo))
		i := 0
		for k, _ := range podInfo {
			infoKeys[i] = k
			i++
		}

		containerState := podInfo[infoKeys[0]].State
		update = true

		job.Success = containerState.ExitCode == 0
	}

	if update {
		_, err := rm.kubeClient.UpdateJob(job)
		if err != nil {
			return err
		}
	}

	return nil
}

func (rm *BuildController) synchronize() {
	jobs, err := rm.kubeClient.ListJobs()
	if err != nil {
		glog.Errorf("Synchronization error: %v (%#v)", err, err)
		return
	}
	for _, job := range jobs.Items {
		err = rm.syncBuildState(job)
		if err != nil {
			glog.Errorf("Error synchronizing: %#v", err)
		}
	}
}
