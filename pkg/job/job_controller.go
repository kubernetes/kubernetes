package job

import (
	"errors"
	"fmt"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

type JobController struct {
	kubeClient client.Interface
	jobRunner  JobRunner
	syncTime   <-chan time.Time
}

// JobRunner is an interface that knows how to run jobs and and was
// created as an interface to allow testing.
type JobRunner interface {
	// run a pod for the specified job
	run(job api.Job) error
}

// DefaultJobRunner is the default implementation of JobRunner interface.
type DefaultJobRunner struct {
	kubeClient client.Interface
}

func MakeJobController(kubeClient client.Interface) *JobController {
	jc := &JobController{
		kubeClient: kubeClient,
		jobRunner: &DefaultJobRunner{
			kubeClient: kubeClient,
		},
	}

	return jc
}

// Run begins watching and syncing.
func (jc *JobController) Run(period time.Duration) {
	jc.syncTime = time.Tick(period)
	go util.Forever(func() { jc.synchronize() }, period)
}

func (jc *JobController) synchronize() {
	jobs, err := jc.kubeClient.ListJobs()
	if err != nil {
		glog.Errorf("Synchronization error: %v (%#v)", err, err)
		return
	}

	// todo(danmace): async in the future?
	for _, job := range jobs.Items {
		err = jc.syncJobState(job)
		if err != nil {
			glog.Errorf("Error synchronizing: %#v (%#v)", err, err)
		}
	}
}

// Sync loop implementation, pointed to by rm.syncHandler
// TODO: improve handling of illegal state transitions
func (jc *JobController) syncJobState(job api.Job) error {
	glog.Infof("Syncing job state for job ID %s", job.ID)
	if job.Status == api.JobNew {
		return jc.jobRunner.run(job)
	} else if job.Status == api.JobPending && len(job.PodID) == 0 {
		return nil
	}

	glog.Infof("Retrieving info for pod ID %s for job ID %s", job.PodID, job.ID)
	jobPod, err := jc.kubeClient.GetPod(job.PodID)
	if err != nil {
		return err
	}

	var (
		podStatus = jobPod.CurrentState.Status
		podInfo   = jobPod.CurrentState.Info
		jobState  = job.Status
		update    = false
	)

	glog.Infof("Status for pod ID %s: %s", job.PodID, podStatus)
	switch podStatus {
	case api.PodRunning:
		switch jobState {
		case api.JobNew, api.JobPending:
			glog.Infof("Setting job state to running")
			job.Status = api.JobRunning
			update = true
		case api.JobComplete:
			return errors.New("Illegal state transition")
		}
	case api.PodPending:
		switch jobState {
		case api.JobNew:
			job.Status = api.JobPending
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
		job.Status = api.JobComplete

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
		_, err := jc.kubeClient.UpdateJob(job)
		if err != nil {
			return err
		}
	}

	return nil
}

func (r *DefaultJobRunner) run(job api.Job) error {
	glog.Infof("Attempting to run pod for job %v: %#v", job.ID, job.Pod)
	pod, err := r.kubeClient.CreatePod(job.Pod)

	if err != nil {
		glog.Errorf("Error creating pod for job ID %v: %#v\n", job.ID, err)

		job.Status = api.JobComplete
		job.Success = false

		_, err := r.kubeClient.UpdateJob(job)
		if err != nil {
			return errors.New(fmt.Sprintf("Couldn't update Job: %#v : %s", job, err.Error()))
		}

		return nil
	}

	glog.Infof("Setting job state to pending, pod id = %s", pod.ID)
	job.Status = api.JobPending
	job.PodID = pod.ID

	_, err = r.kubeClient.UpdateJob(job)
	if err != nil {
		return errors.New(fmt.Sprintf("Couldn't update Job: %#v : %s", job, err.Error()))
	}

	return nil
}
