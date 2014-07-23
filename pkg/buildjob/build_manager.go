package buildjob

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/coreos/go-etcd/etcd"
	"github.com/golang/glog"
)

type BuildJobManager struct {
	etcdClient tools.EtcdClient
	kubeClient client.Interface
	podControl PodControlInterface
	syncTime   <-chan time.Time
}

// PodControlInterface is an interface that knows how to add or delete pods
// created as an interface to allow testing.
type PodControlInterface interface {
	// run a pod for the specified job
	runJob(job api.Job) error
	// deletePod deletes the pod identified by podID.
	deletePod(podID string) error
}

type BuildTypeDelegate func(jobSpec api.Job) (*api.Pod, error)

// RealPodControl is the default implementation of PodControlInterface.
type RealPodControl struct {
	kubeClient    client.Interface
	typeDelegates map[string]BuildTypeDelegate
}

func (r RealPodControl) runJob(job api.Job) error {
	createPodConfig := r.typeDelegates[job.Type]
	if createPodConfig == nil {
		job.State = api.JobComplete
		job.Success = false

		_, err := r.kubeClient.UpdateJob(job)
		if err != nil {
			return errors.New(fmt.Sprintf("Couldn't update Job: %+v : %s", job, err.Error()))
		}

		return nil
	}

	podSpec, err := createPodConfig(job)
	if err != nil {
		job.State = api.JobComplete
		job.Success = false

		_, err := r.kubeClient.UpdateJob(job)
		if err != nil {
			return errors.New(fmt.Sprintf("Couldn't update Job: %+v : %s", job, err.Error()))
		}

		return nil
	}

	pod, err := r.kubeClient.CreatePod(*podSpec)
	if err != nil {
		glog.Errorf("%#v\n", err)

		job.State = api.JobComplete
		job.Success = false

		_, err := r.kubeClient.UpdateJob(job)
		if err != nil {
			return errors.New(fmt.Sprintf("Couldn't update Job: %+v : %s", job, err.Error()))
		}

		return nil
	}

	job.State = api.JobNew
	job.PodID = pod.ID

	_, err = r.kubeClient.UpdateJob(job)
	if err != nil {
		return errors.New(fmt.Sprintf("Couldn't update Job: %+v : %s", job, err.Error()))
	}

	return nil
}

func (r RealPodControl) deletePod(podID string) error {
	return r.kubeClient.DeletePod(podID)
}

func dockerfileBuildJobFor(job api.Job) (*api.Pod, error) {
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
						Env: []api.EnvVar{
							{Name: "BUILD_TAG", Value: job.Context["BUILD_TAG"]},
							{Name: "DOCKER_CONTEXT_URL", Value: job.Context["DOCKER_CONTEXT_URL"]},
						},
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

func MakeBuildJobManager(etcdClient tools.EtcdClient, kubeClient client.Interface) *BuildJobManager {
	rm := &BuildJobManager{
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
func (rm *BuildJobManager) Run(period time.Duration) {
	rm.syncTime = time.Tick(period)
	go util.Forever(func() { rm.watchBuildJobs() }, period)
}

// watch loop
func (rm *BuildJobManager) watchBuildJobs() {
	watchChannel := make(chan *etcd.Response)
	stop := make(chan bool)
	// Ensure that the call to watch ends.
	defer close(stop)

	go func() {
		defer util.HandleCrash()
		_, err := rm.etcdClient.Watch("/jobs/build", 0, true, watchChannel, stop)
		if err == etcd.ErrWatchStoppedByUser {
			close(watchChannel)
		} else {
			glog.Errorf("etcd.Watch stopped unexpectedly: %v (%#v)", err, err)
		}
	}()

	for {
		select {
		case <-rm.syncTime:
			rm.synchronize()
		case watchResponse, open := <-watchChannel:
			if !open || watchResponse == nil {
				// watchChannel has been closed, or something else went
				// wrong with our etcd watch call. Let the util.Forever()
				// that called us call us again.
				return
			}
			glog.Infof("Got watch: %#v", watchResponse)
			job, err := rm.handleWatchResponse(watchResponse)
			if err != nil {
				glog.Errorf("Error handling data: %#v, %#v", err, watchResponse)
				continue
			}

			err = rm.syncJobState(*job)
			if err != nil {
				glog.Errorf("Error syncing job state: %+v, %s", job, err.Error())
			}
		}
	}
}

func (rm *BuildJobManager) handleWatchResponse(response *etcd.Response) (*api.Job, error) {
	switch response.Action {
	case "set":
		if response.Node == nil {
			return nil, fmt.Errorf("response node is null %#v", response)
		}
		var job api.Job
		if err := json.Unmarshal([]byte(response.Node.Value), &job); err != nil {
			return nil, err
		}
		return &job, nil
	case "delete":
		// TODO: determine if some cleanup should be done here
	}

	return nil, nil
}

// Sync loop implementation, pointed to by rm.syncHandler
// TODO: improve handling of illegal state transitions
func (rm *BuildJobManager) syncJobState(job api.Job) error {
	glog.Infof("Syncing job state for job ID %s", job.ID)
	if job.State == api.JobNew {
		return rm.podControl.runJob(job)
	} else if job.State == api.JobPending && len(job.PodID) == 0 {
		return nil
	}

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

	switch podStatus {
	case api.PodRunning:
		switch jobState {
		case api.JobNew, api.JobPending:
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
			return nil
		}

		// TODO: better way of evaluating job completion
		job.State = api.JobComplete

		infoKeys := make([]string, len(podInfo))
		i := 0
		for k, _ := range podInfo {
			infoKeys[i] = k
			i++
		}

		containerState := podInfo[infoKeys[0]].State
		update = true

		if containerState.ExitCode == 0 {
			job.Success = true
		}
	}

	if update {
		_, err := rm.kubeClient.UpdateJob(job)
		if err != nil {
			return err
		}
	}

	return nil
}

func (rm *BuildJobManager) synchronize() {
	jobs, err := rm.kubeClient.ListJobs()
	if err != nil {
		glog.Errorf("Synchronization error: %v (%#v)", err, err)
		return
	}
	for _, job := range jobs.Items {
		err = rm.syncJobState(job)
		if err != nil {
			glog.Errorf("Error synchronizing: %#v", err)
		}
	}
}
