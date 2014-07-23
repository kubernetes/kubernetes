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
	etcdClient    tools.EtcdClient
	kubeClient    client.Interface
	podControl    PodControlInterface
	syncTime      <-chan time.Time
	syncHandler   func(job api.Job) error
	typeDelegates map[string]BuildTypeDelegate
}

type BuildTypeDelegate func(job api.Job) (*api.Pod, error)

// PodControlInterface is an interface that knows how to add or delete pods
// created as an interface to allow testing.
type PodControlInterface interface {
	// run a pod for the specified job
	runJob(job api.Job)
	// deletePod deletes the pod identified by podID.
	deletePod(podID string) error
}

// RealPodControl is the default implementation of PodControlInterface.
type RealPodControl struct {
	kubeClient client.Interface
}

func (r RealPodControl) runJob(job api.Job) {
	createPodConfig := typeDelegates[job.Type]
	if createPodConfig == nil {
		job.State = api.JobComplete
		job.Success = false
		// TODO: handle error
		// kubeClient.UpdateJob(job)
		return
	}

	pod, err := createPodConfig(job)
	if err != nil {
		job.State = api.JobComplete
		job.Success = false
		// TODO: update state of job
		// kubeClient.UpdateJob(job)
		return
	}

	_, err = r.kubeClient.CreatePod(*pod)
	if err != nil {
		glog.Errorf("%#v\n", err)
		job.State = api.JobComplete
		job.Success = false
		// kubeClient.UpdateJob(job)
	}

	job.State = api.JobNew
	// kubeClient.UpdateJob(job)
}

func (r RealPodControl) deletePod(podID string) error {
	return r.kubeClient.DeletePod(podID)
}

func dockerfileBuildJobFor(job api.Job) (*api.Pod, error) {
	pod := api.Pod{}
	return nil, nil
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
		},
		typeDelegates: map[string]BuildTypeDelegate{
			"dockerfile": dockerfileBuildJobFor,
			"sti":        stiBuildJobFor,
		},
	}

	rm.syncHandler = func(job api.Job) error {
		return rm.syncJobState(job)
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
			rm.syncHandler(*job)
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
func (rm *BuildJobManager) syncJobState(job api.Job) error {
	jobPod, err := rm.kubeClient.GetPod(job.PodId)
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
		case api.JobNew || api.JobPending:
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

		job.State = api.JobComplete
		containerState = podInfo[0].State
		update = true

		if containerState.ExitCode == 0 {
			job.Success = true
		}
	}

	if update {
		kubeClient.UpdateJob(job)
	}

	return nil
}

func (rm *BuildJobManager) synchronize() {
	jobs, err := kubeClient.ListJobs()
	if err != nil {
		glog.Errorf("Synchronization error: %v (%#v)", err, err)
		return
	}
	for _, job := range jobs {
		err = rm.syncHandler(job)
		if err != nil {
			glog.Errorf("Error synchronizing: %#v", err)
		}
	}
}
