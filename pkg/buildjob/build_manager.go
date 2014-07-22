package buildjob

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
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
	syncHandler   func(jobSpec api.Job) error
	typeDelegates map[string]BuildTypeDelegate
}

type BuildTypeDelegate func(jobSpec api.Job) (*api.Pod, error)

// PodControlInterface is an interface that knows how to add or delete pods
// created as an interface to allow testing.
type PodControlInterface interface {
	// run a pod for the specified job
	runJob(jobSpec api.Job)
	// deletePod deletes the pod identified by podID.
	deletePod(podID string) error
}

// RealPodControl is the default implementation of PodControlInterface.
type RealPodControl struct {
	kubeClient client.Interface
}

func (r RealPodControl) runJob(jobSpec api.Job) {
	createPodConfig := typeDelegates[jobSpec.Type]
	if createPodConfig == nil {
		jobSpec.State = api.JobComplete
		jobSpec.Success = false
		// TODO: handle error
		// kubeClient.UpdateJob(jobSpec)
		return
	}

	pod, err := createPodConfig(jobSpec)
	if err != nil {
		jobSpec.State = api.JobComplete
		jobSpec.Success = false
		// TODO: update state of job
		// kubeClient.UpdateJob(jobSpec)
		return
	}

	_, err = r.kubeClient.CreatePod(*pod)
	if err != nil {
		glog.Errorf("%#v\n", err)
		jobSpec.State = api.JobComplete
		jobSpec.Success = false
		// kubeClient.UpdateJob(jobSpec)
	}

	jobSpec.State = api.JobNew
	// kubeClient.UpdateJob(jobSpec)
}

func (r RealPodControl) deletePod(podID string) error {
	return r.kubeClient.DeletePod(podID)
}

func dockerfileBuildJobFor(jobSpec api.Job) (*api.Pod, error) {
	pod := api.Pod{}
	return nil, nil
}

func stiBuildJobFor(jobSpec api.Job) (*api.Pod, error) {
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

	rm.syncHandler = func(jobSpec api.Job) error {
		return rm.syncJobState(jobSpec)
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
			controller, err := rm.handleWatchResponse(watchResponse)
			if err != nil {
				glog.Errorf("Error handling data: %#v, %#v", err, watchResponse)
				continue
			}
			rm.syncHandler(*controller)
		}
	}
}

func (rm *BuildJobManager) handleWatchResponse(response *etcd.Response) (*api.Job, error) {
	switch response.Action {
	case "set":
		if response.Node == nil {
			return nil, fmt.Errorf("response node is null %#v", response)
		}
		var jobSpec api.Job
		if err := json.Unmarshal([]byte(response.Node.Value), &jobSpec); err != nil {
			return nil, err
		}
		return &jobSpec, nil
	case "delete":
		// TODO: determine if some cleanup should be done here
	}

	return nil, nil
}

// Sync loop implementation, pointed to by rm.syncHandler
func (rm *BuildJobManager) syncJobState(jobSpec api.Job) error {
	jobPod, err := rm.kubeClient.GetPod(jobSpec.PodId)
	if err != nil {
		return err
	}

	var (
		podStatus = jobPod.CurrentState.Status
		podInfo   = jobPod.CurrentState.Info
		jobState  = jobSpec.State
		update    = false
	)

	switch podStatus {
	case api.PodRunning:
		switch jobState {
		case api.JobNew || api.JobPending:
			jobSpec.State = api.JobRunning
			update = true
		case api.JobComplete:
			return errors.New("Illegal state transition")
		}
	case api.PodPending:
		switch jobState {
		case api.JobNew:
			jobSpec.State = api.JobPending
			update = true
		case api.JobComplete:
			return errors.New("Illegal state transition")
		}
	case api.PodStopped:
		if jobState == api.JobComplete {
			return nil
		}

		jobSpec.State = api.JobComplete
		containerState = podInfo[0].State
		update = true

		if containerState.ExitCode == 0 {
			jobSpec.Success = true
		}
	}

	if update {
		kubeClient.UpdateJob(jobSpec)
	}

	return nil
}

func (rm *BuildJobManager) synchronize() {
	var jobSpecs []api.Job
	helper := tools.EtcdHelper{rm.etcdClient}
	err := helper.ExtractList("/jobs/build", &jobSpecs)
	if err != nil {
		glog.Errorf("Synchronization error: %v (%#v)", err, err)
		return
	}
	for _, jobSpec := range jobSpecs {
		err = rm.syncHandler(jobSpec)
		if err != nil {
			glog.Errorf("Error synchronizing: %#v", err)
		}
	}
}
