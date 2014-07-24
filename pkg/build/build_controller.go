package build

import (
	"errors"
	"fmt"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

type BuildController struct {
	kubeClient  client.Interface
	syncTime    <-chan time.Time
	buildRunner BuildRunner
}

// BuildRunner is an interface that knows how to run builds and was
// created as an interface to allow testing.
type BuildRunner interface {
	// run a job for the specified build
	run(build api.Build) error
}

// DefaultBuildRunner is the default implementation of BuildRunner interface.
type DefaultBuildRunner struct {
	kubeClient client.Interface
}

func MakeBuildController(kubeClient client.Interface) *BuildController {
	bc := &BuildController{
		kubeClient: kubeClient,
		buildRunner: &DefaultBuildRunner{
			kubeClient: kubeClient,
		},
	}

	return bc
}

// Run begins watching and syncing.
func (bc *BuildController) Run(period time.Duration) {
	bc.syncTime = time.Tick(period)
	go util.Forever(func() { bc.synchronize() }, period)
}

// The main sync loop. Iterates over current builds and delegates syncing.
func (bc *BuildController) synchronize() {
	builds, err := bc.kubeClient.ListBuilds()
	if err != nil {
		glog.Errorf("Synchronization error: %v (%#v)", err, err)
		return
	}
	for _, build := range builds.Items {
		err = bc.syncBuildState(build)
		if err != nil {
			glog.Errorf("Error synchronizing: %#v", err)
		}
	}
}

// Sync loop brings the build state into sync with its corresponding job.
// TODO: improve handling of illegal state transitions
func (bc *BuildController) syncBuildState(build api.Build) error {
	glog.Infof("Syncing build state for build ID %s", build.ID)
	glog.Infof("Build details: %#v", build)
	if build.Status == api.BuildNew {
		return bc.buildRunner.run(build)
	} else if build.Status == api.BuildPending && len(build.JobID) == 0 {
		return nil
	}

	glog.Infof("Retrieving info for job ID %s for build ID %s", build.JobID, build.ID)
	job, err := bc.kubeClient.GetJob(build.JobID)
	if err != nil {
		return err
	}

	update := false

	glog.Infof("Status for job ID %s: %s", job.ID, job.Status)
	switch job.Status {
	case api.JobRunning:
		switch build.Status {
		case api.BuildNew, api.BuildPending:
			glog.Infof("Setting build state to running")
			build.Status = api.BuildRunning
			update = true
		case api.BuildComplete:
			return errors.New("Illegal state transition")
		}
	case api.JobPending:
		switch build.Status {
		case api.BuildNew:
			build.Status = api.BuildPending
			update = true
		case api.BuildComplete:
			return errors.New("Illegal state transition")
		}
	case api.JobComplete:
		if build.Status == api.BuildComplete {
			glog.Infof("Build status is already complete - no-op")
			return nil
		}

		// TODO: better way of evaluating job completion
		glog.Infof("Setting build state to complete")
		build.Status = api.BuildComplete
		build.Success = job.Success
		update = true
	}

	if update {
		_, err := bc.kubeClient.UpdateBuild(build)
		if err != nil {
			return err
		}
	}

	return nil
}

func (r *DefaultBuildRunner) run(build api.Build) error {
	var envVars []api.EnvVar
	for k, v := range build.Context {
		envVars = append(envVars, api.EnvVar{Name: k, Value: v})
	}

	job := api.Job{
		Pod: api.Pod{
			Labels: map[string]string{},
			DesiredState: api.PodState{
				Manifest: api.ContainerManifest{
					Version: "v1beta1",
					Containers: []api.Container{
						{
							Name:          "docker-build",
							Image:         "ironcladlou/openshift-docker-builder",
							Privileged:    true,
							RestartPolicy: "runOnce",
							Env:           envVars,
						},
					},
				},
			},
		},
	}

	glog.Infof("Creating job for build %s", build.ID)
	job, err := r.kubeClient.CreateJob(job)

	if err != nil {
		glog.Errorf("Error creating job for build ID %s: %s\n", build.ID, err.Error())

		build.Status = api.BuildComplete
		build.Success = false

		_, err := r.kubeClient.UpdateBuild(build)
		if err != nil {
			return fmt.Errorf("Couldn't update build: %#v : %s", build, err.Error())
		}

		//TODO should this return nil or some error?
		return nil
	}

	glog.Infof("Setting build state to pending, job id = %s", job.ID)
	build.Status = api.BuildPending
	build.JobID = job.ID

	_, err = r.kubeClient.UpdateBuild(build)
	if err != nil {
		return fmt.Errorf("Couldn't update Job: %#v : %s", job, err.Error())
	}

	return nil
}
