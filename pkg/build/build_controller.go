package build

import (
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

type BuildController struct {
	kubeClient client.Interface
	syncTime   <-chan time.Time
}

func MakeBuildController(kubeClient client.Interface) *BuildController {
	bc := &BuildController{
		kubeClient: kubeClient,
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
		glog.Errorf("Error listing builds: %v (%#v)", err, err)
		return
	}

	for _, build := range builds.Items {
		nextStatus, err = bc.process(build)
		if err != nil {
			glog.Errorf("Error processing build ID %v: %#v", build.ID, err)
		}

		if nextStatus != build.Status {
			build.Status = nextStatus
			if _, err := bc.kubeClient.UpdateBuild(build); err != nil {
				glog.Errorf("Error updating build ID %v to status %v: %#v", build.ID, nextStatus, err)
			}
		}
	}
}

// Sync loop brings the build state into sync with its corresponding job.
// TODO: improve handling of illegal state transitions
func (bc *BuildController) process(build *api.Build) (api.BuildStatus, error) {
	glog.Infof("Syncing build %#v", build)

	switch build.Status {
	case api.BuildNew:
		build.PodID = "build-" + build.BuildConfig.Type // TODO: better naming
		return api.BuildPending, nil
	case api.BuildPending:
		podSpec := bc.buildPodSpec(build)
		_, err := r.kubeClient.CreatePod(podSpec)

		// TODO: strongly typed error checking
		switch {
		case err == "pod_exists":
			return build.Status, err // no transition, already handled by someone else
		case err != nil:
			return api.BuildFailed, err
		}

		return api.BuildRunning, nil
	case api.BuildRunning:
		pod, err := bc.kubeClient.GetPod(build.PodID)
		if err != nil {
			glog.Errorf("Error retrieving pod for build ID %v: %#v", build.ID, err)
			return build.Status, err
		}

		// check the exit codes of all the containers in the pod
		if pod.CurrentState.Status == api.PodStopped {
			for _, info := range pod.CurrentState.Info {
				if info.State.ExitCode != 0 {
					return api.BuildFailed, nil
				}
			}

			return api.BuildComplete, nil
		}
	}

	return build.Status, nil
}

func (bc *BuildController) buildPodSpec(build api.Build) api.Pod {
	return api.Pod{
		ID:     build.PodID,
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
					},
				},
			},
		},
	}
}
