package build

import (
	"fmt"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

type buildTypeStrategy func(api.Build) api.Pod

type BuildController struct {
	kubeClient         client.Interface
	syncTime           <-chan time.Time
	typeStrategies     map[api.BuildType]buildTypeStrategy
	dockerBuilderImage string
	dockerRegistry     string
}

func MakeBuildController(kubeClient client.Interface, dockerBuilderImage, dockerRegistry string) *BuildController {
	glog.Infof("Creating build controller with dockerBuilderImage=%s, dockerRegistry=%s", dockerBuilderImage, dockerRegistry)
	bc := &BuildController{
		kubeClient:         kubeClient,
		dockerBuilderImage: dockerBuilderImage,
		dockerRegistry:     dockerRegistry,
	}

	bc.typeStrategies = map[api.BuildType]buildTypeStrategy{
		api.BuildType("docker"): bc.dockerBuildStrategy,
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
		nextStatus, err := bc.process(&build)
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

// Determine the next status of a build given its current state and the state
// of its associated pod.
// TODO: improve handling of illegal state transitions
func (bc *BuildController) process(build *api.Build) (api.BuildStatus, error) {
	glog.Infof("Syncing build %#v", build)

	switch build.Status {
	case api.BuildNew:
		build.PodID = "build-" + string(build.Config.Type) // TODO: better naming
		return api.BuildPending, nil
	case api.BuildPending:
		makePodSpec, ok := bc.typeStrategies[build.Config.Type]
		if !ok {
			return build.Status, fmt.Errorf("No build type for %s")
		}

		podSpec := makePodSpec(*build)
		_, err := bc.kubeClient.CreatePod(podSpec)

		// TODO: strongly typed error checking
		if err != nil {
			if strings.Index(err.Error(), "already exists") != -1 {
				return build.Status, err // no transition, already handled by someone else
			}

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

func (bc BuildController) dockerBuildStrategy(build api.Build) api.Pod {
	return api.Pod{
		JSONBase: api.JSONBase{
			ID: build.PodID,
		},
		DesiredState: api.PodState{
			Manifest: api.ContainerManifest{
				Version: "v1beta1",
				Containers: []api.Container{
					{
						Name:          "docker-build",
						Image:         bc.dockerBuilderImage,
						Privileged:    true,
						RestartPolicy: "runOnce",
						Env: []api.EnvVar{
							{Name: "BUILD_TAG", Value: build.Config.ImageTag},
							{Name: "DOCKER_CONTEXT_URL", Value: build.Config.SourceURI},
							{Name: "DOCKER_REGISTRY", Value: bc.dockerRegistry},
						},
					},
				},
			},
		},
	}
}
