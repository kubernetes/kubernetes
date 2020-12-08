package checkpoint

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/checkpoint-restore/checkpointctl/lib"
	"github.com/pkg/errors"
	v1 "k8s.io/api/core/v1"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

func UpdateMetadata(pod *v1.Pod, podSandBoxID, checkpointDir string, containerTerminationLogPathUID map[string]string) error {
	var checkpointedPods metadata.CheckpointedPods

	// First read the file of checkpointed pods if it exists
	err, checkpointMetadata, _ := metadata.ReadCheckpointedPods(checkpointDir)
	if err != nil && !os.IsNotExist(errors.Unwrap(errors.Unwrap(err))) {
		return err
	}
	if checkpointMetadata == nil {
		checkpointMetadata = &metadata.CheckpointMetadata{}
	}

	checkpointedPods.ID = podSandBoxID
	checkpointedPods.PodUID = string(pod.UID)
	checkpointedPods.Name = pod.Name
	checkpointedPods.Namespace = pod.Namespace
	checkpointedPods.TerminationGracePeriod = *pod.Spec.TerminationGracePeriodSeconds
	checkpointedPods.ConfigSource = pod.Annotations[kubetypes.ConfigSourceAnnotationKey]
	checkpointedPods.ConfigSeen = pod.Annotations[kubetypes.ConfigFirstSeenAnnotationKey]
	checkpointedPods.HostIP = pod.Status.HostIP
	checkpointedPods.PodIP = pod.Status.PodIP

	for _, ip := range pod.Status.PodIPs {
		checkpointedPods.PodIPs = append(checkpointedPods.PodIPs, ip.IP)
	}

	for _, c := range pod.Status.ContainerStatuses {
		var checkpointedContainers metadata.CheckpointedContainers
		checkpointedContainers.Name = c.Name
		checkpointedContainers.ID = strings.Split(c.ContainerID, "://")[1]
		checkpointedContainers.RestartCounter = c.RestartCount
		checkpointedContainers.TerminationMessagePathUID = containerTerminationLogPathUID[c.Name]
		for _, cc := range pod.Spec.Containers {
			if c.Name == cc.Name {
				checkpointedContainers.TerminationMessagePath = cc.TerminationMessagePath
				checkpointedContainers.TerminationMessagePolicy = string(cc.TerminationMessagePolicy)
				checkpointedContainers.Image = cc.Image
			}
		}
		checkpointedPods.Containers = append(checkpointedPods.Containers, checkpointedContainers)
	}

	checkpointMetadata.CheckpointedPods = append(checkpointMetadata.CheckpointedPods, checkpointedPods)

	if err := metadata.WriteKubeletCheckpointsMetadata(checkpointMetadata, checkpointDir); err != nil {
		return err
	}

	return nil
}
