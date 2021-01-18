package checkpoint

import (
	"os"
	"strings"

	"github.com/checkpoint-restore/checkpointctl/lib"
	"github.com/pkg/errors"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"
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

func GetPodsToRestore(checkpointDir string) ([]*v1.Pod, []*map[string]string, error) {
	var pods []*v1.Pod
	var uids []*map[string]string
	err, checkpointMetadata, _ := metadata.ReadCheckpointedPods(checkpointDir)
	if err != nil {
		if !os.IsNotExist(errors.Unwrap(errors.Unwrap(err))) {
			klog.Warning("Error reading checkpoint metadata %v", err)
		}
		return nil, nil, err
	}

	for _, p := range checkpointMetadata.CheckpointedPods {
		var pod v1.Pod
		pod.UID = types.UID(p.PodUID)
		pod.Name = p.Name
		pod.GenerateName = p.ID
		pod.Namespace = p.Namespace
		terminationMessagePathUID := make(map[string]string)
		for _, c := range p.Containers {
			var containerStatus v1.ContainerStatus
			terminationMessagePathUID[c.Name] = c.TerminationMessagePathUID
			containerStatus.Name = c.Name
			containerStatus.ContainerID = c.ID
			containerStatus.RestartCount = c.RestartCounter
			pod.Status.ContainerStatuses = append(pod.Status.ContainerStatuses, containerStatus)

			var container v1.Container
			container.Name = c.Name
			container.TerminationMessagePath = c.TerminationMessagePath
			container.TerminationMessagePolicy = v1.TerminationMessagePolicy(c.TerminationMessagePolicy)
			container.Image = c.Image
			pod.Spec.Containers = append(pod.Spec.Containers, container)
		}
		if len(terminationMessagePathUID) != len(p.Containers) {
			// It is not possible to restore Pods where a container has no
			// information about its terminationMessagePathUID
			continue
		}

		pod.Status.HostIP = p.HostIP
		pod.Status.PodIP = p.PodIP
		for _, ip := range p.PodIPs {
			var pip v1.PodIP
			pip.IP = ip
			pod.Status.PodIPs = append(pod.Status.PodIPs, pip)
		}

		pod.Annotations = make(map[string]string)
		pod.Annotations[kubetypes.ConfigSourceAnnotationKey] = p.ConfigSource
		pod.Annotations[kubetypes.ConfigFirstSeenAnnotationKey] = p.ConfigSeen
		pod.Annotations["kubernetes.io/restored.pod"] = "true"
		pod.Labels = make(map[string]string)
		pod.Labels[kubetypes.KubernetesPodNameLabel] = p.Name
		pod.Labels[kubetypes.KubernetesPodNamespaceLabel] = p.Namespace
		pod.Labels[kubetypes.KubernetesPodUIDLabel] = string(p.PodUID)
		pod.Spec.TerminationGracePeriodSeconds = &p.TerminationGracePeriod
		pods = append(pods, &pod)
		uids = append(uids, &terminationMessagePathUID)
	}
	return pods, uids, nil
}
