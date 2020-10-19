package deployment

import (
	"fmt"
	"sort"
	"strings"

	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog"
	podUtils "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/utils/integer"
)

// 滚动升级
func (dc *DeploymentController) rolloutRollingInPlace(deployment *apps.Deployment, rsList []*apps.ReplicaSet, podMap map[types.UID]*v1.PodList) error {
	// 先确保 newRS 存在，如果原来存在不修改replicas大小，如果不存在设置为0，
	tmpDeployment := deployment.DeepCopy()
	*tmpDeployment.Spec.Replicas = 0
	newRS, oldRSs, err := dc.getAllReplicaSetsAndSyncRevision(tmpDeployment, rsList, true)
	if err != nil {
		return err
	}
	//newRS, err = dc.rsLister.ReplicaSets(newRS.Namespace).Get(newRS.Name)
	//if err != nil {
	//	return err
	//}

	allRSs := append(oldRSs, newRS)

	updateInplaceCount, err := dc.updateInplacePods(oldRSs, newRS, deployment, podMap)
	if updateInplaceCount > 0 {
		scalaCount := *(newRS.Spec.Replicas) + updateInplaceCount
		_, newRS, inErr := dc.scaleReplicaSetAndRecordEventTry(true, newRS, scalaCount, deployment)
		if inErr != nil {
			return inErr
		}
		return dc.syncStatus(allRSs, newRS, deployment)
	}
	if err != nil {
		return err
	}
	// clean un health 和 pods 超标
	scala, err := dc.reconcileOldRSS(allRSs, oldRSs, newRS, deployment)
	if err != nil {
		return err
	}
	if scala {
		return dc.syncStatus(allRSs, newRS, deployment)
	}

	// select pod to unready and Scale down rs , if we can.
	unreadyCount, err := dc.selectPodsToUnReady(allRSs, controller.FilterActiveReplicaSets(oldRSs), newRS, deployment, podMap)
	if err != nil {
		return err
	}
	if unreadyCount > 0 {
		return dc.syncStatus(allRSs, newRS, deployment)
	}

	// Scale up, if we can.
	scaledUp, err := dc.reconcileNewReplicaSet(allRSs, newRS, deployment)
	if err != nil {
		return err
	}
	if scaledUp {
		// Update DeploymentStatus
		err = dc.syncStatus(allRSs, newRS, deployment)
	}

	if deploymentutil.DeploymentComplete(deployment, &deployment.Status) {
		if err := dc.cleanupDeployment(oldRSs, deployment); err != nil {
			return err
		}
	}

	// Sync deployment status
	return dc.syncStatus(allRSs, newRS, deployment)
}

func (dc *DeploymentController) updateInplacePods(oldRSs []*apps.ReplicaSet, newRS *apps.ReplicaSet, deployment *apps.Deployment, podMap map[types.UID]*v1.PodList) (int32, error) {
	count := int32(0)
	totalCount := int32(0)
	var err error = nil
	for i, oldRS := range oldRSs {
		if podItems, ok := podMap[oldRS.UID]; ok {
			for _, pod := range podItems.Items {
				if deploymentutil.CanUpdateInplace(pod) {
					if err = dc.updateInplacePod(&pod, oldRS, newRS); err == nil {
						count += 1
					} else {
						break
					}
				}
			}
			newReplicasCount := integer.Int32Max(0, *(oldRS.Spec.Replicas)-count)
			totalCount += count
			_, updatedOldRS, err := dc.scaleReplicaSetAndRecordEventTry(true, oldRS, newReplicasCount, deployment)
			if err != nil {
				return totalCount, err
			}
			oldRSs[i] = updatedOldRS
		}
	}
	return totalCount, err
}

func (dc *DeploymentController) updateInplacePod(pod *v1.Pod, oldRS, newRS *apps.ReplicaSet) error {
	podTemplateSpecHash, _ := newRS.Labels[apps.DefaultDeploymentUniqueLabelKey]
	oldPodTemplateSpecHash, _ := oldRS.Labels[apps.DefaultDeploymentUniqueLabelKey]
	cpod := pod.DeepCopy()
	cpod.Labels[apps.DefaultDeploymentUniqueLabelKey] = podTemplateSpecHash
	for _, rsContainer := range newRS.Spec.Template.Spec.Containers {
		for i, _ := range pod.Spec.Containers {
			if cpod.Spec.Containers[i].Name == rsContainer.Name {
				cpod.Spec.Containers[i].Image = rsContainer.Image
			}
		}
	}
	cpod.Annotations = pod.Annotations
	if cpod.Annotations == nil {
		cpod.Annotations = map[string]string{}
	}
	if val, ok := cpod.Annotations[deploymentutil.PodUpdateInplaceRSHistory]; ok {
		arrs := strings.Split(val, ",")
		if len(arrs) > 10 {
			arrs = arrs[len(arrs)-10:]
		}
		arrs = append(arrs, oldPodTemplateSpecHash)
		cpod.Annotations[deploymentutil.PodUpdateInplaceRSHistory] = strings.Join(arrs, ",")
	} else {
		cpod.Annotations[deploymentutil.PodUpdateInplaceRSHistory] = oldPodTemplateSpecHash
	}
	cpod.OwnerReferences = nil
	_, err := dc.client.CoreV1().Pods(newRS.Namespace).Update(cpod)
	if err != nil {
		klog.Errorf("update inplace pod %s %s to rs hash %s failed, %v", cpod.Namespace, cpod.Name, podTemplateSpecHash, err)
		return err
	}
	return err
}

func (dc *DeploymentController) selectPodsToUnReady(allRSs []*apps.ReplicaSet, oldRSs []*apps.ReplicaSet,
	newRS *apps.ReplicaSet, deployment *apps.Deployment, podMap map[types.UID]*v1.PodList) (int32, error) {

	maxUnavailable := deploymentutil.MaxUnavailable(*deployment)
	// Check if we can scale down.
	minAvailable := *(deployment.Spec.Replicas) - maxUnavailable
	// Find the number of available pods.
	availablePodCount := deploymentutil.GetAvailableReplicaCountForReplicaSets(allRSs)
	if availablePodCount <= minAvailable {
		// Cannot scale down.
		return 0, nil
	}

	sort.Sort(controller.ReplicaSetsByCreationTimestamp(oldRSs))

	totalUnReady := int32(0)
	maxUnReadyCount := integer.Int32Min(*(deployment.Spec.Replicas)-*(newRS.Spec.Replicas), availablePodCount-minAvailable)
	klog.V(4).Infof("Found %d available pods in deployment %s, update inplace %d pods in old RSes", availablePodCount, deployment.Name, maxUnReadyCount)

	for _, targetRS := range oldRSs {
		if totalUnReady >= maxUnReadyCount {
			break
		}
		if *(targetRS.Spec.Replicas) == 0 {
			continue
		}
		// inplace down. 当前rs需要缩小 inplaceCount 个pod，不能比当前rs的Replicas大
		unreadyCount := int32(integer.IntMin(int(*(targetRS.Spec.Replicas)), int(maxUnReadyCount-totalUnReady)))
		newReplicasCount := *(targetRS.Spec.Replicas) - unreadyCount
		if newReplicasCount > *(targetRS.Spec.Replicas) {
			return 0, fmt.Errorf("when updating old RS InPlace, got invalid request to update inplace%s/%s %d -> %d", targetRS.Namespace, targetRS.Name, *(targetRS.Spec.Replicas), newReplicasCount)
		}
		count := int32(0)
		if podItems, ok := podMap[targetRS.UID]; ok {
			for _, pod := range podItems.Items {
				if count >= unreadyCount {
					break
				}
				err := dc.convertPodToUnReady(&pod)
				if err == nil {
					count += 1
				}
			}
		}
		totalUnReady += count
	}

	return totalUnReady, nil
}

func (dc *DeploymentController) convertPodToUnReady(pod *v1.Pod) error {
	podTemplateSpecHash, _ := pod.Labels[apps.DefaultDeploymentUniqueLabelKey]
	conditions := pod.Status.Conditions
	pod.Status.Conditions = nil
	change1 := false
	for _, condition := range conditions {
		if condition.Type == v1.PodReady {
			continue
		}
		if condition.Type == v1.ContainersReady {
			continue
		}

		add := true
		for _, val := range pod.Spec.ReadinessGates {
			if val.ConditionType == condition.Type {
				add = false
				break
			}
		}
		if add {
			pod.Status.Conditions = append(pod.Status.Conditions, condition)
			change1 = true
		}

	}

	change4 := podUtils.UpdatePodCondition(&pod.Status, &v1.PodCondition{
		Type:    deploymentutil.OriginPodTemplateSpecHash,
		Status:  v1.ConditionStatus(podTemplateSpecHash),
		Message: "for update inplace",
	})
	//for i, _ := range pod.Status.ContainerStatuses {
	//	pod.Status.ContainerStatuses[i].RestartCount = 0
	//}
	if change1 || change4 {
		_, err := dc.client.CoreV1().Pods(pod.Namespace).UpdateStatus(pod)
		if err != nil {
			klog.Errorf("update inplace pod %s %s status failed, %v", pod.Namespace, pod.Name, err)
		}
		return err
	}
	return nil
}

func (dc *DeploymentController) syncStatus(allRSs []*apps.ReplicaSet, newRS *apps.ReplicaSet, deployment *apps.Deployment) error {
	d, err := dc.dLister.Deployments(deployment.Namespace).Get(deployment.Name)
	if err != nil {
		return err
	}
	return dc.syncRolloutStatus(allRSs, newRS, d)
}

// 失败时，检查下是不是rs变化，如果rs的Replicas没有变，则 scala
func (dc *DeploymentController) scaleReplicaSetAndRecordEventTry(retry bool, rs *apps.ReplicaSet, newScale int32, deployment *apps.Deployment) (bool, *apps.ReplicaSet, error) {
	flag, update, err := dc.scaleReplicaSetAndRecordEvent(rs, newScale, deployment)
	if err == nil || !retry {
		return flag, update, err
	}
	new, nErr := dc.client.AppsV1().ReplicaSets(rs.Namespace).Get(rs.Name, metav1.GetOptions{})
	if nErr != nil || *(new.Spec.Replicas) != *(rs.Spec.Replicas) {
		return false, update, err
	}
	return dc.scaleReplicaSetAndRecordEventTry(false, new, newScale, deployment)
}

func (dc *DeploymentController) reconcileOldRSS(allRSs []*apps.ReplicaSet, oldRSs []*apps.ReplicaSet, newRS *apps.ReplicaSet, deployment *apps.Deployment) (bool, error) {
	oldPodsCount := deploymentutil.GetReplicaCountForReplicaSets(oldRSs)
	if oldPodsCount == 0 {
		return false, nil
	}

	allPodsCount := deploymentutil.GetReplicaCountForReplicaSets(allRSs)
	klog.V(4).Infof("New replica set %s/%s has %d available pods.", newRS.Namespace, newRS.Name, newRS.Status.AvailableReplicas)
	maxUnavailable := deploymentutil.MaxUnavailable(*deployment)

	minAvailable := *(deployment.Spec.Replicas) - maxUnavailable
	newRSUnavailablePodCount := *(newRS.Spec.Replicas) - newRS.Status.AvailableReplicas
	maxScaledDown := allPodsCount - minAvailable - newRSUnavailablePodCount
	if maxScaledDown <= 0 {
		return false, nil
	}

	oldRSs, totalScaledDown, err := dc.cleanupUnhealthyReplicas(oldRSs, deployment, maxScaledDown)
	if err != nil {
		return false, nil
	}
	klog.V(4).Infof("Cleaned up unhealthy replicas from old RSes by %d", totalScaledDown)
	if *(newRS.Spec.Replicas) >= *(deployment.Spec.Replicas) {
		allRSs = append(oldRSs, newRS)
		scaledDownCount, err := dc.scaleDownOldReplicaSetsForRollingUpdate(allRSs, oldRSs, deployment)
		if err != nil {
			return false, nil
		}
		klog.V(4).Infof("Scaled down old RSes of deployment %s by %d", deployment.Name, scaledDownCount)
		totalScaledDown += scaledDownCount

	}
	return totalScaledDown > 0, nil
}
