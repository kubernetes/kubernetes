package deployment

import (
	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/deployment/util"
)

/***
	1. 根据升级策略先选择pod，通过设置 Conditions 表示这些 pod 需要迁移到新的 rs 上，
       另外也根据 ReadinessGates 将 pod 设置成 un ready（如果deploy没有配置，则无法控制pod升级的时候，从endpoint中摘除）
    2. 将第一步设置的 pod 的 label 改为 newRS 的 label，同时设置 image 为 newRS 的 image
    3. 根据第二步改变的 pod 个数， 设置 newRS 和 oldRS 到对应的 Replicas。

	另外，由于 label change 和设置 rs 的 Replicas 是分两不走， pod 改变 label 会触发 oldRS 和 newRS 事件，导致两个 RS 的 pod count 不是期望的，从删除或者增加 POD，
	通过在 replica_set controller 对 label change 事件做延后入队列，给足时间让 Replicas 设置成正确的值。
	但是可能由于其他原因，导致 label change 到 Replicas 设置这个时间段仍然出现创建多余的pod或者删除不该删除的pod
	但是一半新创建的pod，ready比原地重启的慢，删除大概率的是新创建的，暂时不去考虑这种比较少的场景，毕竟第一步到第二步是先后调用的api，概率小
*/

// recreate
func (dc *DeploymentController) rolloutRecreateInPlace(deployment *apps.Deployment, rsList []*apps.ReplicaSet, podMap map[types.UID]*v1.PodList) error {
	tmpDeploy := deployment.DeepCopy()
	*tmpDeploy.Spec.Replicas = 0
	newRS, oldRSs, err := dc.getAllReplicaSetsAndSyncRevision(tmpDeploy, rsList, true)
	if err != nil {
		return err
	}

	allRSs := append(oldRSs, newRS)
	activeOldRSs := controller.FilterActiveReplicaSets(oldRSs)

	// relabel 并设置 newRS
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

	// 选pod 并设置为 unready
	totalUnReady, err := dc.selectPodsToUnReadyForRecreate(activeOldRSs, newRS, deployment, podMap)
	if err != nil {
		return err
	}
	if totalUnReady > 0 {
		return dc.syncRolloutStatus(allRSs, newRS, deployment)
	}

	// 保低，保证recreate能成功
	// scale down old replica sets.
	scaledDown, err := dc.scaleDownOldReplicaSetsForRecreate(activeOldRSs, deployment)
	if err != nil {
		return err
	}
	if scaledDown {
		// Update DeploymentStatus.
		return dc.syncRolloutStatus(allRSs, newRS, deployment)
	}

	// scale up new replica set.
	if _, err := dc.scaleUpNewReplicaSetForRecreate(newRS, deployment); err != nil {
		return err
	}

	if util.DeploymentComplete(deployment, &deployment.Status) {
		if err := dc.cleanupDeployment(oldRSs, deployment); err != nil {
			return err
		}
	}

	// Sync deployment status.
	return dc.syncRolloutStatus(allRSs, newRS, deployment)
}

func (dc *DeploymentController) selectPodsToUnReadyForRecreate(oldRSs []*apps.ReplicaSet, newRS *apps.ReplicaSet,
	deployment *apps.Deployment, podMap map[types.UID]*v1.PodList) (int32, error) {
	totalUnReady := int32(0)
	var err error
	for i := range oldRSs {
		if err != nil {
			return totalUnReady, err
		}
		rs := oldRSs[i]
		if *(rs.Spec.Replicas) == 0 {
			continue
		}
		unReady := int32(0)
		if podList, ok := podMap[rs.UID]; ok {
			// 将老pod都转移到新的rs下
			for _, pod := range podList.Items {
				switch pod.Status.Phase {
				case v1.PodFailed, v1.PodSucceeded:
					// Don't count pods in terminal state.
					continue
				default:
					err = dc.convertPodToUnReady(&pod)
					if err != nil {
						klog.Warningf("change replica set %s 's pod [%s] to [%s] failed, %v", rs.Name, pod.Name, newRS.Name, err)
					} else {
						unReady += 1
					}
				}
			}
		}
		if unReady > 0 {
			totalUnReady += unReady
			dc.eventRecorder.Eventf(deployment, v1.EventTypeNormal, "UpdateInplaceDeployment", "Recreate UpdateInplace %s to replica set %s %d pods unready", deployment.Name, rs.Name, unReady)
		}
	}
	return totalUnReady, nil
}
