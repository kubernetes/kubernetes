package predicates

import (
	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	schedulercache "k8s.io/kubernetes/pkg/scheduler/cache"
	"strconv"
	"github.com/golang/glog"
	"strings"
	"k8s.io/kubernetes/pkg/kubelet/apis"
)

const CSIMaxVolumeBytesLabel = "csi.kubernetes.io/max_bytes"

var CSIMaxVolumeBytesNoAvailableBytes = newPredicateFailureError("CSIMaxVolumeBytes", "node(s) exceed csi max volume bytes")

type CSIMaxVolumeBytesChecker struct {
	pvInfo  PersistentVolumeInfo
	pvcInfo PersistentVolumeClaimInfo
	storageCl StorageClassInfo
}

func NewCSIMaxVolumeBytesChecker(pvInfo PersistentVolumeInfo, pvcInfo PersistentVolumeClaimInfo, storageCl StorageClassInfo) algorithm.FitPredicate {
	c := &CSIMaxVolumeBytesChecker{
		pvInfo:  pvInfo,
		pvcInfo: pvcInfo,
		storageCl: storageCl,
	}
	return c.attachableLimitPredicate
}

func (c *CSIMaxVolumeBytesChecker) attachableLimitPredicate(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	// If a pod doesn't have any volume attached to it, the predicate will always be true.
	// Thus we make a fast path for it, to avoid unnecessary computations in this case.
	if len(pod.Spec.Volumes) == 0 {
		return true, nil, nil
	}

	// If node doesn't have max bytes label we have to assume it can't fit any volumes
	if _, ok := nodeInfo.Node().Labels[CSIMaxVolumeBytesLabel]; !ok {
		return false, []algorithm.PredicateFailureReason{CSIMaxVolumeBytesNoAvailableBytes}, nil
	}

	maxBytes := make(map[string]int64, 0)

	for label, value := range nodeInfo.Node().Labels {
		if strings.HasPrefix(label, CSIMaxVolumeBytesLabel) {
			csiName := strings.Replace(label, CSIMaxVolumeBytesLabel+"/", "", 1)
			csiMaxBytes, err := strconv.Atoi(value)
			// The CSI was dumb and didn't put a integer in the label so lets skip it
			if err != nil {
				glog.Infof("CSI Label %s doesn't have a int value", label)
				continue
			}
			maxBytes[csiName] = int64(csiMaxBytes)
		}
	}

	// No max bytes so nothing available
	if len(maxBytes) == 0 {
		return false, []algorithm.PredicateFailureReason{CSIMaxVolumeBytesNoAvailableBytes}, nil
	}

	attachedVolumes := nodeInfo.Node().Status.VolumesAttached
	glog.Infof("Attached volumes on node %s: ", nodeInfo.Node().Name, attachedVolumes)

	for _, fullVolName := range attachedVolumes {
		//what happens when volume is not seperated via ^
		volName := strings.Split(string(fullVolName.Name), "^")[1]

		pv, err := c.pvInfo.GetPersistentVolumeInfo(volName)
		if err != nil {
			glog.V(4).Infof("Unable to look up PV info for %s", volName)
			continue
		}

		csiSource := pv.Spec.PersistentVolumeSource.CSI
		if csiSource == nil {
			glog.V(4).Infof("Not considering non-CSI volume %s", volName)
			continue
		}

		if _, ok := maxBytes[csiSource.Driver]; ok {
			maxBytes[csiSource.Driver] -= pv.Spec.Capacity[v1.ResourceStorage].Value()

			if maxBytes[csiSource.Driver] <= 0 {
				return false, []algorithm.PredicateFailureReason{CSIMaxVolumeBytesNoAvailableBytes}, nil
			}
		}
	}


	for _, podVol := range pod.Spec.Volumes {
		if podVol.PersistentVolumeClaim == nil {
			continue
		}

		pvc, err := c.pvcInfo.GetPersistentVolumeClaimInfo(pod.Namespace, podVol.Name)
		if err != nil {
			glog.V(4).Infof("Unable to look up PVC info in namespace %s on pod %ws for %s", pod.Namespace, pod.Name, podVol.Name)
			continue
		}

		// If pvc is bound only consider it if it is on the node
		if pvc.Status.Phase == v1.ClaimBound {
			pv, err := c.pvInfo.GetPersistentVolumeInfo(pvc.Spec.VolumeName)
			if err != nil {
				glog.V(4).Infof("Unable to look up PV info for %s", pvc.Spec.VolumeName)
				continue
			}

			myNode := false
			for _, nst := range pv.Spec.NodeAffinity.Required.NodeSelectorTerms {
				for _, nsr := range nst.MatchExpressions {
					if nsr.Key == apis.LabelHostname && nsr.Operator == v1.NodeSelectorOpIn {
						for _, value := range nsr.Values {
							if value == nodeInfo.Node().Name {
								myNode = true
								break
							}
						}
						if myNode {
							break
						}
					}
				}
				if myNode {
					break
				}
			}
			if !myNode {
				continue
			}
		}


		// Going to assume storage class name = driver name
		if _, ok := maxBytes[*pvc.Spec.StorageClassName]; ok {
			maxBytes[*pvc.Spec.StorageClassName] -= pvc.Spec.Resources.Requests[v1.ResourceStorage].Value()
			if maxBytes[*pvc.Spec.StorageClassName] <= 0 {
				return false, []algorithm.PredicateFailureReason{CSIMaxVolumeBytesNoAvailableBytes}, nil
			}
		}
	}


	return true, nil, nil
}