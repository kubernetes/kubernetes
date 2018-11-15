package predicates

import (
	"strconv"
	"strings"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	schedulercache "k8s.io/kubernetes/pkg/scheduler/cache"
	"k8s.io/kubernetes/pkg/scheduler/volumebinder"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

const CSIMaxVolumeBytesLabel = "csi.kubernetes.io/max_bytes-"

var CSIMaxVolumeBytesNoAvailableBytes = newPredicateFailureError("CSIMaxVolumeBytes", "node(s) exceed csi max volume bytes")

type CSIMaxVolumeBytesChecker struct {
	pvInfo  PersistentVolumeInfo
	pvcInfo PersistentVolumeClaimInfo
	binder  *volumebinder.VolumeBinder
}

func NewCSIMaxVolumeBytesChecker(pvInfo PersistentVolumeInfo, pvcInfo PersistentVolumeClaimInfo, binder  *volumebinder.VolumeBinder) algorithm.FitPredicate {
	c := &CSIMaxVolumeBytesChecker{
		pvInfo:  pvInfo,
		pvcInfo: pvcInfo,
		binder: binder,
	}
	return c.predicate
}

func (c *CSIMaxVolumeBytesChecker) predicate(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	glog.Info("IN CSIMaxVolumeBytesChecker predicate")

	// If a pod doesn't have any volume attached to it, the predicate will always be true.
	// Thus we make a fast path for it, to avoid unnecessary computations in this case.
	if len(pod.Spec.Volumes) == 0 {
		return true, nil, nil
	}

	maxBytes := make(map[string]int64, 0)

	glog.Info("Finding CSI Max Bytes Labels")
	for label, value := range nodeInfo.Node().Labels {
		if strings.HasPrefix(label, CSIMaxVolumeBytesLabel) {
			csiName := strings.Replace(label, CSIMaxVolumeBytesLabel, "", 1)
			csiMaxBytes, err := strconv.Atoi(value)
			// The CSI was dumb and didn't put a integer in the label so lets skip it
			if err != nil {
				glog.Infof("CSI Label %s doesn't have a int value", label)
				continue
			}
			maxBytes[csiName] = int64(csiMaxBytes)
		}
	}

	//No max bytes so nothing available
	if len(maxBytes) == 0 {
		glog.Infof("No maxBytes label for node %s", nodeInfo.Node().Name)
		return true, nil, nil
	}

	for _, podVol := range pod.Spec.Volumes {
		if podVol.PersistentVolumeClaim == nil {
			continue
		}

		pvc, err := c.pvcInfo.GetPersistentVolumeClaimInfo(pod.Namespace, podVol.PersistentVolumeClaim.ClaimName)
		if err != nil {
			glog.V(4).Infof("Unable to look up PVC info in namespace %s on pod %ws for %s", pod.Namespace, pod.Name, podVol.Name)
			continue
		}

		// Going to assume storage class name = driver name
		if _, ok := maxBytes[*pvc.Spec.StorageClassName]; ok && pvc.Spec.VolumeName == "" {
			capacity := pvc.Spec.Resources.Requests[v1.ResourceStorage]
			capacityPtr := &capacity
			maxBytes[*pvc.Spec.StorageClassName] -= capacityPtr.Value()
			if maxBytes[*pvc.Spec.StorageClassName] <= 0 {
				return false, []algorithm.PredicateFailureReason{CSIMaxVolumeBytesNoAvailableBytes}, nil
			}
		}
	}

	for k := range maxBytes {
		volsForClass := c.binder.Binder.GetPVAssumeCache().ListPVs(k)
		glog.Infof("Attached volumes in class %s on node %s: ", k, nodeInfo.Node().Name, volsForClass)

		for _, pv := range volsForClass {
			csiSource := pv.Spec.PersistentVolumeSource.CSI
			if csiSource == nil {
				glog.V(4).Infof("Not considering non-CSI volume %s", pv.Name)
				continue
			}
			err := volumeutil.CheckNodeAffinity(pv, nodeInfo.Node().Labels)
			if err != nil {
				continue
			}

			if _, ok := maxBytes[k]; ok {
				capacity := pv.Spec.Capacity[v1.ResourceStorage]
				capacityPtr := &capacity
				maxBytes[k] -= capacityPtr.Value()


				if maxBytes[k] <= 0 {
					return false, []algorithm.PredicateFailureReason{CSIMaxVolumeBytesNoAvailableBytes}, nil
				}
			}
		}
	}

	return true, nil, nil
}