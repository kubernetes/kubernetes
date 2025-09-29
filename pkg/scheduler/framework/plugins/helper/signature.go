package helper

import v1 "k8s.io/api/core/v1"

// Common signature element: the pod's Volumes.  Note that
// we exclude ConfigMap and Secret volumes because they are synthetic.
func SignatureVolumes(pod *v1.Pod) any {
	volumes := []v1.Volume{}
	for _, volume := range pod.Spec.Volumes {
		if volume.VolumeSource.ConfigMap == nil && volume.VolumeSource.Secret == nil {
			volumes = append(volumes, volume)
		}
	}
	return volumes
}
