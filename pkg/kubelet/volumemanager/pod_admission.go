package volumemanager

import (
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/volumemanager/cache"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/csimigration"
	"k8s.io/kubernetes/pkg/volume/util"
)

// volumeAdmitHandler is a lifecycle.PodAdmitHandler that admits pods based on the volumes it uses
// and their compatibility with the volume manager's state.
type volumeAdmitHandler struct {
	desiredStateOfWorld      cache.DesiredStateOfWorld
	csiMigratedPluginManager csimigration.PluginManager
	intreeToCSITranslator    csimigration.InTreeToCSITranslator
	volumePluginMgr          *volume.VolumePluginMgr
	kubeClient               clientset.Interface
}

var _ lifecycle.PodAdmitHandler = &volumeAdmitHandler{}

func (v *volumeAdmitHandler) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	pod := attrs.Pod
	podVolumeInfos := util.NewPodVolumeInfos(pod, v.csiMigratedPluginManager, v.intreeToCSITranslator, v.volumePluginMgr, v.kubeClient)
	return v.desiredStateOfWorld.AdmitPodVolumes(pod, podVolumeInfos)
}
