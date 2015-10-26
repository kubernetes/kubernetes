/*
Copyright 2014 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package persistentvolume

import (
	"fmt"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	ioutil "k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/watch"
)

var _ volume.VolumeHost = &PersistentVolumeRecycler{}

// PersistentVolumeRecycler is a controller that watches for PersistentVolumes that are released from their claims.
// This controller will Recycle those volumes whose reclaim policy is set to PersistentVolumeReclaimRecycle and make them
// available again for a new claim.
type PersistentVolumeRecycler struct {
	volumeController *framework.Controller
	stopChannel      chan struct{}
	client           recyclerClient
	kubeClient       client.Interface
	pluginMgr        volume.VolumePluginMgr
}

// PersistentVolumeRecycler creates a new PersistentVolumeRecycler
func NewPersistentVolumeRecycler(kubeClient client.Interface, syncPeriod time.Duration, plugins []volume.VolumePlugin) (*PersistentVolumeRecycler, error) {
	recyclerClient := NewRecyclerClient(kubeClient)
	recycler := &PersistentVolumeRecycler{
		client:     recyclerClient,
		kubeClient: kubeClient,
	}

	if err := recycler.pluginMgr.InitPlugins(plugins, recycler); err != nil {
		return nil, fmt.Errorf("Could not initialize volume plugins for PVClaimBinder: %+v", err)
	}

	_, volumeController := framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return kubeClient.PersistentVolumes().List(labels.Everything(), fields.Everything())
			},
			WatchFunc: func(resourceVersion string) (watch.Interface, error) {
				return kubeClient.PersistentVolumes().Watch(labels.Everything(), fields.Everything(), resourceVersion)
			},
		},
		&api.PersistentVolume{},
		// TODO: Can we have much longer period here?
		syncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				pv := obj.(*api.PersistentVolume)
				recycler.reclaimVolume(pv)
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				pv := newObj.(*api.PersistentVolume)
				recycler.reclaimVolume(pv)
			},
		},
	)

	recycler.volumeController = volumeController
	return recycler, nil
}

func (recycler *PersistentVolumeRecycler) reclaimVolume(pv *api.PersistentVolume) error {
	if pv.Status.Phase == api.VolumeReleased && pv.Spec.ClaimRef != nil {
		glog.V(5).Infof("Reclaiming PersistentVolume[%s]\n", pv.Name)

		latest, err := recycler.client.GetPersistentVolume(pv.Name)
		if err != nil {
			return fmt.Errorf("Could not find PersistentVolume %s", pv.Name)
		}
		if latest.Status.Phase != api.VolumeReleased {
			return fmt.Errorf("PersistentVolume[%s] phase is %s, expected %s.  Skipping.", pv.Name, latest.Status.Phase, api.VolumeReleased)
		}

		// both handleRecycle and handleDelete block until completion
		// TODO: allow parallel recycling operations to increase throughput
		switch pv.Spec.PersistentVolumeReclaimPolicy {
		case api.PersistentVolumeReclaimRecycle:
			err = recycler.handleRecycle(pv)
		case api.PersistentVolumeReclaimDelete:
			err = recycler.handleDelete(pv)
		case api.PersistentVolumeReclaimRetain:
			glog.V(5).Infof("Volume %s is set to retain after release.  Skipping.\n", pv.Name)
		default:
			err = fmt.Errorf("No PersistentVolumeReclaimPolicy defined for spec: %+v", pv)
		}
		if err != nil {
			errMsg := fmt.Sprintf("Could not recycle volume spec: %+v", err)
			glog.Errorf(errMsg)
			return fmt.Errorf(errMsg)
		}
	}
	return nil
}

func (recycler *PersistentVolumeRecycler) handleRecycle(pv *api.PersistentVolume) error {
	glog.V(5).Infof("Recycling PersistentVolume[%s]\n", pv.Name)

	currentPhase := pv.Status.Phase
	nextPhase := currentPhase

	spec := volume.NewSpecFromPersistentVolume(pv, false)
	plugin, err := recycler.pluginMgr.FindRecyclablePluginBySpec(spec)
	if err != nil {
		return fmt.Errorf("Could not find recyclable volume plugin for spec: %+v", err)
	}
	volRecycler, err := plugin.NewRecycler(spec)
	if err != nil {
		return fmt.Errorf("Could not obtain Recycler for spec: %+v", err)
	}
	// blocks until completion
	err = volRecycler.Recycle()
	if err != nil {
		glog.Errorf("PersistentVolume[%s] failed recycling: %+v", pv.Name, err)
		pv.Status.Message = fmt.Sprintf("Recycling error: %s", err)
		nextPhase = api.VolumeFailed
	} else {
		glog.V(5).Infof("PersistentVolume[%s] successfully recycled\n", pv.Name)
		nextPhase = api.VolumePending
		if err != nil {
			glog.Errorf("Error updating pv.Status: %+v", err)
		}
	}

	if currentPhase != nextPhase {
		glog.V(5).Infof("PersistentVolume[%s] changing phase from %s to %s\n", pv.Name, currentPhase, nextPhase)
		pv.Status.Phase = nextPhase
		_, err := recycler.client.UpdatePersistentVolumeStatus(pv)
		if err != nil {
			// Rollback to previous phase
			pv.Status.Phase = currentPhase
		}
	}

	return nil
}

func (recycler *PersistentVolumeRecycler) handleDelete(pv *api.PersistentVolume) error {
	glog.V(5).Infof("Deleting PersistentVolume[%s]\n", pv.Name)

	currentPhase := pv.Status.Phase
	nextPhase := currentPhase

	spec := volume.NewSpecFromPersistentVolume(pv, false)
	plugin, err := recycler.pluginMgr.FindDeletablePluginBySpec(spec)
	if err != nil {
		return fmt.Errorf("Could not find deletable volume plugin for spec: %+v", err)
	}
	deleter, err := plugin.NewDeleter(spec)
	if err != nil {
		return fmt.Errorf("could not obtain Deleter for spec: %+v", err)
	}
	// blocks until completion
	err = deleter.Delete()
	if err != nil {
		glog.Errorf("PersistentVolume[%s] failed deletion: %+v", pv.Name, err)
		pv.Status.Message = fmt.Sprintf("Deletion error: %s", err)
		nextPhase = api.VolumeFailed
	} else {
		glog.V(5).Infof("PersistentVolume[%s] successfully deleted through plugin\n", pv.Name)
		// after successful deletion through the plugin, we can also remove the PV from the cluster
		err = recycler.client.DeletePersistentVolume(pv)
		if err != nil {
			return fmt.Errorf("error deleting persistent volume: %+v", err)
		}
	}

	if currentPhase != nextPhase {
		glog.V(5).Infof("PersistentVolume[%s] changing phase from %s to %s\n", pv.Name, currentPhase, nextPhase)
		pv.Status.Phase = nextPhase
		_, err := recycler.client.UpdatePersistentVolumeStatus(pv)
		if err != nil {
			// Rollback to previous phase
			pv.Status.Phase = currentPhase
		}
	}

	return nil
}

// Run starts this recycler's control loops
func (recycler *PersistentVolumeRecycler) Run() {
	glog.V(5).Infof("Starting PersistentVolumeRecycler\n")
	if recycler.stopChannel == nil {
		recycler.stopChannel = make(chan struct{})
		go recycler.volumeController.Run(recycler.stopChannel)
	}
}

// Stop gracefully shuts down this binder
func (recycler *PersistentVolumeRecycler) Stop() {
	glog.V(5).Infof("Stopping PersistentVolumeRecycler\n")
	if recycler.stopChannel != nil {
		close(recycler.stopChannel)
		recycler.stopChannel = nil
	}
}

// recyclerClient abstracts access to PVs
type recyclerClient interface {
	GetPersistentVolume(name string) (*api.PersistentVolume, error)
	UpdatePersistentVolume(volume *api.PersistentVolume) (*api.PersistentVolume, error)
	DeletePersistentVolume(volume *api.PersistentVolume) error
	UpdatePersistentVolumeStatus(volume *api.PersistentVolume) (*api.PersistentVolume, error)
}

func NewRecyclerClient(c client.Interface) recyclerClient {
	return &realRecyclerClient{c}
}

type realRecyclerClient struct {
	client client.Interface
}

func (c *realRecyclerClient) GetPersistentVolume(name string) (*api.PersistentVolume, error) {
	return c.client.PersistentVolumes().Get(name)
}

func (c *realRecyclerClient) UpdatePersistentVolume(volume *api.PersistentVolume) (*api.PersistentVolume, error) {
	return c.client.PersistentVolumes().Update(volume)
}

func (c *realRecyclerClient) DeletePersistentVolume(volume *api.PersistentVolume) error {
	return c.client.PersistentVolumes().Delete(volume.Name)
}

func (c *realRecyclerClient) UpdatePersistentVolumeStatus(volume *api.PersistentVolume) (*api.PersistentVolume, error) {
	return c.client.PersistentVolumes().UpdateStatus(volume)
}

// PersistentVolumeRecycler is host to the volume plugins, but does not actually mount any volumes.
// Because no mounting is performed, most of the VolumeHost methods are not implemented.
func (f *PersistentVolumeRecycler) GetPluginDir(podUID string) string {
	return ""
}

func (f *PersistentVolumeRecycler) GetPodVolumeDir(podUID types.UID, pluginName, volumeName string) string {
	return ""
}

func (f *PersistentVolumeRecycler) GetPodPluginDir(podUID types.UID, pluginName string) string {
	return ""
}

func (f *PersistentVolumeRecycler) GetKubeClient() client.Interface {
	return f.kubeClient
}

func (f *PersistentVolumeRecycler) NewWrapperBuilder(spec *volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Builder, error) {
	return nil, fmt.Errorf("NewWrapperBuilder not supported by PVClaimBinder's VolumeHost implementation")
}

func (f *PersistentVolumeRecycler) NewWrapperCleaner(spec *volume.Spec, podUID types.UID) (volume.Cleaner, error) {
	return nil, fmt.Errorf("NewWrapperCleaner not supported by PVClaimBinder's VolumeHost implementation")
}

func (f *PersistentVolumeRecycler) GetCloudProvider() cloudprovider.Interface {
	return nil
}

func (f *PersistentVolumeRecycler) GetMounter() mount.Interface {
	return nil
}

func (f *PersistentVolumeRecycler) GetWriter() ioutil.Writer {
	return nil
}
