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

package volumeclaimbinder

import (
	"fmt"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/controller/framework"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/golang/glog"
)

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

		// handleRecycle blocks until completion
		// TODO: allow parallel recycling operations to increase throughput
		// TODO implement handleDelete in a separate PR w/ cloud volumes
		switch pv.Spec.PersistentVolumeReclaimPolicy {
		case api.PersistentVolumeReclaimRecycle:
			err = recycler.handleRecycle(pv)
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

	spec := volume.NewSpecFromPersistentVolume(pv)
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
		glog.Errorf("PersistentVolume[%s] failed recycling: %+v", err)
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

func (f *PersistentVolumeRecycler) NewWrapperBuilder(spec *volume.Spec, pod *api.Pod, opts volume.VolumeOptions, mounter mount.Interface) (volume.Builder, error) {
	return nil, fmt.Errorf("NewWrapperBuilder not supported by PVClaimBinder's VolumeHost implementation")
}

func (f *PersistentVolumeRecycler) NewWrapperCleaner(spec *volume.Spec, podUID types.UID, mounter mount.Interface) (volume.Cleaner, error) {
	return nil, fmt.Errorf("NewWrapperCleaner not supported by PVClaimBinder's VolumeHost implementation")
}
