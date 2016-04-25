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
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	ioutil "k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/metrics"
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
	kubeClient       clientset.Interface
	pluginMgr        volume.VolumePluginMgr
	cloud            cloudprovider.Interface
	maximumRetry     int
	syncPeriod       time.Duration
	// Local cache of failed recycle / delete operations. Map volume.Name -> status of the volume.
	// Only PVs in Released state have an entry here.
	releasedVolumes map[string]releasedVolumeStatus
}

// releasedVolumeStatus holds state of failed delete/recycle operation on a
// volume. The controller re-tries the operation several times and it stores
// retry count + timestamp of the last attempt here.
type releasedVolumeStatus struct {
	// How many recycle/delete operations failed.
	retryCount int
	// Timestamp of the last attempt.
	lastAttempt time.Time
}

// NewPersistentVolumeRecycler creates a new PersistentVolumeRecycler
func NewPersistentVolumeRecycler(kubeClient clientset.Interface, syncPeriod time.Duration, maximumRetry int, plugins []volume.VolumePlugin, cloud cloudprovider.Interface) (*PersistentVolumeRecycler, error) {
	recyclerClient := NewRecyclerClient(kubeClient)
	if kubeClient != nil && kubeClient.Core().GetRESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("pv_recycler_controller", kubeClient.Core().GetRESTClient().GetRateLimiter())
	}
	recycler := &PersistentVolumeRecycler{
		client:          recyclerClient,
		kubeClient:      kubeClient,
		cloud:           cloud,
		maximumRetry:    maximumRetry,
		syncPeriod:      syncPeriod,
		releasedVolumes: make(map[string]releasedVolumeStatus),
	}

	if err := recycler.pluginMgr.InitPlugins(plugins, recycler); err != nil {
		return nil, fmt.Errorf("Could not initialize volume plugins for PVClaimBinder: %+v", err)
	}

	_, volumeController := framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return kubeClient.Core().PersistentVolumes().List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return kubeClient.Core().PersistentVolumes().Watch(options)
			},
		},
		&api.PersistentVolume{},
		syncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				pv, ok := obj.(*api.PersistentVolume)
				if !ok {
					glog.Errorf("Error casting object to PersistentVolume: %v", obj)
					return
				}
				recycler.reclaimVolume(pv)
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				pv, ok := newObj.(*api.PersistentVolume)
				if !ok {
					glog.Errorf("Error casting object to PersistentVolume: %v", newObj)
					return
				}
				recycler.reclaimVolume(pv)
			},
			DeleteFunc: func(obj interface{}) {
				pv, ok := obj.(*api.PersistentVolume)
				if !ok {
					glog.Errorf("Error casting object to PersistentVolume: %v", obj)
					return
				}
				recycler.reclaimVolume(pv)
				recycler.removeReleasedVolume(pv)
			},
		},
	)

	recycler.volumeController = volumeController
	return recycler, nil
}

// shouldRecycle checks a volume and returns nil, if the volume should be
// recycled right now. Otherwise it returns an error with reason why it should
// not be recycled.
func (recycler *PersistentVolumeRecycler) shouldRecycle(pv *api.PersistentVolume) error {
	if pv.Spec.ClaimRef == nil {
		return fmt.Errorf("Volume does not have a reference to claim")
	}
	if pv.Status.Phase != api.VolumeReleased {
		return fmt.Errorf("The volume is not in 'Released' phase")
	}

	// The volume is Released, should we retry recycling?
	status, found := recycler.releasedVolumes[pv.Name]
	if !found {
		// We don't know anything about this volume. The controller has been
		// restarted or the volume has been marked as Released by another
		// controller. Recycle/delete this volume as if it was just Released.
		glog.V(5).Infof("PersistentVolume[%s] not found in local cache, recycling", pv.Name)
		return nil
	}

	// Check the timestamp
	expectedRetry := status.lastAttempt.Add(recycler.syncPeriod)
	if time.Now().After(expectedRetry) {
		glog.V(5).Infof("PersistentVolume[%s] retrying recycle after timeout", pv.Name)
		return nil
	}
	// It's too early
	glog.V(5).Infof("PersistentVolume[%s] skipping recycle, it's too early: now: %v, next retry: %v", pv.Name, time.Now(), expectedRetry)
	return fmt.Errorf("Too early after previous failure")
}

func (recycler *PersistentVolumeRecycler) reclaimVolume(pv *api.PersistentVolume) error {
	glog.V(5).Infof("Recycler: checking PersistentVolume[%s]\n", pv.Name)
	// Always load the latest version of the volume
	newPV, err := recycler.client.GetPersistentVolume(pv.Name)
	if err != nil {
		return fmt.Errorf("Could not find PersistentVolume %s", pv.Name)
	}
	pv = newPV

	err = recycler.shouldRecycle(pv)
	if err == nil {
		glog.V(5).Infof("Reclaiming PersistentVolume[%s]\n", pv.Name)

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
		return nil
	}
	glog.V(3).Infof("PersistentVolume[%s] phase %s - skipping: %v", pv.Name, pv.Status.Phase, err)
	return nil
}

// handleReleaseFailure evaluates a failed Recycle/Delete operation, updates
// internal controller state with new nr. of attempts and timestamp of the last
// attempt. Based on the number of failures it returns the next state of the
// volume (Released / Failed).
func (recycler *PersistentVolumeRecycler) handleReleaseFailure(pv *api.PersistentVolume) api.PersistentVolumePhase {
	status, found := recycler.releasedVolumes[pv.Name]
	if !found {
		// First failure, set retryCount to 0 (will be inceremented few lines below)
		status = releasedVolumeStatus{}
	}
	status.retryCount += 1

	if status.retryCount > recycler.maximumRetry {
		// This was the last attempt. Remove any internal state and mark the
		// volume as Failed.
		glog.V(3).Infof("PersistentVolume[%s] failed %d times - marking Failed", pv.Name, status.retryCount)
		recycler.removeReleasedVolume(pv)
		return api.VolumeFailed
	}

	status.lastAttempt = time.Now()
	recycler.releasedVolumes[pv.Name] = status
	return api.VolumeReleased
}

func (recycler *PersistentVolumeRecycler) removeReleasedVolume(pv *api.PersistentVolume) {
	delete(recycler.releasedVolumes, pv.Name)
}

func (recycler *PersistentVolumeRecycler) handleRecycle(pv *api.PersistentVolume) error {
	glog.V(5).Infof("Recycling PersistentVolume[%s]\n", pv.Name)

	currentPhase := pv.Status.Phase
	nextPhase := currentPhase

	spec := volume.NewSpecFromPersistentVolume(pv, false)
	plugin, err := recycler.pluginMgr.FindRecyclablePluginBySpec(spec)
	if err != nil {
		nextPhase = api.VolumeFailed
		pv.Status.Message = fmt.Sprintf("%v", err)
	}

	// an error above means a suitable plugin for this volume was not found.
	// we don't need to attempt recycling when plugin is nil, but we do need to persist the next/failed phase
	// of the volume so that subsequent syncs won't attempt recycling through this handler func.
	if plugin != nil {
		volRecycler, err := plugin.NewRecycler(spec)
		if err != nil {
			return fmt.Errorf("Could not obtain Recycler for spec: %#v  error: %v", spec, err)
		}
		// blocks until completion
		if err := volRecycler.Recycle(); err != nil {
			glog.Errorf("PersistentVolume[%s] failed recycling: %+v", pv.Name, err)
			pv.Status.Message = fmt.Sprintf("Recycling error: %s", err)
			nextPhase = recycler.handleReleaseFailure(pv)
		} else {
			glog.V(5).Infof("PersistentVolume[%s] successfully recycled\n", pv.Name)
			// The volume has been recycled. Remove any internal state to make
			// any subsequent bind+recycle cycle working.
			recycler.removeReleasedVolume(pv)
			nextPhase = api.VolumePending
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
		nextPhase = api.VolumeFailed
		pv.Status.Message = fmt.Sprintf("%v", err)
	}

	// an error above means a suitable plugin for this volume was not found.
	// we don't need to attempt deleting when plugin is nil, but we do need to persist the next/failed phase
	// of the volume so that subsequent syncs won't attempt deletion through this handler func.
	if plugin != nil {
		deleter, err := plugin.NewDeleter(spec)
		if err != nil {
			return fmt.Errorf("Could not obtain Deleter for spec: %#v  error: %v", spec, err)
		}
		// blocks until completion
		err = deleter.Delete()
		if err != nil {
			glog.Errorf("PersistentVolume[%s] failed deletion: %+v", pv.Name, err)
			pv.Status.Message = fmt.Sprintf("Deletion error: %s", err)
			nextPhase = recycler.handleReleaseFailure(pv)
		} else {
			glog.V(5).Infof("PersistentVolume[%s] successfully deleted through plugin\n", pv.Name)
			recycler.removeReleasedVolume(pv)
			// after successful deletion through the plugin, we can also remove the PV from the cluster
			if err := recycler.client.DeletePersistentVolume(pv); err != nil {
				return fmt.Errorf("error deleting persistent volume: %+v", err)
			}
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

func NewRecyclerClient(c clientset.Interface) recyclerClient {
	return &realRecyclerClient{c}
}

type realRecyclerClient struct {
	client clientset.Interface
}

func (c *realRecyclerClient) GetPersistentVolume(name string) (*api.PersistentVolume, error) {
	return c.client.Core().PersistentVolumes().Get(name)
}

func (c *realRecyclerClient) UpdatePersistentVolume(volume *api.PersistentVolume) (*api.PersistentVolume, error) {
	return c.client.Core().PersistentVolumes().Update(volume)
}

func (c *realRecyclerClient) DeletePersistentVolume(volume *api.PersistentVolume) error {
	return c.client.Core().PersistentVolumes().Delete(volume.Name, nil)
}

func (c *realRecyclerClient) UpdatePersistentVolumeStatus(volume *api.PersistentVolume) (*api.PersistentVolume, error) {
	return c.client.Core().PersistentVolumes().UpdateStatus(volume)
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

func (f *PersistentVolumeRecycler) GetKubeClient() clientset.Interface {
	return f.kubeClient
}

func (f *PersistentVolumeRecycler) NewWrapperMounter(volName string, spec volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Mounter, error) {
	return nil, fmt.Errorf("NewWrapperMounter not supported by PVClaimBinder's VolumeHost implementation")
}

func (f *PersistentVolumeRecycler) NewWrapperUnmounter(volName string, spec volume.Spec, podUID types.UID) (volume.Unmounter, error) {
	return nil, fmt.Errorf("NewWrapperUnmounter not supported by PVClaimBinder's VolumeHost implementation")
}

func (f *PersistentVolumeRecycler) GetCloudProvider() cloudprovider.Interface {
	return f.cloud
}

func (f *PersistentVolumeRecycler) GetMounter() mount.Interface {
	return nil
}

func (f *PersistentVolumeRecycler) GetWriter() ioutil.Writer {
	return nil
}

func (f *PersistentVolumeRecycler) GetHostName() string {
	return ""
}
