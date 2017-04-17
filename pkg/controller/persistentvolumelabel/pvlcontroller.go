/*
Copyright 2017 The Kubernetes Authors.

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

package cloud

import (
	"fmt"
	"sync"
	"time"

	"github.com/golang/glog"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"

	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	coreinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions/core/v1"
	clientretry "k8s.io/kubernetes/pkg/client/retry"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/pkg/controller"
	vol "k8s.io/kubernetes/pkg/volume"
)

var UpdatePVLabelsBackoff = wait.Backoff{
	Steps:    20,
	Duration: 50 * time.Millisecond,
	Jitter:   1.0,
}

// PersistentVolumeLabelController handles adding labels to persistent volumes when they are created
type PersistentVolumeLabelController struct {
	ebsVolumes       aws.Volumes
	gceCloudProvider *gce.GCECloud
	cloud            cloudprovider.Interface
	mutex            sync.Mutex
	pvSynced         cache.InformerSynced
	kubeClient       clientset.Interface

	syncHandler func(vol *v1.PersistentVolume) error

	// queue is where incoming work is placed to de-dup and to allow "easy" rate limited requeues on errors
	queue workqueue.RateLimitingInterface
}

// NewPersistentVolumeLabelController creates a PersistentVolumeLabelController object
func NewPersistentVolumeLabelController(
	pvInformer coreinformers.PersistentVolumeInformer,
	kubeClient clientset.Interface,
	cloud cloudprovider.Interface) *PersistentVolumeLabelController {
	pvlc := &PersistentVolumeLabelController{
		cloud:      cloud,
		kubeClient: kubeClient,
		pvSynced:   pvInformer.Informer().HasSynced,
		queue:      workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "pvLabels"),
	}
	pvlc.syncHandler = pvlc.AddLabels

	pvInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			glog.Info("New Pv Added")
			cast := obj.(*v1.PersistentVolume)
			pvlc.queue.Add(cast)
		},
	})

	return pvlc
}

// This controller adds labels to persistent volumes and sets them to
// an available state
func (pvlc *PersistentVolumeLabelController) Run(threadiness int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer pvlc.queue.ShutDown()

	glog.Infof("Starting PersistentVolumeLabelController")
	defer glog.Infof("Shutting down PersistentVolumeLabelController")

	// wait for your secondary caches to fill before starting your work
	if !controller.WaitForCacheSync("pvLabels", stopCh, pvlc.pvSynced) {
		return
	}

	// start up your worker threads based on threadiness.  Some controllers have multiple kinds of workers
	for i := 0; i < threadiness; i++ {
		// runWorker will loop until "something bad" happens.  The .Until will then rekick the worker
		// after one second
		go wait.Until(pvlc.runWorker, time.Second, stopCh)
	}

	// wait until we're told to stop
	<-stopCh
}

func (pvlc *PersistentVolumeLabelController) runWorker() {
	// hot loop until we're told to stop.  processNextWorkItem will automatically wait until there's work
	// available, so we don't worry about secondary waits
	for pvlc.processNextWorkItem() {
	}
}

// processNextWorkItem deals with one key off the queue.  It returns false when it's time to quit.
func (pvlc *PersistentVolumeLabelController) processNextWorkItem() bool {
	// pull the next work item from queue.  It should be a key we use to lookup something in a cache
	key, quit := pvlc.queue.Get()
	glog.Info("Pulled PV work from queue")
	if quit {
		return false
	}
	// you always have to indicate to the queue that you've completed a piece of work
	defer pvlc.queue.Done(key)

	// do your work on the key.  This method will contains your "do stuff" logic
	err := pvlc.syncHandler(key.(*v1.PersistentVolume))
	if err == nil {
		// if you had no error, tell the queue to stop tracking history for your key.  This will
		// reset things like failure counts for per-item rate limiting
		pvlc.queue.Forget(key)
		return true
	}

	// there was a failure so be sure to report it.  This method allows for pluggable error handling
	// which can be used for things like cluster-monitoring
	utilruntime.HandleError(fmt.Errorf("%v failed with : %v", key, err))

	// since we failed, we should requeue the item to work on later.  This method will add a backoff
	// to avoid hotlooping on particular items (they're probably still not going to work right away)
	// and overall controller protection (everything I've done is broken, this controller needs to
	// calm down or it can starve other useful work) cases.
	pvlc.queue.AddRateLimited(key)

	return true
}

// AddLabels adds appropriate labels to persistent volumes and sets the
// volume as available if successful.
func (pvlc *PersistentVolumeLabelController) AddLabels(volume *v1.PersistentVolume) error {
	glog.Info("Adding PV Labels")
	var volumeLabels map[string]string
	if volume.Spec.AWSElasticBlockStore != nil {
		labels, err := pvlc.findAWSEBSLabels(volume)
		if err != nil {
			return fmt.Errorf("error querying AWS EBS volume %s: %v", volume.Spec.AWSElasticBlockStore.VolumeID, err)
		} else {
			volumeLabels = labels
		}
	}
	if volume.Spec.GCEPersistentDisk != nil {
		labels, err := pvlc.findGCEPDLabels(volume)
		if err != nil {
			return fmt.Errorf("error querying GCE PD volume %s: %v", volume.Spec.GCEPersistentDisk.PDName, err)
		} else {
			volumeLabels = labels
		}
	}

	glog.Infof("Found %d labels: %v", len(volumeLabels), volumeLabels)
	if len(volumeLabels) != 0 {
		if volume.Labels == nil {
			volume.Labels = make(map[string]string)
		}
		for k, v := range volumeLabels {
			// We (silently) replace labels if they are provided.
			// This should be OK because they are in the kubernetes.io namespace
			// i.e. we own them
			glog.Infof("Adding label %s=%s", k, v)
			volume.Labels[k] = v
		}
		//	glog.Info("Pausing before setting volume available")
		//	time.Sleep(30 * time.Second)
		volume.Status.Phase = v1.VolumeAvailable
		return pvlc.updateVolume(volume)
	}

	return nil
}

func (pvlc *PersistentVolumeLabelController) findAWSEBSLabels(volume *v1.PersistentVolume) (map[string]string, error) {
	// Ignore any volumes that are being provisioned
	if volume.Spec.AWSElasticBlockStore.VolumeID == vol.ProvisionedVolumeName {
		return nil, nil
	}
	ebsVolumes, err := pvlc.getEBSVolumes()
	if err != nil {
		return nil, err
	}

	// TODO: GetVolumeLabels is actually a method on the Volumes interface
	// If that gets standardized we can refactor to reduce code duplication
	spec := aws.KubernetesVolumeID(volume.Spec.AWSElasticBlockStore.VolumeID)
	labels, err := ebsVolumes.GetVolumeLabels(spec)
	if err != nil {
		return nil, err
	}

	return labels, nil
}

// getEBSVolumes returns the AWS Volumes interface for ebs
func (pvlc *PersistentVolumeLabelController) getEBSVolumes() (aws.Volumes, error) {
	pvlc.mutex.Lock()
	defer pvlc.mutex.Unlock()

	if pvlc.ebsVolumes == nil {
		awsCloudProvider, ok := pvlc.cloud.(*aws.Cloud)
		if !ok {
			// GetCloudProvider has gone very wrong
			return nil, fmt.Errorf("error retrieving AWS cloud provider")
		}
		pvlc.ebsVolumes = awsCloudProvider
	}
	return pvlc.ebsVolumes, nil
}

func (pvlc *PersistentVolumeLabelController) findGCEPDLabels(volume *v1.PersistentVolume) (map[string]string, error) {
	// Ignore any volumes that are being provisioned
	if volume.Spec.GCEPersistentDisk.PDName == vol.ProvisionedVolumeName {
		return nil, nil
	}

	provider, err := pvlc.getGCECloudProvider()
	if err != nil {
		return nil, err
	}

	// If the zone is already labeled, honor the hint
	zone := volume.Labels[metav1.LabelZoneFailureDomain]

	labels, err := provider.GetAutoLabelsForPD(volume.Spec.GCEPersistentDisk.PDName, zone)
	if err != nil {
		return nil, err
	}

	return labels, nil
}

// getGCECloudProvider returns the GCE cloud provider, for use for querying volume labels
func (pvlc *PersistentVolumeLabelController) getGCECloudProvider() (*gce.GCECloud, error) {
	pvlc.mutex.Lock()
	defer pvlc.mutex.Unlock()

	if pvlc.gceCloudProvider == nil {
		gceCloudProvider, ok := pvlc.cloud.(*gce.GCECloud)
		if !ok {
			// GetCloudProvider has gone very wrong
			return nil, fmt.Errorf("error retrieving GCE cloud provider")
		}
		pvlc.gceCloudProvider = gceCloudProvider
	}
	return pvlc.gceCloudProvider, nil
}

func (pvlc *PersistentVolumeLabelController) updateVolume(volume *v1.PersistentVolume) error {
	glog.V(4).Infof("updating PersistentVolume %s", volume.Name)
	err := clientretry.RetryOnConflict(UpdatePVLabelsBackoff, func() error {
		curVol, err := pvlc.kubeClient.Core().PersistentVolumes().Get(volume.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		curVol.Labels = volume.Labels
		curVol.Status.Phase = volume.Status.Phase

		_, err = pvlc.kubeClient.Core().PersistentVolumes().Update(curVol)
		return err
	})
	if err == nil {
		glog.V(4).Infof("updated PersistentVolume %s", volume.Name)
	}
	return err
}
