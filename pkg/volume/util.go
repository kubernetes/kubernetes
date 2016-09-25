/*
Copyright 2014 The Kubernetes Authors.

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

package volume

import (
	"fmt"
	"reflect"

	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/watch"

	"hash/fnv"
	"math/rand"
	"strconv"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/util/sets"
)

type RecycleEventRecorder func(eventtype, message string)

// RecycleVolumeByWatchingPodUntilCompletion is intended for use with volume
// Recyclers. This function will save the given Pod to the API and watch it
// until it completes, fails, or the pod's ActiveDeadlineSeconds is exceeded,
// whichever comes first. An attempt to delete a recycler pod is always
// attempted before returning.
//
// In case there is a pod with the same namespace+name already running, this
// function assumes it's an older instance of the recycler pod and watches
// this old pod instead of starting a new one.
//
//  pod - the pod designed by a volume plugin to recycle the volume. pod.Name
//        will be overwritten with unique name based on PV.Name.
//	client - kube client for API operations.
func RecycleVolumeByWatchingPodUntilCompletion(pvName string, pod *api.Pod, kubeClient clientset.Interface, recorder RecycleEventRecorder) error {
	return internalRecycleVolumeByWatchingPodUntilCompletion(pvName, pod, newRecyclerClient(kubeClient, recorder))
}

// same as above func comments, except 'recyclerClient' is a narrower pod API
// interface to ease testing
func internalRecycleVolumeByWatchingPodUntilCompletion(pvName string, pod *api.Pod, recyclerClient recyclerClient) error {
	glog.V(5).Infof("creating recycler pod for volume %s\n", pod.Name)

	// Generate unique name for the recycler pod - we need to get "already
	// exists" error when a previous controller has already started recycling
	// the volume. Here we assume that pv.Name is already unique.
	pod.Name = "recycler-for-" + pvName
	pod.GenerateName = ""

	stopChannel := make(chan struct{})
	defer close(stopChannel)
	podCh, err := recyclerClient.WatchPod(pod.Name, pod.Namespace, stopChannel)
	if err != nil {
		glog.V(4).Infof("cannot start watcher for pod %s/%s: %v", pod.Namespace, pod.Name, err)
		return err
	}

	// Start the pod
	_, err = recyclerClient.CreatePod(pod)
	if err != nil {
		if errors.IsAlreadyExists(err) {
			glog.V(5).Infof("old recycler pod %q found for volume", pod.Name)
		} else {
			return fmt.Errorf("unexpected error creating recycler pod:  %+v\n", err)
		}
	}
	defer recyclerClient.DeletePod(pod.Name, pod.Namespace)

	// Now only the old pod or the new pod run. Watch it until it finishes
	// and send all events on the pod to the PV
	for {
		event := <-podCh
		switch event.Object.(type) {
		case *api.Pod:
			// POD changed
			pod := event.Object.(*api.Pod)
			glog.V(4).Infof("recycler pod update received: %s %s/%s %s", event.Type, pod.Namespace, pod.Name, pod.Status.Phase)
			switch event.Type {
			case watch.Added, watch.Modified:
				if pod.Status.Phase == api.PodSucceeded {
					// Recycle succeeded.
					return nil
				}
				if pod.Status.Phase == api.PodFailed {
					if pod.Status.Message != "" {
						return fmt.Errorf(pod.Status.Message)
					} else {
						return fmt.Errorf("pod failed, pod.Status.Message unknown.")
					}
				}

			case watch.Deleted:
				return fmt.Errorf("recycler pod was deleted")

			case watch.Error:
				return fmt.Errorf("recycler pod watcher failed")
			}

		case *api.Event:
			// Event received
			podEvent := event.Object.(*api.Event)
			glog.V(4).Infof("recycler event received: %s %s/%s %s/%s %s", event.Type, podEvent.Namespace, podEvent.Name, podEvent.InvolvedObject.Namespace, podEvent.InvolvedObject.Name, podEvent.Message)
			if event.Type == watch.Added {
				recyclerClient.Event(podEvent.Type, podEvent.Message)
			}
		}
	}
}

// recyclerClient abstracts access to a Pod by providing a narrower interface.
// This makes it easier to mock a client for testing.
type recyclerClient interface {
	CreatePod(pod *api.Pod) (*api.Pod, error)
	GetPod(name, namespace string) (*api.Pod, error)
	DeletePod(name, namespace string) error
	// WatchPod returns a ListWatch for watching a pod.  The stopChannel is used
	// to close the reflector backing the watch.  The caller is responsible for
	// derring a close on the channel to stop the reflector.
	WatchPod(name, namespace string, stopChannel chan struct{}) (<-chan watch.Event, error)
	// Event sends an event to the volume that is being recycled.
	Event(eventtype, message string)
}

func newRecyclerClient(client clientset.Interface, recorder RecycleEventRecorder) recyclerClient {
	return &realRecyclerClient{
		client,
		recorder,
	}
}

type realRecyclerClient struct {
	client   clientset.Interface
	recorder RecycleEventRecorder
}

func (c *realRecyclerClient) CreatePod(pod *api.Pod) (*api.Pod, error) {
	return c.client.Core().Pods(pod.Namespace).Create(pod)
}

func (c *realRecyclerClient) GetPod(name, namespace string) (*api.Pod, error) {
	return c.client.Core().Pods(namespace).Get(name)
}

func (c *realRecyclerClient) DeletePod(name, namespace string) error {
	return c.client.Core().Pods(namespace).Delete(name, nil)
}

func (c *realRecyclerClient) Event(eventtype, message string) {
	c.recorder(eventtype, message)
}

func (c *realRecyclerClient) WatchPod(name, namespace string, stopChannel chan struct{}) (<-chan watch.Event, error) {
	podSelector, _ := fields.ParseSelector("metadata.name=" + name)
	options := api.ListOptions{
		FieldSelector: podSelector,
		Watch:         true,
	}

	podWatch, err := c.client.Core().Pods(namespace).Watch(options)
	if err != nil {
		return nil, err
	}

	eventSelector, _ := fields.ParseSelector("involvedObject.name=" + name)
	eventWatch, err := c.client.Core().Events(namespace).Watch(api.ListOptions{
		FieldSelector: eventSelector,
		Watch:         true,
	})
	if err != nil {
		podWatch.Stop()
		return nil, err
	}

	eventCh := make(chan watch.Event, 0)

	go func() {
		defer eventWatch.Stop()
		defer podWatch.Stop()
		defer close(eventCh)

		for {
			select {
			case _ = <-stopChannel:
				return

			case podEvent, ok := <-podWatch.ResultChan():
				if !ok {
					return
				}
				eventCh <- podEvent

			case eventEvent, ok := <-eventWatch.ResultChan():
				if !ok {
					return
				}
				eventCh <- eventEvent
			}
		}
	}()

	return eventCh, nil
}

// CalculateTimeoutForVolume calculates time for a Recycler pod to complete a
// recycle operation. The calculation and return value is either the
// minimumTimeout or the timeoutIncrement per Gi of storage size, whichever is
// greater.
func CalculateTimeoutForVolume(minimumTimeout, timeoutIncrement int, pv *api.PersistentVolume) int64 {
	giQty := resource.MustParse("1Gi")
	pvQty := pv.Spec.Capacity[api.ResourceStorage]
	giSize := giQty.Value()
	pvSize := pvQty.Value()
	timeout := (pvSize / giSize) * int64(timeoutIncrement)
	if timeout < int64(minimumTimeout) {
		return int64(minimumTimeout)
	} else {
		return timeout
	}
}

// RoundUpSize calculates how many allocation units are needed to accommodate
// a volume of given size. E.g. when user wants 1500MiB volume, while AWS EBS
// allocates volumes in gibibyte-sized chunks,
// RoundUpSize(1500 * 1024*1024, 1024*1024*1024) returns '2'
// (2 GiB is the smallest allocatable volume that can hold 1500MiB)
func RoundUpSize(volumeSizeBytes int64, allocationUnitBytes int64) int64 {
	return (volumeSizeBytes + allocationUnitBytes - 1) / allocationUnitBytes
}

// GenerateVolumeName returns a PV name with clusterName prefix. The function
// should be used to generate a name of GCE PD or Cinder volume. It basically
// adds "<clusterName>-dynamic-" before the PV name, making sure the resulting
// string fits given length and cuts "dynamic" if not.
func GenerateVolumeName(clusterName, pvName string, maxLength int) string {
	prefix := clusterName + "-dynamic"
	pvLen := len(pvName)

	// cut the "<clusterName>-dynamic" to fit full pvName into maxLength
	// +1 for the '-' dash
	if pvLen+1+len(prefix) > maxLength {
		prefix = prefix[:maxLength-pvLen-1]
	}
	return prefix + "-" + pvName
}

// Check if the path from the mounter is empty.
func GetPath(mounter Mounter) (string, error) {
	path := mounter.GetPath()
	if path == "" {
		return "", fmt.Errorf("Path is empty %s", reflect.TypeOf(mounter).String())
	}
	return path, nil
}

// ChooseZone implements our heuristics for choosing a zone for volume creation based on the volume name
// Volumes are generally round-robin-ed across all active zones, using the hash of the PVC Name.
// However, if the PVCName ends with `-<integer>`, we will hash the prefix, and then add the integer to the hash.
// This means that a PetSet's volumes (`claimname-petsetname-id`) will spread across available zones,
// assuming the id values are consecutive.
func ChooseZoneForVolume(zones sets.String, pvcName string) string {
	// We create the volume in a zone determined by the name
	// Eventually the scheduler will coordinate placement into an available zone
	var hash uint32
	var index uint32

	if pvcName == "" {
		// We should always be called with a name; this shouldn't happen
		glog.Warningf("No name defined during volume create; choosing random zone")

		hash = rand.Uint32()
	} else {
		hashString := pvcName

		// Heuristic to make sure that volumes in a PetSet are spread across zones
		// PetSet PVCs are (currently) named ClaimName-PetSetName-Id,
		// where Id is an integer index
		lastDash := strings.LastIndexByte(pvcName, '-')
		if lastDash != -1 {
			petIDString := pvcName[lastDash+1:]
			petID, err := strconv.ParseUint(petIDString, 10, 32)
			if err == nil {
				// Offset by the pet id, so we round-robin across zones
				index = uint32(petID)
				// We still hash the volume name, but only the base
				hashString = pvcName[:lastDash]
				glog.V(2).Infof("Detected PetSet-style volume name %q; index=%d", pvcName, index)
			}
		}

		// We hash the (base) volume name, so we don't bias towards the first N zones
		h := fnv.New32()
		h.Write([]byte(hashString))
		hash = h.Sum32()
	}

	// Zones.List returns zones in a consistent order (sorted)
	// We do have a potential failure case where volumes will not be properly spread,
	// if the set of zones changes during PetSet volume creation.  However, this is
	// probably relatively unlikely because we expect the set of zones to be essentially
	// static for clusters.
	// Hopefully we can address this problem if/when we do full scheduler integration of
	// PVC placement (which could also e.g. avoid putting volumes in overloaded or
	// unhealthy zones)
	zoneSlice := zones.List()
	zone := zoneSlice[(hash+index)%uint32(len(zoneSlice))]

	glog.V(2).Infof("Creating volume for PVC %q; chose zone=%q from zones=%q", pvcName, zone, zoneSlice)
	return zone
}
