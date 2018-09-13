/*
Copyright 2016 The Kubernetes Authors.

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

package images

import (
	"fmt"
	"strings"
	"sync"

	dockerref "github.com/docker/distribution/reference"
	"github.com/golang/groupcache/lru"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/klog"

	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/runtime/v1alpha2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/util/parsers"
	"k8s.io/utils/clock"
)

const (
	// max cache size of identicalEventSpamFilter
	maxLruCacheEntries = 4096

	// by default, allow imageManager to send 1 burst identical event about an object
	// and control the refill rate to 1 new event every 5 minutes
	// this helps avoid sending lots of identical events to kuberuntime manager's
	// EventSourceObjectSpamFilter and consuming EventSourceObjectSpamFilter's token too fast
	defaultSpamBurst = 1
	// same as EventSourceObjectSpamFilter's qps
	defaultSpamQPS = 1. / 300.
)

// imageManager provides the functionalities for image pulling.
type imageManager struct {
	recorder     record.EventRecorder
	imageService kubecontainer.ImageService
	backOff      *flowcontrol.Backoff
	// It will check the presence of the image, and report the 'image pulling', image pulled' events correspondingly.
	puller imagePuller
	// throttles the amount of identical events
	identicalEventSpamFilter *IdenticalEventSpamFilter
}

var _ ImageManager = &imageManager{}

// NewImageManager instantiates a new ImageManager object.
func NewImageManager(recorder record.EventRecorder, imageService kubecontainer.ImageService, imageBackOff *flowcontrol.Backoff, serialized bool, qps float32, burst int) ImageManager {
	imageService = throttleImagePulling(imageService, qps, burst)

	var puller imagePuller
	if serialized {
		puller = newSerialImagePuller(imageService)
	} else {
		puller = newParallelImagePuller(imageService)
	}
	return &imageManager{
		recorder:                 recorder,
		imageService:             imageService,
		backOff:                  imageBackOff,
		puller:                   puller,
		identicalEventSpamFilter: defaultIdenticalEventSpamFilter(clock.RealClock{}),
	}
}

// shouldPullImage returns whether we should pull an image according to
// the presence and pull policy of the image.
func shouldPullImage(container *v1.Container, imagePresent bool) bool {
	if container.ImagePullPolicy == v1.PullNever {
		return false
	}

	if container.ImagePullPolicy == v1.PullAlways ||
		(container.ImagePullPolicy == v1.PullIfNotPresent && (!imagePresent)) {
		return true
	}

	return false
}

// records an event using ref, event msg.  log to glog using prefix, msg, logFn
func (m *imageManager) logIt(ref *v1.ObjectReference, eventtype, event, prefix, msg string, logFn func(args ...interface{})) {
	if ref != nil {
		if m.identicalEventSpamFilter.Filter(ref, eventtype, event, msg) {
			glog.V(4).Infof("Skipping to record identical event for %v %v(%v), event type:%v, reason:%v, message:%v",
				ref.Kind, ref.Name, ref.UID, eventtype, event, msg)
			return
		}
		m.recorder.Event(ref, eventtype, event, msg)
	} else {
		logFn(fmt.Sprint(prefix, " ", msg))
	}
}

// EnsureImageExists pulls the image for the specified pod and container, and returns
// (imageRef, error message, error).
func (m *imageManager) EnsureImageExists(pod *v1.Pod, container *v1.Container, pullSecrets []v1.Secret, podSandboxConfig *runtimeapi.PodSandboxConfig) (string, string, error) {
	logPrefix := fmt.Sprintf("%s/%s", pod.Name, container.Image)
	ref, err := kubecontainer.GenerateContainerRef(pod, container)
	if err != nil {
		klog.Errorf("Couldn't make a ref to pod %v, container %v: '%v'", pod.Name, container.Name, err)
	}

	// If the image contains no tag or digest, a default tag should be applied.
	image, err := applyDefaultImageTag(container.Image)
	if err != nil {
		msg := fmt.Sprintf("Failed to apply default image tag %q: %v", container.Image, err)
		m.logIt(ref, v1.EventTypeWarning, events.FailedToInspectImage, logPrefix, msg, klog.Warning)
		return "", msg, ErrInvalidImageName
	}

	spec := kubecontainer.ImageSpec{Image: image}
	imageRef, err := m.imageService.GetImageRef(spec)
	if err != nil {
		msg := fmt.Sprintf("Failed to inspect image %q: %v", container.Image, err)
		m.logIt(ref, v1.EventTypeWarning, events.FailedToInspectImage, logPrefix, msg, klog.Warning)
		return "", msg, ErrImageInspect
	}

	present := imageRef != ""
	if !shouldPullImage(container, present) {
		if present {
			msg := fmt.Sprintf("Container image %q already present on machine", container.Image)
			m.logIt(ref, v1.EventTypeNormal, events.PulledImage, logPrefix, msg, klog.Info)
			return imageRef, "", nil
		}
		msg := fmt.Sprintf("Container image %q is not present with pull policy of Never", container.Image)
		m.logIt(ref, v1.EventTypeWarning, events.ErrImageNeverPullPolicy, logPrefix, msg, klog.Warning)
		return "", msg, ErrImageNeverPull
	}

	backOffKey := fmt.Sprintf("%s_%s", pod.UID, container.Image)
	if m.backOff.IsInBackOffSinceUpdate(backOffKey, m.backOff.Clock.Now()) {
		msg := fmt.Sprintf("Back-off pulling image %q", container.Image)
		m.logIt(ref, v1.EventTypeNormal, events.BackOffPullImage, logPrefix, msg, klog.Info)
		return "", msg, ErrImagePullBackOff
	}
	m.logIt(ref, v1.EventTypeNormal, events.PullingImage, logPrefix, fmt.Sprintf("pulling image %q", container.Image), klog.Info)
	pullChan := make(chan pullResult)
	m.puller.pullImage(spec, pullSecrets, pullChan, podSandboxConfig)
	imagePullResult := <-pullChan
	if imagePullResult.err != nil {
		m.logIt(ref, v1.EventTypeWarning, events.FailedToPullImage, logPrefix, fmt.Sprintf("Failed to pull image %q: %v", container.Image, imagePullResult.err), klog.Warning)
		m.backOff.Next(backOffKey, m.backOff.Clock.Now())
		if imagePullResult.err == ErrRegistryUnavailable {
			msg := fmt.Sprintf("image pull failed for %s because the registry is unavailable.", container.Image)
			return "", msg, imagePullResult.err
		}

		return "", imagePullResult.err.Error(), ErrImagePull
	}
	m.logIt(ref, v1.EventTypeNormal, events.PulledImage, logPrefix, fmt.Sprintf("Successfully pulled image %q", container.Image), klog.Info)
	m.backOff.GC()
	return imagePullResult.imageRef, "", nil
}

// applyDefaultImageTag parses a docker image string, if it doesn't contain any tag or digest,
// a default tag will be applied.
func applyDefaultImageTag(image string) (string, error) {
	named, err := dockerref.ParseNormalizedNamed(image)
	if err != nil {
		return "", fmt.Errorf("couldn't parse image reference %q: %v", image, err)
	}
	_, isTagged := named.(dockerref.Tagged)
	_, isDigested := named.(dockerref.Digested)
	if !isTagged && !isDigested {
		// we just concatenate the image name with the default tag here instead
		// of using dockerref.WithTag(named, ...) because that would cause the
		// image to be fully qualified as docker.io/$name if it's a short name
		// (e.g. just busybox). We don't want that to happen to keep the CRI
		// agnostic wrt image names and default hostnames.
		image = image + ":" + parsers.DefaultImageTag
	}
	return image, nil
}

// spamRecord holds data used to perform spam filtering decisions.
type spamRecord struct {
	// rateLimiter controls the rate of events about this object
	rateLimiter flowcontrol.RateLimiter
}

// IdenticalEventSpamFilter is responsible for throttling
// the amount of identical events.
type IdenticalEventSpamFilter struct {
	sync.RWMutex

	// the cache that manages last synced state
	cache *lru.Cache

	// burst is the amount of identical events we allow per object
	burst int

	// qps is the refill rate of the token bucket in queries per second
	qps float32

	// clock is used to allow for testing over a time interval
	clock clock.Clock
}

// defaultIdenticalEventSpamFilter allows defaultSpamBurst identical events about an object with defaultSpamQPS refill.
func defaultIdenticalEventSpamFilter(clock clock.Clock) *IdenticalEventSpamFilter {
	return &IdenticalEventSpamFilter{
		cache: lru.New(maxLruCacheEntries),
		burst: defaultSpamBurst,
		qps:   defaultSpamQPS,
		clock: clock,
	}
}

// Filter controls that identical events are not exceeding the allowed rate.
func (f *IdenticalEventSpamFilter) Filter(object *v1.ObjectReference, eventtype, reason, message string) bool {
	var record spamRecord

	// controls our cached information about this event
	eventKey := getSpamKey(object, eventtype, reason, message)

	f.Lock()
	defer f.Unlock()
	value, found := f.cache.Get(eventKey)
	if found {
		record = value.(spamRecord)
	}

	// verify we have a rate limiter for this record
	if record.rateLimiter == nil {
		record.rateLimiter = flowcontrol.NewTokenBucketRateLimiterWithClock(f.qps, f.burst, f.clock)
	}

	// ensure we have available rate
	filter := !record.rateLimiter.TryAccept()

	// update the cache
	f.cache.Add(eventKey, record)

	return filter
}

// getSpamKey builds unique event key based on involved object and message itself
func getSpamKey(object *v1.ObjectReference, eventtype, reason, message string) string {
	return strings.Join([]string{
		string(object.UID),
		eventtype,
		reason,
		message,
	},
		"-")
}
