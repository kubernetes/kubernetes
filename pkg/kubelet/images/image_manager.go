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
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/klog/v2"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	crierrors "k8s.io/cri-api/pkg/errors"
	"k8s.io/kubernetes/pkg/credentialprovider"
	credentialprovidersecrets "k8s.io/kubernetes/pkg/credentialprovider/secrets"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/util/parsers"
)

type ImagePodPullingTimeRecorder interface {
	RecordImageStartedPulling(podUID types.UID)
	RecordImageFinishedPulling(podUID types.UID)
}

// imageManager provides the functionalities for image pulling.
type imageManager struct {
	recorder     record.EventRecorder
	imageService kubecontainer.ImageService
	backOff      *flowcontrol.Backoff
	// It will check the presence of the image, and report the 'image pulling', image pulled' events correspondingly.
	puller imagePuller

	podPullingTimeRecorder ImagePodPullingTimeRecorder

	keyring *credentialprovider.DockerKeyring
	// ensureSecretPulledImages - map of imageref (image digest) to successful secret pulled image details
	ensureSecretPulledImages     map[string]*imagePullInfo
	pullImageSecretRecheckPeriod metav1.Duration
	lock                         sync.RWMutex
}

var _ ImageManager = &imageManager{}

// NewImageManager instantiates a new ImageManager object.
func NewImageManager(recorder record.EventRecorder, imageService kubecontainer.ImageService, imageBackOff *flowcontrol.Backoff,
	serialized bool, maxParallelImagePulls *int32, qps float32, burst int, podPullingTimeRecorder ImagePodPullingTimeRecorder,
	keyring *credentialprovider.DockerKeyring, pullImageSecretRecheckPeriod metav1.Duration) ImageManager {
	imageService = throttleImagePulling(imageService, qps, burst)

	var puller imagePuller
	if serialized {
		puller = newSerialImagePuller(imageService)
	} else {
		puller = newParallelImagePuller(imageService, maxParallelImagePulls)
	}

	return &imageManager{
		recorder:                     recorder,
		imageService:                 imageService,
		backOff:                      imageBackOff,
		puller:                       puller,
		podPullingTimeRecorder:       podPullingTimeRecorder,
		keyring:                      keyring,
		ensureSecretPulledImages:     make(map[string]*imagePullInfo),
		pullImageSecretRecheckPeriod: pullImageSecretRecheckPeriod,
	}
}

// shouldPullImage returns whether we should pull an image according to
// the presence and pull policy of the image.
func shouldPullImage(pullPolicy v1.PullPolicy, imagePresent, pulledBySecret, ensuredBySecret bool) bool {
	switch pullPolicy {
	case v1.PullNever:
		return false
	case v1.PullAlways:
		return true
	case v1.PullIfNotPresent:
		// pull if image doesn't exist
		if !imagePresent {
			return true
		}
		// if the imageRef has been pulled by a secret and Pull Policy is PullIfNotPresent
		// we need to ensure that the current pod's secrets map to an auth that has Already
		// pulled the image successfully. Otherwise pod B could use pod A's images
		// without auth. So in this case if pulledBySecret but not ensured by matching
		// secret auth for a pull again for the pod B scenario where the auth does not match
		if utilfeature.DefaultFeatureGate.Enabled((features.KubeletEnsureSecretPulledImages)) {
			if pulledBySecret && !ensuredBySecret {
				return true // noting here that old behaviour returns false in this case indicating the image should not be pulled
			}
		}
		return false
	}
	return false
}

// records an event using ref, event msg.  log to glog using prefix, msg, logFn
func (m *imageManager) logIt(objRef *v1.ObjectReference, eventtype, event, prefix, msg string, logFn func(args ...interface{})) {
	if objRef != nil {
		m.recorder.Event(objRef, eventtype, event, msg)
	} else {
		logFn(fmt.Sprint(prefix, " ", msg))
	}
}

type ensuredInfo struct {
	// true for ensured secret
	ensured bool
	// the secret should be verified again if current time is after the due date.
	// and the due date is `PullImageSecretRecheckPeriod` after last ensured date.
	// `PullImageSecretRecheckPeriod` is configurable in kubelet config.
	lastEnsuredDate time.Time
}

type imagePullInfo struct {
	// TODO: (mikebrow) time of last pull for this imageRef
	// TODO: (mikebrow) time of pull for each particular auth hash
	//       note @mrunalp makes a good point that we can utilize /apimachinery/pkg/util/sets/string.go here

	// map of auths hash (keys) used to successfully pull this imageref
	Auths map[string]*ensuredInfo
}

// EnsureImageExists pulls the image for the specified pod and imageRef, and returns
// (imageRef, error message, error).
func (m *imageManager) EnsureImageExists(ctx context.Context, objRef *v1.ObjectReference, pod *v1.Pod, imgRef string, pullSecrets []v1.Secret, podSandboxConfig *runtimeapi.PodSandboxConfig, podRuntimeHandler string, pullPolicy v1.PullPolicy) (imageRef, message string, err error) {
	logPrefix := fmt.Sprintf("%s/%s/%s", pod.Namespace, pod.Name, imgRef)

	// If the image contains no tag or digest, a default tag should be applied.
	image, err := applyDefaultImageTag(imgRef)
	if err != nil {
		msg := fmt.Sprintf("Failed to apply default image tag %q: %v", imgRef, err)
		m.logIt(objRef, v1.EventTypeWarning, events.FailedToInspectImage, logPrefix, msg, klog.Warning)
		return "", msg, ErrInvalidImageName
	}

	var podAnnotations []kubecontainer.Annotation
	for k, v := range pod.GetAnnotations() {
		podAnnotations = append(podAnnotations, kubecontainer.Annotation{
			Name:  k,
			Value: v,
		})
	}

	spec := kubecontainer.ImageSpec{
		Image:          image,
		Annotations:    podAnnotations,
		RuntimeHandler: podRuntimeHandler,
	}

	imageRef, err = m.imageService.GetImageRef(ctx, spec)
	if err != nil {
		msg := fmt.Sprintf("Failed to inspect image %q: %v", imgRef, err)
		m.logIt(objRef, v1.EventTypeWarning, events.FailedToInspectImage, logPrefix, msg, klog.Warning)
		return "", msg, ErrImageInspect
	}

	present := imageRef != ""
	var pulledBySecret, ensuredBySecret bool
	if present {
		pulledBySecret, ensuredBySecret, err := m.isEnsuredBySecret(imageRef, spec, pullSecrets)
		if err != nil {
			return imageRef, "Error get ensured check by secret", err
		}
		klog.V(5).InfoS("Get ensured check by secret", "image", image, "imageRef", imageRef, "pulledBySecret", pulledBySecret, "ensuredBySecret", ensuredBySecret)
	}

	if !shouldPullImage(pullPolicy, present, pulledBySecret, ensuredBySecret) {
		// should not pull when pull never, or if present and correctly authenticated
		if present {
			msg := fmt.Sprintf("Container image %q already present on machine", imgRef)
			m.logIt(objRef, v1.EventTypeNormal, events.PulledImage, logPrefix, msg, klog.Info)
			if utilfeature.DefaultFeatureGate.Enabled(features.KubeletEnsureSecretPulledImages) && (pulledBySecret && !ensuredBySecret) {
				// TODO: add integration test for this thrown error message
				msg := fmt.Sprintf("Container image %q is present with pull policy of %q but does not have the proper auth (image secret) that was used to pull the image", imageRef, pullPolicy)
				m.logIt(objRef, v1.EventTypeWarning, events.ErrImageNotEnsured, logPrefix, msg, klog.Warning)
				return imageRef, msg, ErrImageNotEnsured
			}
			return imageRef, "", nil
		}
		msg := fmt.Sprintf("Container image %q is not present with pull policy of Never", imgRef)
		m.logIt(objRef, v1.EventTypeWarning, events.ErrImageNeverPullPolicy, logPrefix, msg, klog.Warning)
		return "", msg, ErrImageNeverPull
	}

	backOffKey := fmt.Sprintf("%s_%s", pod.UID, imgRef)
	if m.backOff.IsInBackOffSinceUpdate(backOffKey, m.backOff.Clock.Now()) {
		msg := fmt.Sprintf("Back-off pulling image %q", imgRef)
		m.logIt(objRef, v1.EventTypeNormal, events.BackOffPullImage, logPrefix, msg, klog.Info)
		return "", msg, ErrImagePullBackOff
	}
	m.podPullingTimeRecorder.RecordImageStartedPulling(pod.UID)
	m.logIt(objRef, v1.EventTypeNormal, events.PullingImage, logPrefix, fmt.Sprintf("Pulling image %q", imgRef), klog.Info)
	startTime := time.Now()
	pullChan := make(chan pullResult)
	m.puller.pullImage(ctx, spec, pullSecrets, pullChan, podSandboxConfig)
	imagePullResult := <-pullChan
	if imagePullResult.err != nil {
		m.logIt(objRef, v1.EventTypeWarning, events.FailedToPullImage, logPrefix, fmt.Sprintf("Failed to pull image %q: %v", imgRef, imagePullResult.err), klog.Warning)
		m.backOff.Next(backOffKey, m.backOff.Clock.Now())

		msg, err := evalCRIPullErr(imgRef, imagePullResult.err)
		return "", msg, err
	}
	m.podPullingTimeRecorder.RecordImageFinishedPulling(pod.UID)
	imagePullDuration := time.Since(startTime).Truncate(time.Millisecond)
	m.logIt(objRef, v1.EventTypeNormal, events.PulledImage, logPrefix, fmt.Sprintf("Successfully pulled image %q in %v (%v including waiting). Image size: %v bytes.",
		imgRef, imagePullResult.pullDuration.Truncate(time.Millisecond), imagePullDuration, imagePullResult.imageSize), klog.Info)
	metrics.ImagePullDuration.WithLabelValues(metrics.GetImageSizeBucket(imagePullResult.imageSize)).Observe(imagePullDuration.Seconds())
	m.backOff.GC()

	m.lock.Lock()
	if imagePullResult.pullCredentialsHash == "" {
		// successful pull no auth hash returned, auth was not required so we should reset the hashmap for this
		// imageref since auth is no longer required for the local image cache, allowing use of the ImageRef
		// by other pods if it remains cached and pull policy is PullIfNotPresent
		delete(m.ensureSecretPulledImages, imageRef)
	} else {
		// store/create hashMatch map entry for auth config hash key used to pull the image
		// for this imageref (digest)
		digest := m.ensureSecretPulledImages[imageRef]
		if digest == nil {
			digest = &imagePullInfo{Auths: make(map[string]*ensuredInfo)}
			m.ensureSecretPulledImages[imageRef] = digest
		}
		digest.Auths[imagePullResult.pullCredentialsHash] = &ensuredInfo{true, time.Now()}
	}
	m.lock.Unlock()

	return imagePullResult.imageRef, "", nil
}

func evalCRIPullErr(imgRef string, err error) (errMsg string, errRes error) {
	// Error assertions via errors.Is is not supported by gRPC (remote runtime) errors right now.
	// See https://github.com/grpc/grpc-go/issues/3616
	if strings.HasPrefix(err.Error(), crierrors.ErrRegistryUnavailable.Error()) {
		errMsg = fmt.Sprintf(
			"image pull failed for %s because the registry is unavailable%s",
			imgRef,
			// Trim the error name from the message to convert errors like:
			// "RegistryUnavailable: a more detailed explanation" to:
			// "...because the registry is unavailable: a more detailed explanation"
			strings.TrimPrefix(err.Error(), crierrors.ErrRegistryUnavailable.Error()),
		)
		return errMsg, crierrors.ErrRegistryUnavailable
	}

	if strings.HasPrefix(err.Error(), crierrors.ErrSignatureValidationFailed.Error()) {
		errMsg = fmt.Sprintf(
			"image pull failed for %s because the signature validation failed%s",
			imgRef,
			// Trim the error name from the message to convert errors like:
			// "SignatureValidationFailed: a more detailed explanation" to:
			// "...because the signature validation failed: a more detailed explanation"
			strings.TrimPrefix(err.Error(), crierrors.ErrSignatureValidationFailed.Error()),
		)
		return errMsg, crierrors.ErrSignatureValidationFailed
	}

	// Fallback for no specific error
	return err.Error(), ErrImagePull
}

// applyDefaultImageTag parses a docker image string, if it doesn't contain any tag or digest,
// a default tag will be applied.
func applyDefaultImageTag(image string) (string, error) {
	_, tag, digest, err := parsers.ParseImageName(image)
	if err != nil {
		return "", err
	}
	// we just concatenate the image name with the default tag here instead
	if len(digest) == 0 && len(tag) > 0 && !strings.HasSuffix(image, ":"+tag) {
		// we just concatenate the image name with the default tag here instead
		// of using dockerref.WithTag(named, ...) because that would cause the
		// image to be fully qualified as docker.io/$name if it's a short name
		// (e.g. just busybox). We don't want that to happen to keep the CRI
		// agnostic wrt image names and default hostnames.
		image = image + ":" + tag
	}
	return image, nil
}

// isEnsuredBySecret - returns two bools and error
// 1. pulledBySecret: true if it is in ensured secret pulled image list,
// the list will clean the image after PullImageSecretRecheckPeriod
// 2. ensuredBySecret: true if the secret for an auth used to pull an
// image has already been authenticated through a successful pull request
// and the same auth exists for this podSandbox/image.
func (m *imageManager) isEnsuredBySecret(imageRef string, image kubecontainer.ImageSpec, pullSecrets []v1.Secret) (bool, bool, error) {
	m.lock.Lock()
	defer m.lock.Unlock()
	if imageRef == "" {
		return false, false, errors.New("imageRef is empty")
	}

	// if the image is in the ensured secret pulled image list, it is pulled by secret
	pulledBySecret := m.ensureSecretPulledImages[imageRef] != nil

	img := image.Image
	repoToPull, _, _, err := parsers.ParseImageName(img)
	if err != nil {
		return pulledBySecret, false, err
	}

	if m.keyring == nil {
		return pulledBySecret, false, nil
	}
	keyring, err := credentialprovidersecrets.MakeDockerKeyring(pullSecrets, *m.keyring)
	if err != nil {
		return pulledBySecret, false, err
	}

	creds, withCredentials := keyring.Lookup(repoToPull)
	if !withCredentials {
		return pulledBySecret, false, nil
	}
	// if the pullImageSecretRecheckPeriod is 0, we don't need to check the secret again
	if m.pullImageSecretRecheckPeriod.Duration == 0 {
		return pulledBySecret, true, nil
	}

	for _, currentCreds := range creds {
		auth := &runtimeapi.AuthConfig{
			Username:      currentCreds.Username,
			Password:      currentCreds.Password,
			Auth:          currentCreds.Auth,
			ServerAddress: currentCreds.ServerAddress,
			IdentityToken: currentCreds.IdentityToken,
			RegistryToken: currentCreds.RegistryToken,
		}

		hash, err := kubecontainer.HashAuth(auth)
		if err != nil {
			klog.ErrorS(err, "Failed to hash auth", "auth", auth)
			continue
		}
		digest := m.ensureSecretPulledImages[imageRef]
		if digest != nil {
			ensuredInfo := digest.Auths[hash]
			if ensuredInfo != nil && ensuredInfo.ensured && ensuredInfo.lastEnsuredDate.Add(m.pullImageSecretRecheckPeriod.Duration).After(time.Now()) {
				return pulledBySecret, true, nil
			}
		}
	}
	return pulledBySecret, false, nil
}
