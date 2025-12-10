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
	"fmt"
	"strings"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/klog/v2"

	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	crierrors "k8s.io/cri-api/pkg/errors"
	"k8s.io/kubernetes/pkg/credentialprovider"
	credentialproviderplugin "k8s.io/kubernetes/pkg/credentialprovider/plugin"
	credentialprovidersecrets "k8s.io/kubernetes/pkg/credentialprovider/secrets"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfiginternal "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/images/pullmanager"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/util/parsers"
	"k8s.io/utils/ptr"
)

func init() {
	registerMetrics()
}

type ImagePodPullingTimeRecorder interface {
	RecordImageStartedPulling(podUID types.UID)
	RecordImageFinishedPulling(podUID types.UID)
}

// imageManager provides the functionalities for image pulling.
type imageManager struct {
	recorder         record.EventRecorder
	imageService     kubecontainer.ImageService
	imagePullManager pullmanager.ImagePullManager
	backOff          *flowcontrol.Backoff
	prevPullErrMsg   sync.Map

	// It will check the presence of the image, and report the 'image pulling', image pulled' events correspondingly.
	puller      imagePuller
	nodeKeyring credentialprovider.DockerKeyring

	podPullingTimeRecorder ImagePodPullingTimeRecorder
}

var _ ImageManager = &imageManager{}

// NewImageManager instantiates a new ImageManager object.
func NewImageManager(
	recorder record.EventRecorder,
	nodeKeyring credentialprovider.DockerKeyring,
	imageService kubecontainer.ImageService,
	imagePullManager pullmanager.ImagePullManager,
	imageBackOff *flowcontrol.Backoff,
	serialized bool,
	maxParallelImagePulls *int32,
	qps float32,
	burst int,
	podPullingTimeRecorder ImagePodPullingTimeRecorder,
) ImageManager {

	imageService = throttleImagePulling(imageService, qps, burst)

	var puller imagePuller
	if serialized {
		puller = newSerialImagePuller(imageService)
	} else {
		puller = newParallelImagePuller(imageService, maxParallelImagePulls)
	}
	return &imageManager{
		recorder:               recorder,
		imageService:           imageService,
		imagePullManager:       imagePullManager,
		nodeKeyring:            nodeKeyring,
		backOff:                imageBackOff,
		puller:                 puller,
		podPullingTimeRecorder: podPullingTimeRecorder,
	}
}

// imagePullPrecheck inspects the pull policy and checks for image presence accordingly,
// returning (imageRef, error msg, err) and logging any errors.
func (m *imageManager) imagePullPrecheck(
	ctx context.Context,
	objRef *v1.ObjectReference,
	logPrefix string,
	pullPolicy v1.PullPolicy,
	spec *kubecontainer.ImageSpec,
	requestedImage string,
) (imageRef string, imageFound *bool, msg string, err error) {
	switch pullPolicy {
	case v1.PullAlways:
		return "", nil, msg, nil
	case v1.PullIfNotPresent, v1.PullNever:
		imageRef, err = m.imageService.GetImageRef(ctx, *spec)
		if err != nil {
			msg = fmt.Sprintf("Failed to inspect image %q: %v", imageRef, err)
			m.logIt(objRef, v1.EventTypeWarning, events.FailedToInspectImage, logPrefix, msg, klog.Warning)
			return "", nil, msg, ErrImageInspect
		}
	}

	imageFound = ptr.To(len(imageRef) > 0)
	if !*imageFound && pullPolicy == v1.PullNever {
		msg, err = m.imageNotPresentOnNeverPolicyError(logPrefix, objRef, requestedImage)
		return "", ptr.To(false), msg, err
	}

	return imageRef, imageFound, msg, nil
}

// records an event using ref, event msg.  log to glog using prefix, msg, logFn
func (m *imageManager) logIt(objRef *v1.ObjectReference, eventtype, event, prefix, msg string, logFn func(args ...interface{})) {
	if objRef != nil {
		m.recorder.Event(objRef, eventtype, event, msg)
	} else {
		logFn(fmt.Sprint(prefix, " ", msg))
	}
}

// imageNotPresentOnNeverPolicy error is a utility function that emits an event about
// an image not being present and returns the appropriate error to be passed on.
//
// Called in 2 scenarios:
//  1. image is not present with `imagePullPolicy: Never“
//  2. image is present but cannot be accessed with the presented set of credentials
//
// We don't want to reveal the presence of an image if it cannot be accessed, hence we
// want the same behavior in both the above scenarios.
func (m *imageManager) imageNotPresentOnNeverPolicyError(logPrefix string, objRef *v1.ObjectReference, requestedImage string) (string, error) {
	msg := fmt.Sprintf("Container image %q is not present with pull policy of Never", requestedImage)
	m.logIt(objRef, v1.EventTypeWarning, events.ErrImageNeverPullPolicy, logPrefix, msg, klog.Warning)
	return msg, ErrImageNeverPull
}

// EnsureImageExists pulls the image for the specified pod and requestedImage, and returns
// (imageRef, error message, error).
func (m *imageManager) EnsureImageExists(ctx context.Context, objRef *v1.ObjectReference, pod *v1.Pod, requestedImage string, pullSecrets []v1.Secret, podSandboxConfig *runtimeapi.PodSandboxConfig, podRuntimeHandler string, pullPolicy v1.PullPolicy) (imageRef, message string, err error) {
	logPrefix := fmt.Sprintf("%s/%s/%s", pod.Namespace, pod.Name, requestedImage)

	var ( // variables used to record metrics
		imagePresentLocally *bool
		imagePullRequired   *bool
	)
	defer func() {
		recordEnsureImageRequest(pullPolicy, imagePresentLocally, imagePullRequired)
	}()

	// If the image contains no tag or digest, a default tag should be applied.
	image, err := applyDefaultImageTag(requestedImage)
	if err != nil {
		msg := fmt.Sprintf("Failed to apply default image tag %q: %v", requestedImage, err)
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

	imageRef, imagePresentLocally, message, err = m.imagePullPrecheck(ctx, objRef, logPrefix, pullPolicy, &spec, requestedImage)
	if err != nil {
		return "", message, err
	}

	// wrap the lookup in a function to ensure that we only look up credentials once and no earlier than needed
	lookupPullCredentials := m.makeLookupPullCredentialsFunc(spec.Image, pod, pullSecrets, podSandboxConfig)

	getPodCredentials := func() ([]kubeletconfiginternal.ImagePullSecret, *kubeletconfiginternal.ImagePullServiceAccount, error) {
		pullCredentials, err := lookupPullCredentials()
		if err != nil {
			return nil, nil, err
		}

		var imagePullSecrets []kubeletconfiginternal.ImagePullSecret
		// we don't take the audience of the service account into account, so there can only
		// be one imagePullServiceAccount per pod when we try to make a decision.
		var imagePullServiceAccount *kubeletconfiginternal.ImagePullServiceAccount
		for _, s := range pullCredentials {
			if s.Source == nil {
				// we're only interested in creds that are not node accessible
				continue
			}
			switch {
			case s.Source.Secret != nil:
				imagePullSecrets = append(imagePullSecrets, kubeletconfiginternal.ImagePullSecret{
					UID:            s.Source.Secret.UID,
					Name:           s.Source.Secret.Name,
					Namespace:      s.Source.Secret.Namespace,
					CredentialHash: s.AuthConfigHash,
				})
			case s.Source.ServiceAccount != nil && imagePullServiceAccount == nil:
				imagePullServiceAccount = &kubeletconfiginternal.ImagePullServiceAccount{
					UID:       s.Source.ServiceAccount.UID,
					Name:      s.Source.ServiceAccount.Name,
					Namespace: s.Source.ServiceAccount.Namespace,
				}
			}
		}

		return imagePullSecrets, imagePullServiceAccount, nil
	}

	if imageRef != "" {
		imagePullRequired = ptr.To(false) // should be overridden right after this if-block in case pull was required

		if !utilfeature.DefaultFeatureGate.Enabled(features.KubeletEnsureSecretPulledImages) {
			msg := fmt.Sprintf("Container image %q already present on machine", requestedImage)
			m.logIt(objRef, v1.EventTypeNormal, events.PulledImage, logPrefix, msg, klog.Info)

			// we need to stop recording if it is still in progress, as the image
			// has already been pulled by another pod.
			m.podPullingTimeRecorder.RecordImageFinishedPulling(pod.UID)

			return imageRef, msg, nil
		}
		if pullRequired, err := m.imagePullManager.MustAttemptImagePull(ctx, requestedImage, imageRef, getPodCredentials); err != nil {
			imagePullRequired = nil
			return "", err.Error(), err
		} else if !pullRequired {
			msg := fmt.Sprintf("Container image %q already present on machine and can be accessed by the pod", requestedImage)
			m.logIt(objRef, v1.EventTypeNormal, events.PulledImage, logPrefix, msg, klog.Info)

			// we need to stop recording if it is still in progress, as the image
			// has already been pulled by another pod.
			m.podPullingTimeRecorder.RecordImageFinishedPulling(pod.UID)

			return imageRef, msg, nil
		}
	}
	imagePullRequired = ptr.To(true)

	if pullPolicy == v1.PullNever {
		// The image is present as confirmed by imagePullPrecheck but it apparently
		// wasn't accessible given the credentials check by the imagePullManager.
		msg, err := m.imageNotPresentOnNeverPolicyError(logPrefix, objRef, requestedImage)
		return "", msg, err
	}

	pullCredentials, err := lookupPullCredentials()
	if err != nil {
		return "", err.Error(), err
	}

	return m.pullImage(ctx, logPrefix, objRef, pod.UID, requestedImage, spec, pullCredentials, podSandboxConfig)
}

func (m *imageManager) makeLookupPullCredentialsFunc(image string, pod *v1.Pod, pullSecrets []v1.Secret, podSandboxConfig *runtimeapi.PodSandboxConfig) func() ([]credentialprovider.TrackedAuthConfig, error) {
	return sync.OnceValues(func() ([]credentialprovider.TrackedAuthConfig, error) {
		repoToPull, _, _, err := parsers.ParseImageName(image)
		if err != nil {
			return nil, err
		}

		// construct the dynamic keyring using the providers we have in the kubelet
		var podName, podNamespace, podUID string
		if utilfeature.DefaultFeatureGate.Enabled(features.KubeletServiceAccountTokenForCredentialProviders) {
			sandboxMetadata := podSandboxConfig.GetMetadata()

			podName = sandboxMetadata.Name
			podNamespace = sandboxMetadata.Namespace
			podUID = sandboxMetadata.Uid
		}

		externalCredentialProviderKeyring := credentialproviderplugin.NewExternalCredentialProviderDockerKeyring(
			podNamespace,
			podName,
			podUID,
			pod.Spec.ServiceAccountName)

		keyring, err := credentialprovidersecrets.MakeDockerKeyring(pullSecrets, credentialprovider.UnionDockerKeyring{m.nodeKeyring, externalCredentialProviderKeyring})
		if err != nil {
			return nil, err
		}

		pullCredentials, _ := keyring.Lookup(repoToPull)
		return pullCredentials, nil
	})
}

func (m *imageManager) pullImage(ctx context.Context, logPrefix string, objRef *v1.ObjectReference, podUID types.UID, image string, imgSpec kubecontainer.ImageSpec, pullCredentials []credentialprovider.TrackedAuthConfig, podSandboxConfig *runtimeapi.PodSandboxConfig) (imageRef, message string, err error) {
	var pullSucceeded bool
	var finalPullCredentials *credentialprovider.TrackedAuthConfig

	if utilfeature.DefaultFeatureGate.Enabled(features.KubeletEnsureSecretPulledImages) {
		if err := m.imagePullManager.RecordPullIntent(image); err != nil {
			return "", fmt.Sprintf("Failed to record image pull intent for container image %q: %v", image, err), err
		}

		defer func() {
			if pullSucceeded {
				m.imagePullManager.RecordImagePulled(ctx, image, imageRef, trackedToImagePullCreds(finalPullCredentials))
			} else {
				m.imagePullManager.RecordImagePullFailed(ctx, image)
			}
		}()
	}

	backOffKey := fmt.Sprintf("%s_%s", podUID, image)
	if m.backOff.IsInBackOffSinceUpdate(backOffKey, m.backOff.Clock.Now()) {
		msg := fmt.Sprintf("Back-off pulling image %q", image)
		m.logIt(objRef, v1.EventTypeNormal, events.BackOffPullImage, logPrefix, msg, klog.Info)

		// Wrap the error from the actual pull if available.
		// This information is populated to the pods
		// .status.containerStatuses[*].state.waiting.message.
		prevPullErrMsg, ok := m.prevPullErrMsg.Load(backOffKey)
		if ok {
			msg = fmt.Sprintf("%s: %s", msg, prevPullErrMsg)
		}

		return "", msg, ErrImagePullBackOff
	}
	// Ensure that the map cannot grow indefinitely.
	m.prevPullErrMsg.Delete(backOffKey)

	m.podPullingTimeRecorder.RecordImageStartedPulling(podUID)
	m.logIt(objRef, v1.EventTypeNormal, events.PullingImage, logPrefix, fmt.Sprintf("Pulling image %q", image), klog.Info)
	startTime := time.Now()

	pullChan := make(chan pullResult)
	m.puller.pullImage(ctx, imgSpec, pullCredentials, pullChan, podSandboxConfig)
	imagePullResult := <-pullChan
	if imagePullResult.err != nil {
		m.logIt(objRef, v1.EventTypeWarning, events.FailedToPullImage, logPrefix, fmt.Sprintf("Failed to pull image %q: %v", image, imagePullResult.err), klog.Warning)
		m.backOff.Next(backOffKey, m.backOff.Clock.Now())
		msg, err := evalCRIPullErr(image, imagePullResult.err)

		// Store the actual pull error for providing that information during
		// the image pull back-off.
		m.prevPullErrMsg.Store(backOffKey, fmt.Sprintf("%s: %s", err, msg))

		return "", msg, err
	}
	m.podPullingTimeRecorder.RecordImageFinishedPulling(podUID)
	imagePullDuration := time.Since(startTime).Truncate(time.Millisecond)
	m.logIt(objRef, v1.EventTypeNormal, events.PulledImage, logPrefix, fmt.Sprintf("Successfully pulled image %q in %v (%v including waiting). Image size: %v bytes.",
		image, imagePullResult.pullDuration.Truncate(time.Millisecond), imagePullDuration, imagePullResult.imageSize), klog.Info)
	metrics.ImagePullDuration.WithLabelValues(metrics.GetImageSizeBucket(imagePullResult.imageSize)).Observe(imagePullDuration.Seconds())
	m.backOff.GC()
	finalPullCredentials = imagePullResult.credentialsUsed
	pullSucceeded = true

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

func trackedToImagePullCreds(trackedCreds *credentialprovider.TrackedAuthConfig) *kubeletconfiginternal.ImagePullCredentials {
	ret := &kubeletconfiginternal.ImagePullCredentials{}
	switch {
	case trackedCreds == nil, trackedCreds.Source == nil:
		ret.NodePodsAccessible = true
	case trackedCreds.Source.Secret != nil:
		sourceSecret := trackedCreds.Source.Secret
		ret.KubernetesSecrets = []kubeletconfiginternal.ImagePullSecret{
			{
				UID:            sourceSecret.UID,
				Name:           sourceSecret.Name,
				Namespace:      sourceSecret.Namespace,
				CredentialHash: trackedCreds.AuthConfigHash,
			},
		}
	case trackedCreds.Source.ServiceAccount != nil:
		sourceServiceAccount := trackedCreds.Source.ServiceAccount
		ret.KubernetesServiceAccounts = []kubeletconfiginternal.ImagePullServiceAccount{
			{
				UID:       sourceServiceAccount.UID,
				Name:      sourceServiceAccount.Name,
				Namespace: sourceServiceAccount.Namespace,
			},
		}
	}

	return ret
}
