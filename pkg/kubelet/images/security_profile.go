/*
Copyright The Kubernetes Authors.

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
	"strconv"
	"strings"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	v1 "k8s.io/api/core/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	crierrors "k8s.io/cri-api/pkg/errors"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
)

// Values of the result label of kubelet_security_profile_pull_duration_seconds.
const (
	securityProfilePullResultSuccess            = "success"
	securityProfilePullErrorRegistryUnavailable = "registry_unavailable"
	securityProfilePullErrorSignatureValidation = "signature_validation_failed"
	securityProfilePullErrorInvalid             = "security_profile_invalid"
	securityProfilePullErrorUnauthenticated     = "unauthenticated"
	securityProfilePullErrorOther               = "other"
)

// EnsureSecurityProfile pulls the security profile at ref for the pod.
func (m *imageManager) EnsureSecurityProfile(ctx context.Context, objRef *v1.ObjectReference, pod *v1.Pod, ref string, kind runtimeapi.SecurityProfileKind, pullSecrets []v1.Secret, podSandboxConfig *runtimeapi.PodSandboxConfig, podRuntimeHandler string) error {
	if m.securityProfilePuller == nil {
		return fmt.Errorf("security profile pulls require the %s feature gate", features.SecurityProfileOCI)
	}
	logger := klog.FromContext(ctx)
	logPrefix := fmt.Sprintf("%s/%s/%s", pod.Namespace, pod.Name, ref)

	backOffKey := fmt.Sprintf("%s_securityprofile_%s", pod.UID, ref)
	if m.backOff.IsInBackOffSinceUpdate(backOffKey, m.backOff.Clock.Now()) {
		msg := fmt.Sprintf("Back-off pulling security profile %q", ref)
		m.logIt(logger, objRef, v1.EventTypeNormal, events.BackOffPullImage, logPrefix, msg)
		return fmt.Errorf("%w: %s", ErrSecurityProfilePullBackOff, msg)
	}

	startTime := time.Now()
	pullCredentials, err := m.makeLookupPullCredentialsFunc(ref, pod, pullSecrets, podSandboxConfig)()
	if err != nil {
		metrics.SecurityProfilePullDuration.WithLabelValues("false", securityProfilePullErrorOther).Observe(time.Since(startTime).Seconds())
		m.logIt(logger, objRef, v1.EventTypeWarning, events.FailedToPullSecurityProfile, logPrefix, fmt.Sprintf("Failed to look up credentials for security profile %q: %v", ref, err))
		m.backOff.Next(backOffKey, m.backOff.Clock.Now())
		return fmt.Errorf("failed to look up credentials for security profile %q: %w", ref, err)
	}

	spec := kubecontainer.ImageSpec{Image: ref, RuntimeHandler: podRuntimeHandler}
	cached, err := m.securityProfilePuller.pullSecurityProfile(ctx, spec, pullCredentials, podSandboxConfig, kind)
	if err != nil && ctx.Err() != nil {
		// The pod sync was cancelled, which is neither a pull failure nor a
		// reason to back off.
		return ctx.Err()
	}
	if err != nil {
		reason := securityProfilePullErrorReason(err)
		metrics.SecurityProfilePullDuration.WithLabelValues("false", reason).Observe(time.Since(startTime).Seconds())
		m.logIt(logger, objRef, v1.EventTypeWarning, events.FailedToPullSecurityProfile, logPrefix, fmt.Sprintf("Failed to pull security profile %q: %v", ref, err))
		// A started pod keeps syncing after a rejection, so it backs off too.
		m.backOff.Next(backOffKey, m.backOff.Clock.Now())
		if IsSecurityProfileRejected(err) {
			return fmt.Errorf("%w: security profile %q: %w", ErrSecurityProfileRejected, ref, err)
		}
		return fmt.Errorf("failed to pull security profile %q: %w", ref, err)
	}
	metrics.SecurityProfilePullDuration.WithLabelValues(strconv.FormatBool(cached), securityProfilePullResultSuccess).Observe(time.Since(startTime).Seconds())
	m.backOff.GC()
	if !cached {
		m.logIt(logger, objRef, v1.EventTypeNormal, events.PulledSecurityProfile, logPrefix, fmt.Sprintf("Successfully pulled security profile %q in %v", ref, time.Since(startTime).Truncate(time.Millisecond)))
	}
	return nil
}

// IsSecurityProfileRejected returns whether a PullSecurityProfile error is
// permanent: the runtime rejected the request or the artifact with the
// SecurityProfileInvalid well-known error, or it does not implement the call.
// The kubelet retries every other error.
func IsSecurityProfileRejected(err error) bool {
	return securityProfilePullErrorReason(err) == securityProfilePullErrorInvalid || status.Code(err) == codes.Unimplemented
}

// securityProfilePullErrorReason classifies a PullSecurityProfile error by
// the well-known error message prefixes of the CRI, which runtimes may return
// with a gRPC status code.
func securityProfilePullErrorReason(err error) string {
	msg := status.Convert(err).Message()
	switch {
	case strings.HasPrefix(msg, crierrors.ErrSecurityProfileInvalid.Error()):
		return securityProfilePullErrorInvalid
	case strings.HasPrefix(msg, crierrors.ErrRegistryUnavailable.Error()):
		return securityProfilePullErrorRegistryUnavailable
	case strings.HasPrefix(msg, crierrors.ErrSignatureValidationFailed.Error()):
		return securityProfilePullErrorSignatureValidation
	}
	switch status.Code(err) {
	case codes.Unauthenticated, codes.PermissionDenied:
		return securityProfilePullErrorUnauthenticated
	}
	return securityProfilePullErrorOther
}
