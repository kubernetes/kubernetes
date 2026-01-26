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

package lifecycle

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	versionutil "k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	ndf "k8s.io/component-helpers/nodedeclaredfeatures"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	httpprobe "k8s.io/kubernetes/pkg/probe/http"
	"k8s.io/kubernetes/pkg/security/apparmor"
)

const (
	maxRespBodyLength = 10 * 1 << 10 // 10KB

	AppArmorNotAdmittedReason          = "AppArmor"
	PodLevelResourcesNotAdmittedReason = "PodLevelResourcesNotSupported"

	// Reasons for pod features admission failure
	PodFeatureUnsupported = "PodFeatureUnsupported"
)

type handlerRunner struct {
	httpDoer         kubetypes.HTTPDoer
	commandRunner    kubecontainer.CommandRunner
	containerManager podStatusProvider
	eventRecorder    record.EventRecorder
}

type podStatusProvider interface {
	GetPodStatus(ctx context.Context, uid types.UID, name, namespace string) (*kubecontainer.PodStatus, error)
}

// NewHandlerRunner returns a configured lifecycle handler for a container.
func NewHandlerRunner(httpDoer kubetypes.HTTPDoer, commandRunner kubecontainer.CommandRunner, containerManager podStatusProvider, eventRecorder record.EventRecorder) kubecontainer.HandlerRunner {
	return &handlerRunner{
		httpDoer:         httpDoer,
		commandRunner:    commandRunner,
		containerManager: containerManager,
		eventRecorder:    eventRecorder,
	}
}

func (hr *handlerRunner) Run(ctx context.Context, containerID kubecontainer.ContainerID, pod *v1.Pod, container *v1.Container, handler *v1.LifecycleHandler) (string, error) {
	logger := klog.FromContext(ctx)
	switch {
	case handler.Exec != nil:
		var msg string
		// TODO(tallclair): Pass a proper timeout value.
		output, err := hr.commandRunner.RunInContainer(ctx, containerID, handler.Exec.Command, 0)
		if err != nil {
			msg = fmt.Sprintf("Exec lifecycle hook (%v) for Container %q in Pod %q failed - error: %v, message: %q", handler.Exec.Command, container.Name, format.Pod(pod), err, string(output))
			logger.V(1).Info("Exec lifecycle hook for Container in Pod failed", "execCommand", handler.Exec.Command, "containerName", container.Name, "pod", klog.KObj(pod), "message", string(output), "err", err)
		}
		return msg, err
	case handler.HTTPGet != nil:
		err := hr.runHTTPHandler(ctx, pod, container, handler, hr.eventRecorder)
		var msg string
		if err != nil {
			msg = fmt.Sprintf("HTTP lifecycle hook (%s) for Container %q in Pod %q failed - error: %v", handler.HTTPGet.Path, container.Name, format.Pod(pod), err)
			logger.V(1).Info("HTTP lifecycle hook for Container in Pod failed", "path", handler.HTTPGet.Path, "containerName", container.Name, "pod", klog.KObj(pod), "err", err)
		}
		return msg, err
	case handler.Sleep != nil:
		err := hr.runSleepHandler(ctx, handler.Sleep.Seconds)
		var msg string
		if err != nil {
			msg = fmt.Sprintf("Sleep lifecycle hook (%d) for Container %q in Pod %q failed - error: %v", handler.Sleep.Seconds, container.Name, format.Pod(pod), err)
			logger.V(1).Info("Sleep lifecycle hook for Container in Pod failed", "sleepSeconds", handler.Sleep.Seconds, "containerName", container.Name, "pod", klog.KObj(pod), "err", err)
		}
		return msg, err
	default:
		err := fmt.Errorf("invalid handler: %v", handler)
		msg := fmt.Sprintf("Cannot run handler: %v", err)
		logger.Error(err, "Cannot run handler")
		return msg, err
	}
}

func (hr *handlerRunner) runSleepHandler(ctx context.Context, seconds int64) error {
	if !utilfeature.DefaultFeatureGate.Enabled(features.PodLifecycleSleepAction) {
		return nil
	}
	c := time.After(time.Duration(seconds) * time.Second)
	select {
	case <-ctx.Done():
		// unexpected termination
		metrics.LifecycleHandlerSleepTerminated.Inc()
		return fmt.Errorf("container terminated before sleep hook finished")
	case <-c:
		return nil
	}
}

func (hr *handlerRunner) runHTTPHandler(ctx context.Context, pod *v1.Pod, container *v1.Container, handler *v1.LifecycleHandler, eventRecorder record.EventRecorder) error {
	logger := klog.FromContext(ctx)
	host := handler.HTTPGet.Host
	podIP := host
	if len(host) == 0 {
		status, err := hr.containerManager.GetPodStatus(ctx, pod.UID, pod.Name, pod.Namespace)
		if err != nil {
			logger.Error(err, "Unable to get pod info, event handlers may be invalid.", "pod", klog.KObj(pod))
			return err
		}
		if len(status.IPs) == 0 {
			return fmt.Errorf("failed to find networking container: %v", status)
		}
		host = status.IPs[0]
		podIP = host
	}

	req, err := httpprobe.NewRequestForHTTPGetAction(handler.HTTPGet, container, podIP, "lifecycle")
	if err != nil {
		return err
	}
	resp, err := hr.httpDoer.Do(req)
	discardHTTPRespBody(resp)

	if isHTTPResponseError(err) {
		logger.V(1).Info("HTTPS request to lifecycle hook got HTTP response, retrying with HTTP.", "pod", klog.KObj(pod), "host", req.URL.Host, "err", err)

		req := req.Clone(ctx)
		req.URL.Scheme = "http"
		req.Header.Del("Authorization")
		resp, httpErr := hr.httpDoer.Do(req)

		// clear err since the fallback succeeded
		if httpErr == nil {
			metrics.LifecycleHandlerHTTPFallbacks.Inc()
			if eventRecorder != nil {
				// report the fallback with an event
				eventRecorder.Event(pod, v1.EventTypeWarning, "LifecycleHTTPFallback", fmt.Sprintf("request to HTTPS lifecycle hook %s got HTTP response, retry with HTTP succeeded", req.URL.Host))
			}
			err = nil
		}
		discardHTTPRespBody(resp)
	}
	return err
}

func discardHTTPRespBody(resp *http.Response) {
	if resp == nil {
		return
	}

	// Ensure the response body is fully read and closed
	// before we reconnect, so that we reuse the same TCP
	// connection.
	defer resp.Body.Close()

	if resp.ContentLength <= maxRespBodyLength {
		io.Copy(io.Discard, &io.LimitedReader{R: resp.Body, N: maxRespBodyLength})
	}
}

// NewAppArmorAdmitHandler returns a PodAdmitHandler which is used to evaluate
// if a pod can be admitted from the perspective of AppArmor.
func NewAppArmorAdmitHandler(validator apparmor.Validator) PodAdmitHandler {
	return &appArmorAdmitHandler{
		Validator: validator,
	}
}

type appArmorAdmitHandler struct {
	apparmor.Validator
}

func (a *appArmorAdmitHandler) Admit(attrs *PodAdmitAttributes) PodAdmitResult {
	// If the pod is already running or terminated, no need to recheck AppArmor.
	if attrs.Pod.Status.Phase != v1.PodPending {
		return PodAdmitResult{Admit: true}
	}

	err := a.Validate(attrs.Pod)
	if err == nil {
		return PodAdmitResult{Admit: true}
	}
	return PodAdmitResult{
		Admit:   false,
		Reason:  AppArmorNotAdmittedReason,
		Message: fmt.Sprintf("Cannot enforce AppArmor: %v", err),
	}
}

func isHTTPResponseError(err error) bool {
	if err == nil {
		return false
	}
	urlErr := &url.Error{}
	if !errors.As(err, &urlErr) {
		return false
	}
	return strings.Contains(urlErr.Err.Error(), "server gave HTTP response to HTTPS client")
}

// NewPodFeaturesAdmitHandler returns a PodAdmitHandler which is used to evaluate
// if a pod can be admitted from the perspective of pod features compatibility.
func NewPodFeaturesAdmitHandler() PodAdmitHandler {
	return &podFeaturesAdmitHandler{}
}

type podFeaturesAdmitHandler struct{}

func (h *podFeaturesAdmitHandler) Admit(attrs *PodAdmitAttributes) PodAdmitResult {
	return isPodLevelResourcesSupported(attrs.Pod)
}

// declaredFeaturesAdmitHandler is a PodAdmitHandler that checks a pod's feature requirements.
type declaredFeaturesAdmitHandler struct {
	ndfFramework *ndf.Framework
	ndfSet       ndf.FeatureSet
	version      *versionutil.Version
}

// NewDeclaredFeaturesAdmitHandler returns a new features admit handler.
func NewDeclaredFeaturesAdmitHandler(nodeDeclaredFeaturesHelper *ndf.Framework, nodeDeclaredFeaturesSet ndf.FeatureSet, version *versionutil.Version) PodAdmitHandler {
	return &declaredFeaturesAdmitHandler{
		ndfFramework: nodeDeclaredFeaturesHelper,
		ndfSet:       nodeDeclaredFeaturesSet,
		version:      version,
	}
}

// Admit checks if a pod's feature requirements are met by the node.
func (c *declaredFeaturesAdmitHandler) Admit(attrs *PodAdmitAttributes) PodAdmitResult {
	pod := attrs.Pod

	podInfo := &ndf.PodInfo{Spec: &pod.Spec, Status: &pod.Status}
	reqs, err := c.ndfFramework.InferForPodScheduling(podInfo, c.version)
	if err != nil {
		return PodAdmitResult{
			Admit:   false,
			Reason:  PodFeatureUnsupported,
			Message: fmt.Sprintf("Failed to infer pod's feature requirements: %v", err),
		}
	}

	if reqs.Len() == 0 {
		return PodAdmitResult{Admit: true}
	}

	matchResult, err := ndf.MatchNodeFeatureSet(reqs, c.ndfSet)
	if err != nil {
		return PodAdmitResult{
			Admit:   false,
			Reason:  PodFeatureUnsupported,
			Message: fmt.Sprintf("Failed to match pod's feature requirements against the node: %v", err),
		}
	}

	if !matchResult.IsMatch {
		return PodAdmitResult{
			Admit:   false,
			Reason:  PodFeatureUnsupported,
			Message: fmt.Sprintf("Pod requires node features that are not available: %s", strings.Join(matchResult.UnsatisfiedRequirements, ", ")),
		}
	}

	return PodAdmitResult{Admit: true}
}
