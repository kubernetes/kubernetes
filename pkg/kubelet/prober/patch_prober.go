package prober

import (
	"net/http"
	"net/url"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/probe"
	httpprobe "k8s.io/kubernetes/pkg/probe/http"
)

func (pb *prober) maybeProbeForBody(prober httpprobe.Prober, url *url.URL, headers http.Header, timeout time.Duration, pod *v1.Pod, container v1.Container, probeType probeType) (probe.Result, string, error) {
	if !isInterestingPod(pod) {
		return prober.Probe(url, headers, timeout)
	}
	bodyProber, ok := prober.(httpprobe.DetailedProber)
	if !ok {
		return prober.Probe(url, headers, timeout)
	}
	result, output, body, probeError := bodyProber.ProbeForBody(url, headers, timeout)
	switch result {
	case probe.Success:
		return result, output, probeError
	case probe.Warning, probe.Failure, probe.Unknown:
		// these pods are interesting enough to show the body content
		klog.Infof("interesting pod/%s container/%s namespace/%s: %s probe status=%v output=%q start-of-body=%s",
			pod.Name, container.Name, pod.Namespace, probeType, result, output, body)

		// in fact, they are so interesting we'll try to send events for them
		pb.recordContainerEvent(pod, &container, v1.EventTypeWarning, "ProbeError", "%s probe error: %s\nbody: %s\n", probeType, output, body)
		return result, output, probeError
	default:
		return result, output, probeError
	}
}

func isInterestingPod(pod *v1.Pod) bool {
	if pod == nil {
		return false
	}
	if strings.HasPrefix(pod.Namespace, "openshift-") {
		return true
	}

	return false
}
