/*
Copyright 2015 The Kubernetes Authors.

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

package exec

import (
	"bytes"
	"sync"

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/util/ioutils"
	"k8s.io/kubernetes/pkg/probe"

	"k8s.io/klog/v2"
	"k8s.io/utils/exec"
)

const (
	maxReadLength = 10 * 1 << 10 // 10KB
)

// New creates a Prober.
func New(recorder record.EventRecorder) Prober {
	return execProber{
		recorder:   recorder,
		recordOnce: &sync.Once{},
	}
}

// Prober is an interface defining the Probe object for container readiness/liveness checks.
type Prober interface {
	Probe(e exec.Cmd, pod *v1.Pod) (probe.Result, string, error)
}

type execProber struct {
	// recorder is used to emit a single Event when an exec probe times out
	// but the ExecProbeTimeout feature gate is disabled.
	// TODO: remove when ExecProbeTimeout feature gate is removed.
	recorder record.EventRecorder
	// recordOnce is used to ensure only 1 event is emitted by recorder per probe.
	// TODO: remove when ExecProbeTimeout feature gate is removed.
	recordOnce *sync.Once
}

// Probe executes a command to check the liveness/readiness of container
// from executing a command. Returns the Result status, command output, and
// errors if any.
func (pr execProber) Probe(e exec.Cmd, pod *v1.Pod) (probe.Result, string, error) {
	var dataBuffer bytes.Buffer
	writer := ioutils.LimitWriter(&dataBuffer, maxReadLength)

	e.SetStderr(writer)
	e.SetStdout(writer)
	err := e.Start()
	if err == nil {
		err = e.Wait()
	}
	data := dataBuffer.Bytes()

	klog.V(4).Infof("Exec probe response: %q", string(data))
	if err != nil {
		exit, ok := err.(exec.ExitError)
		if ok {
			if exit.ExitStatus() == 0 {
				return probe.Success, string(data), nil
			}
			return probe.Failure, string(data), nil
		}

		timeoutErr, ok := err.(*TimeoutError)
		if ok {
			if utilfeature.DefaultFeatureGate.Enabled(features.ExecProbeTimeout) {
				// When exec probe timeout, data is empty, so we should return timeoutErr.Error() as the stdout.
				return probe.Failure, timeoutErr.Error(), nil
			}

			pr.recordOnce.Do(func() {
				pr.recorder.Event(pod, v1.EventTypeWarning, "KubeletExecProbeTimeoutIgnored", "Kubelet exec probe timed out but was ignored since ExecProbeTimeout feature gate was disabled")
			})

			klog.Warningf("Exec probe timed out after %s but ExecProbeTimeout feature gate was disabled", timeoutErr.Timeout())
		}

		return probe.Unknown, "", err
	}
	return probe.Success, string(data), nil
}
