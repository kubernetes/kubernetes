/*
Copyright 2023 The Kubernetes Authors.

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

package prober

import (
	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	"testing"
)

func TestNewExecProbeRunner(t *testing.T) {
	runner := newExecProbeRunner()

	pod := newPod(v1.ProbeHandler{Exec: &v1.ExecAction{
		Command: []string{"test", "-f", "/app/run.lock"},
	}})

	// runner.synced should be false, as we haven't synced it yet.
	assert.False(t, runner.synced)
	_ = runner.sync(pod.Spec.Containers[0], pod.Status, liveness)
	command := runner.command

	// runner.synced should be true, after first sync.
	assert.True(t, runner.synced)

	// command should not be recreated after calling sync again
	_ = runner.sync(pod.Spec.Containers[0], pod.Status, liveness)
	for i := 0; i < len(command); i++ {
		assert.Same(t, &command[i], &runner.command[i])
	}
}
