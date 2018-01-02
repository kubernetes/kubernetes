// Copyright 2016 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package rkt

import pkgPod "github.com/coreos/rkt/pkg/pod"

// NewPodFromInternalPod converts *pkgPod.Pod to *Pod
func NewPodFromInternalPod(p *pkgPod.Pod) (*Pod, error) {
	pod := &Pod{
		UUID:     p.UUID.String(),
		State:    p.State(),
		Networks: p.Nets,
	}

	startTime, err := p.StartTime()
	if err != nil {
		return nil, err
	}

	if !startTime.IsZero() {
		startedAt := startTime.Unix()
		pod.StartedAt = &startedAt
	}

	if !p.PodManifestAvailable() {
		return pod, nil
	}
	// TODO(vc): we should really hold a shared lock here to prevent gc of the pod
	_, manifest, err := p.PodManifest()
	if err != nil {
		return nil, err
	}

	for _, app := range manifest.Apps {
		pod.AppNames = append(pod.AppNames, app.Name.String())
	}

	if len(manifest.UserAnnotations) > 0 {
		pod.UserAnnotations = make(map[string]string)
		for name, value := range manifest.UserAnnotations {
			pod.UserAnnotations[name] = value
		}
	}

	if len(manifest.UserLabels) > 0 {
		pod.UserLabels = make(map[string]string)
		for name, value := range manifest.UserLabels {
			pod.UserLabels[name] = value
		}
	}

	return pod, nil
}
