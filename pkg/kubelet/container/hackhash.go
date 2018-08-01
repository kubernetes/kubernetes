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

package container

import (
	"hash/fnv"
	"regexp"

	"k8s.io/api/core/v1"
	hashutil "k8s.io/kubernetes/pkg/util/hash"
)

// TODO: Remove when all pods in cluster have no pod-version annotations

const (
	podVersionLabel   = "k8s.qiniu.com/pod-version"
	podVersionLabel17 = "1.7"
	podVersionLabel18 = "1.8"
)

// Specific func to hack container hash
func hashContainerExt(container *v1.Container, hackFunc hashutil.HackFunc) uint64 {
	hash := fnv.New32a()
	hashutil.DeepHashObjectExt(hash, *container, hackFunc)
	return uint64(hash.Sum32())
}

// For container hash compatibility when upgrade cluster from old version.
// Use different hash method by pod version annotations
func HashContainerByPodVersion(pod *v1.Pod, container *v1.Container) uint64 {
	var containerHash uint64
	version, exists := pod.Annotations[podVersionLabel]
	if exists && version == podVersionLabel17 {
		containerHash = hashContainerExt(container, HackContainerHashTo17)
	} else if exists && version == podVersionLabel18 {
		containerHash = hashContainerExt(container, HackContainerHashTo18)
	} else {
		containerHash = HashContainer(container)
	}
	return containerHash
}

// Convert container hash from 1.8 to 1.7
func hackGoString18to17(s string) string {
	// MountPropagation:(*v1.MountPropagationMode)<nil>
	re := regexp.MustCompile(`\s*MountPropagation:(.*?)([\s}])`)
	s = re.ReplaceAllString(s, `${2}`)
	// AllowPrivilegeEscalation:(*bool)<nil>
	re = regexp.MustCompile(`\s*AllowPrivilegeEscalation:(.*?)([\s}])`)
	s = re.ReplaceAllString(s, `${2}`)
	return s
}

// Convert container hash from 1.9 to 1.8
func hackGoString19to18(s string) string {
	// VolumeDevices:([]v1.VolumeDevice)<nil>
	re := regexp.MustCompile(`\s*VolumeDevices:(.*?)([\s}])`)
	s = re.ReplaceAllString(s, `${2}`)
	return s
}

func HackContainerHashTo17(s string) string {
	return hackGoString18to17(hackGoString19to18(s))
}

func HackContainerHashTo18(s string) string {
	return hackGoString19to18(s)
}
