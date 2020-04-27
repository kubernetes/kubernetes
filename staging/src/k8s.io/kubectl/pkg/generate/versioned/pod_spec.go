/*
Copyright 2018 The Kubernetes Authors.

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

package versioned

import (
	"fmt"
	v1 "k8s.io/api/core/v1"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"strings"
)

type PodSpec struct{
	Images []string
}

func (s PodSpec) Validate() error {
	if len(s.Images) == 0 {
		return fmt.Errorf("at least one image must be specified")
	}
	return nil
}

// buildPodSpec: parse the image strings and assemble them into the Containers
// of a PodSpec. This is all you need to create the PodSpec for a deployment.
func (s PodSpec) Build() v1.PodSpec {
	podSpec := v1.PodSpec{Containers: []v1.Container{}}
	for _, imageString := range s.Images {
		// Retain just the image name
		imageSplit := strings.Split(imageString, "/")
		name := imageSplit[len(imageSplit)-1]
		// Remove any tag or hash
		if strings.Contains(name, ":") {
			name = strings.Split(name, ":")[0]
		}
		if strings.Contains(name, "@") {
			name = strings.Split(name, "@")[0]
		}
		name = sanitizeAndUniquify(name)
		podSpec.Containers = append(podSpec.Containers, v1.Container{Name: name, Image: imageString})
	}
	return podSpec
}

// sanitizeAndUniquify: replaces characters like "." or "_" into "-" to follow DNS1123 rules.
// Then add random suffix to make it uniquified.
func sanitizeAndUniquify(name string) string {
	if strings.Contains(name, "_") || strings.Contains(name, ".") {
		name = strings.Replace(name, "_", "-", -1)
		name = strings.Replace(name, ".", "-", -1)
		name = fmt.Sprintf("%s-%s", name, utilrand.String(5))
	}
	return name
}

