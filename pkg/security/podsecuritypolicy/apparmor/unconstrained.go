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

package apparmor

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/maps"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

type unconstrained struct{}

var _ AppArmorStrategy = &unconstrained{}

// NewUnconstrainedStrategy creates a new strategy that allows any AppArmor options.
func NewUnconstrainedStrategy() AppArmorStrategy {
	return &unconstrained{}
}

func (_ unconstrained) Generate(annotations map[string]string, container *api.Container) (map[string]string, error) {
	return maps.CopySS(annotations), nil
}

func (_ unconstrained) Validate(pod *api.Pod, container *api.Container) field.ErrorList {
	return nil
}
