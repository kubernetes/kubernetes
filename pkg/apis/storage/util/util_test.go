/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"fmt"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/util/diff"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/features"
)

func TestDropAllowVolumeExpansion(t *testing.T) {
	allowVolumeExpansion := false
	scWithoutAllowVolumeExpansion := func() *storage.StorageClass {
		return &storage.StorageClass{}
	}
	scWithAllowVolumeExpansion := func() *storage.StorageClass {
		return &storage.StorageClass{
			AllowVolumeExpansion: &allowVolumeExpansion,
		}
	}

	scInfo := []struct {
		description             string
		hasAllowVolumeExpansion bool
		sc                      func() *storage.StorageClass
	}{
		{
			description:             "StorageClass Without AllowVolumeExpansion",
			hasAllowVolumeExpansion: false,
			sc:                      scWithoutAllowVolumeExpansion,
		},
		{
			description:             "StorageClass With AllowVolumeExpansion",
			hasAllowVolumeExpansion: true,
			sc:                      scWithAllowVolumeExpansion,
		},
		{
			description:             "is nil",
			hasAllowVolumeExpansion: false,
			sc:                      func() *storage.StorageClass { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldStorageClassInfo := range scInfo {
			for _, newStorageClassInfo := range scInfo {
				oldStorageClassHasAllowVolumeExpansion, oldStorageClass := oldStorageClassInfo.hasAllowVolumeExpansion, oldStorageClassInfo.sc()
				newStorageClassHasAllowVolumeExpansion, newStorageClass := newStorageClassInfo.hasAllowVolumeExpansion, newStorageClassInfo.sc()
				if newStorageClass == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old StorageClass %v, new StorageClass %v", enabled, oldStorageClassInfo.description, newStorageClassInfo.description), func(t *testing.T) {
					defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ExpandPersistentVolumes, enabled)()

					DropDisabledFields(newStorageClass, oldStorageClass)

					// old StorageClass should never be changed
					if !reflect.DeepEqual(oldStorageClass, oldStorageClassInfo.sc()) {
						t.Errorf("old StorageClass changed: %v", diff.ObjectReflectDiff(oldStorageClass, oldStorageClassInfo.sc()))
					}

					switch {
					case enabled || oldStorageClassHasAllowVolumeExpansion:
						// new StorageClass should not be changed if the feature is enabled, or if the old StorageClass had AllowVolumeExpansion
						if !reflect.DeepEqual(newStorageClass, newStorageClassInfo.sc()) {
							t.Errorf("new StorageClass changed: %v", diff.ObjectReflectDiff(newStorageClass, newStorageClassInfo.sc()))
						}
					case newStorageClassHasAllowVolumeExpansion:
						// new StorageClass should be changed
						if reflect.DeepEqual(newStorageClass, newStorageClassInfo.sc()) {
							t.Errorf("new StorageClass was not changed")
						}
						// new StorageClass should not have AllowVolumeExpansion
						if !reflect.DeepEqual(newStorageClass, scWithoutAllowVolumeExpansion()) {
							t.Errorf("new StorageClass had StorageClassAllowVolumeExpansion: %v", diff.ObjectReflectDiff(newStorageClass, scWithoutAllowVolumeExpansion()))
						}
					default:
						// new StorageClass should not need to be changed
						if !reflect.DeepEqual(newStorageClass, newStorageClassInfo.sc()) {
							t.Errorf("new StorageClass changed: %v", diff.ObjectReflectDiff(newStorageClass, newStorageClassInfo.sc()))
						}
					}
				})
			}
		}
	}
}

func TestDropVolumeBindingMode(t *testing.T) {
	volumeBindingMode := storage.VolumeBindingWaitForFirstConsumer
	scWithoutVolumeBindingMode := func() *storage.StorageClass {
		return &storage.StorageClass{}
	}
	scWithVolumeBindingMode := func() *storage.StorageClass {
		return &storage.StorageClass{
			VolumeBindingMode: &volumeBindingMode,
		}
	}

	scInfo := []struct {
		description          string
		hasVolumeBindingMode bool
		sc                   func() *storage.StorageClass
	}{
		{
			description:          "StorageClass Without VolumeBindingMode",
			hasVolumeBindingMode: false,
			sc:                   scWithoutVolumeBindingMode,
		},
		{
			description:          "StorageClass With VolumeBindingMode",
			hasVolumeBindingMode: true,
			sc:                   scWithVolumeBindingMode,
		},
		{
			description:          "is nil",
			hasVolumeBindingMode: false,
			sc:                   func() *storage.StorageClass { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldStorageClassInfo := range scInfo {
			for _, newStorageClassInfo := range scInfo {
				oldStorageClassHasVolumeBindingMode, oldStorageClass := oldStorageClassInfo.hasVolumeBindingMode, oldStorageClassInfo.sc()
				newStorageClassHasVolumeBindingMode, newStorageClass := newStorageClassInfo.hasVolumeBindingMode, newStorageClassInfo.sc()
				if newStorageClass == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old StorageClass %v, new StorageClass %v", enabled, oldStorageClassInfo.description, newStorageClassInfo.description), func(t *testing.T) {
					defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.VolumeScheduling, enabled)()

					DropDisabledFields(newStorageClass, oldStorageClass)

					// old StorageClass should never be changed
					if !reflect.DeepEqual(oldStorageClass, oldStorageClassInfo.sc()) {
						t.Errorf("old StorageClass changed: %v", diff.ObjectReflectDiff(oldStorageClass, oldStorageClassInfo.sc()))
					}

					switch {
					case enabled || oldStorageClassHasVolumeBindingMode:
						// new StorageClass should not be changed if the feature is enabled, or if the old StorageClass had VolumeBindingMode
						if !reflect.DeepEqual(newStorageClass, newStorageClassInfo.sc()) {
							t.Errorf("new StorageClass changed: %v", diff.ObjectReflectDiff(newStorageClass, newStorageClassInfo.sc()))
						}
					case newStorageClassHasVolumeBindingMode:
						// new StorageClass should be changed
						if reflect.DeepEqual(newStorageClass, newStorageClassInfo.sc()) {
							t.Errorf("new StorageClass was not changed")
						}
						// new StorageClass should not have VolumeBindingMode
						if !reflect.DeepEqual(newStorageClass, scWithoutVolumeBindingMode()) {
							t.Errorf("new StorageClass had StorageClassVolumeBindingMode: %v", diff.ObjectReflectDiff(newStorageClass, scWithoutVolumeBindingMode()))
						}
					default:
						// new StorageClass should not need to be changed
						if !reflect.DeepEqual(newStorageClass, newStorageClassInfo.sc()) {
							t.Errorf("new StorageClass changed: %v", diff.ObjectReflectDiff(newStorageClass, newStorageClassInfo.sc()))
						}
					}
				})
			}
		}
	}
}

func TestDropAllowedTopologies(t *testing.T) {
	allowedTopologies := []api.TopologySelectorTerm{
		{
			MatchLabelExpressions: []api.TopologySelectorLabelRequirement{
				{
					Key:    "key1",
					Values: []string{"value1"},
				},
			},
		},
	}
	scWithoutAllowedTopologies := func() *storage.StorageClass {
		return &storage.StorageClass{}
	}
	scWithAllowedTopologies := func() *storage.StorageClass {
		return &storage.StorageClass{
			AllowedTopologies: allowedTopologies,
		}
	}

	scInfo := []struct {
		description          string
		hasAllowedTopologies bool
		sc                   func() *storage.StorageClass
	}{
		{
			description:          "StorageClass Without AllowedTopologies",
			hasAllowedTopologies: false,
			sc:                   scWithoutAllowedTopologies,
		},
		{
			description:          "StorageClass With AllowedTopologies",
			hasAllowedTopologies: true,
			sc:                   scWithAllowedTopologies,
		},
		{
			description:          "is nil",
			hasAllowedTopologies: false,
			sc:                   func() *storage.StorageClass { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldStorageClassInfo := range scInfo {
			for _, newStorageClassInfo := range scInfo {
				oldStorageClassHasAllowedTopologies, oldStorageClass := oldStorageClassInfo.hasAllowedTopologies, oldStorageClassInfo.sc()
				newStorageClassHasAllowedTopologies, newStorageClass := newStorageClassInfo.hasAllowedTopologies, newStorageClassInfo.sc()
				if newStorageClass == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old StorageClass %v, new StorageClass %v", enabled, oldStorageClassInfo.description, newStorageClassInfo.description), func(t *testing.T) {
					defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.VolumeScheduling, enabled)()

					DropDisabledFields(newStorageClass, oldStorageClass)

					// old StorageClass should never be changed
					if !reflect.DeepEqual(oldStorageClass, oldStorageClassInfo.sc()) {
						t.Errorf("old StorageClass changed: %v", diff.ObjectReflectDiff(oldStorageClass, oldStorageClassInfo.sc()))
					}

					switch {
					case enabled || oldStorageClassHasAllowedTopologies:
						// new StorageClass should not be changed if the feature is enabled, or if the old StorageClass had AllowedTopologies
						if !reflect.DeepEqual(newStorageClass, newStorageClassInfo.sc()) {
							t.Errorf("new StorageClass changed: %v", diff.ObjectReflectDiff(newStorageClass, newStorageClassInfo.sc()))
						}
					case newStorageClassHasAllowedTopologies:
						// new StorageClass should be changed
						if reflect.DeepEqual(newStorageClass, newStorageClassInfo.sc()) {
							t.Errorf("new StorageClass was not changed")
						}
						// new StorageClass should not have AllowedTopologies
						if !reflect.DeepEqual(newStorageClass, scWithoutAllowedTopologies()) {
							t.Errorf("new StorageClass had StorageClassAllowedTopologies: %v", diff.ObjectReflectDiff(newStorageClass, scWithoutAllowedTopologies()))
						}
					default:
						// new StorageClass should not need to be changed
						if !reflect.DeepEqual(newStorageClass, newStorageClassInfo.sc()) {
							t.Errorf("new StorageClass changed: %v", diff.ObjectReflectDiff(newStorageClass, newStorageClassInfo.sc()))
						}
					}
				})
			}
		}
	}
}
