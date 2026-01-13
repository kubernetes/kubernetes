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

package nodedeclaredfeatures

import (
	"fmt"
	"slices"
)

// FeatureSet is a set of node features.
type FeatureSet = bitmap

// FeatureMapper maps feature names to bit positions in a FeatureSet.
type FeatureMapper struct {
	registeredFeatures []string
}

// NewFeatureMapper creates a FeatureMapper from a list of known features.
func NewFeatureMapper(features []string) *FeatureMapper {
	sortedFeatures := slices.Clone(features)
	slices.Sort(sortedFeatures)
	return &FeatureMapper{sortedFeatures}
}

// NewFeatureSet returns an empty FeatureSet sized to the registered features.
func (m *FeatureMapper) NewFeatureSet() FeatureSet {
	return FeatureSet(newBitmap(len(m.registeredFeatures)))
}

// MapSorted creates a FeatureSet from a sorted slice of feature names.
func (m *FeatureMapper) MapSorted(sortedFeatures []string) (FeatureSet, error) {
	return m.mapSorted(sortedFeatures, false)
}

// MustMapSorted is a convenience wrapper around MapSorted that panics on errors.
func (m *FeatureMapper) MustMapSorted(sortedFeatures []string) FeatureSet {
	s, err := m.MapSorted(sortedFeatures)
	if err != nil {
		panic(err)
	}
	return s
}

// TryMap creates a FeatureSet from a sorted slice, ignoring unknown features.
func (m *FeatureMapper) TryMap(sortedFeatures []string) FeatureSet {
	fs, _ := m.mapSorted(sortedFeatures, true)
	return fs
}

func (m *FeatureMapper) mapSorted(sortedFeatures []string, ignoreUnknown bool) (FeatureSet, error) {
	s := m.NewFeatureSet()

	if len(sortedFeatures) == 0 {
		return s, nil
	}

	i, j := 0, 0
	for i < len(sortedFeatures) && j < len(m.registeredFeatures) {
		if sortedFeatures[i] == m.registeredFeatures[j] {
			s.Set(j)
			i++
			j++
		} else if sortedFeatures[i] < m.registeredFeatures[j] {
			if !ignoreUnknown {
				return s, fmt.Errorf("unknown feature %s", sortedFeatures[i])
			}
			i++
		} else {
			j++
		}
	}

	if !ignoreUnknown && i < len(sortedFeatures) {
		return s, fmt.Errorf("unknown feature %s", sortedFeatures[i])
	}
	return s, nil
}

// Unmap returns the names of the features set in the FeatureSet (sorted).
func (m *FeatureMapper) Unmap(s FeatureSet) []string {
	if s.IsEmpty() {
		return nil
	}

	var keys []string
	for i, k := range m.registeredFeatures {
		if s.Get(i) {
			keys = append(keys, k)
		}
	}
	return keys
}
