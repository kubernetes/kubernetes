// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

import (
	"regexp"

	"sigs.k8s.io/yaml"
)

// FixKustomizationPreUnmarshalling modifies the raw data
// before marshalling - e.g. changes old field names to
// new field names.
func FixKustomizationPreUnmarshalling(data []byte) ([]byte, error) {
	deprecatedFieldsMap := map[string]string{
		"imageTags:": "images:",
	}
	for oldname, newname := range deprecatedFieldsMap {
		pattern := regexp.MustCompile(oldname)
		data = pattern.ReplaceAll(data, []byte(newname))
	}
	doLegacy, err := useLegacyPatch(data)
	if err != nil {
		return nil, err
	}
	if doLegacy {
		pattern := regexp.MustCompile("patches:")
		data = pattern.ReplaceAll(data, []byte("patchesStrategicMerge:"))
	}
	return data, nil
}

func useLegacyPatch(data []byte) (bool, error) {
	found := false
	var object map[string]interface{}
	err := yaml.Unmarshal(data, &object)
	if err != nil {
		return false, err
	}
	if rawPatches, ok := object["patches"]; ok {
		patches, ok := rawPatches.([]interface{})
		if !ok {
			return false, err
		}
		for _, p := range patches {
			_, ok := p.(string)
			if ok {
				found = true
			}
		}
	}
	return found, nil
}
