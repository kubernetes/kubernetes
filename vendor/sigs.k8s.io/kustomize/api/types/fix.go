// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

import (
	"log"
	"regexp"

	"sigs.k8s.io/yaml"
)

// FixKustomizationPreUnmarshalling modies the raw data
// before marshalling - e.g. changes old field names to
// new field names.
func FixKustomizationPreUnmarshalling(data []byte) []byte {
	deprecateFieldsMap := map[string]string{
		"imageTags:": "images:",
	}
	for oldname, newname := range deprecateFieldsMap {
		pattern := regexp.MustCompile(oldname)
		data = pattern.ReplaceAll(data, []byte(newname))
	}
	if useLegacyPatch(data) {
		pattern := regexp.MustCompile("patches:")
		data = pattern.ReplaceAll(data, []byte("patchesStrategicMerge:"))
	}
	return data
}

func useLegacyPatch(data []byte) bool {
	found := false
	var object map[string]interface{}
	err := yaml.Unmarshal(data, &object)
	if err != nil {
		log.Fatalf("invalid content from %s\n", string(data))
	}
	if rawPatches, ok := object["patches"]; ok {
		patches, ok := rawPatches.([]interface{})
		if !ok {
			log.Fatalf("invalid patches from %v\n", rawPatches)
		}
		for _, p := range patches {
			_, ok := p.(string)
			if ok {
				found = true
			}
		}
	}
	return found
}
