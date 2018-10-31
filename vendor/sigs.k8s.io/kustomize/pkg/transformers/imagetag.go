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

package transformers

import (
	"regexp"

	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/types"
)

// imageTagTransformer replace image tags
type imageTagTransformer struct {
	imageTags []types.ImageTag
}

var _ Transformer = &imageTagTransformer{}

// NewImageTagTransformer constructs a imageTagTransformer.
func NewImageTagTransformer(slice []types.ImageTag) (Transformer, error) {
	return &imageTagTransformer{slice}, nil
}

// Transform finds the matching images and replace the tag
func (pt *imageTagTransformer) Transform(resources resmap.ResMap) error {
	if len(pt.imageTags) == 0 {
		return nil
	}
	for _, res := range resources {
		err := pt.findAndReplaceTag(res.Map())
		if err != nil {
			return err
		}
	}
	return nil
}

/*
 findAndReplaceTag replaces the image tags inside one object
 It searches the object for container session
 then loops though all images inside containers session, finds matched ones and update the tag name
*/
func (pt *imageTagTransformer) findAndReplaceTag(obj map[string]interface{}) error {
	paths := []string{"containers", "initContainers"}
	found := false
	for _, path := range paths {
		_, found = obj[path]
		if found {
			err := pt.updateContainers(obj, path)
			if err != nil {
				return err
			}
		}
	}
	if !found {
		return pt.findContainers(obj)
	}
	return nil
}

func (pt *imageTagTransformer) updateContainers(obj map[string]interface{}, path string) error {
	containers := obj[path].([]interface{})
	for i := range containers {
		container := containers[i].(map[string]interface{})
		image, found := container["image"]
		if !found {
			continue
		}
		for _, imagetag := range pt.imageTags {
			if isImageMatched(image.(string), imagetag.Name) {
				container["image"] = imagetag.Name + ":" + imagetag.NewTag

				if imagetag.Digest != "" {
					container["image"] = imagetag.Name + "@" + imagetag.Digest
				}

				break
			}
		}
	}
	return nil
}

func (pt *imageTagTransformer) findContainers(obj map[string]interface{}) error {
	for key := range obj {
		switch typedV := obj[key].(type) {
		case map[string]interface{}:
			err := pt.findAndReplaceTag(typedV)
			if err != nil {
				return err
			}
		case []interface{}:
			for i := range typedV {
				item := typedV[i]
				typedItem, ok := item.(map[string]interface{})
				if ok {
					err := pt.findAndReplaceTag(typedItem)
					if err != nil {
						return err
					}
				}
			}
		}
	}
	return nil
}

func isImageMatched(s, t string) bool {
	// Tag values are limited to [a-zA-Z0-9_.-].
	pattern, _ := regexp.Compile("^" + t + "(:[a-zA-Z0-9_.-]*)?$")
	return pattern.MatchString(s)
}
