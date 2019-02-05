/*
Copyright 2019 The Kubernetes Authors.

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
	"strings"

	"sigs.k8s.io/kustomize/pkg/image"
	"sigs.k8s.io/kustomize/pkg/resmap"
)

// imageTransformer replace image names and tags
type imageTransformer struct {
	images []image.Image
}

var _ Transformer = &imageTransformer{}

// NewImageTransformer constructs an imageTransformer.
func NewImageTransformer(slice []image.Image) (Transformer, error) {
	return &imageTransformer{slice}, nil
}

// Transform finds the matching images and replaces name, tag and/or digest
func (pt *imageTransformer) Transform(resources resmap.ResMap) error {
	if len(pt.images) == 0 {
		return nil
	}
	for _, res := range resources {
		err := pt.findAndReplaceImage(res.Map())
		if err != nil {
			return err
		}
	}
	return nil
}

/*
 findAndReplaceImage replaces the image name and tags inside one object
 It searches the object for container session
 then loops though all images inside containers session,
 finds matched ones and update the image name and tag name
*/
func (pt *imageTransformer) findAndReplaceImage(obj map[string]interface{}) error {
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

func (pt *imageTransformer) updateContainers(obj map[string]interface{}, path string) error {
	containers := obj[path].([]interface{})
	for i := range containers {
		container := containers[i].(map[string]interface{})
		containerImage, found := container["image"]
		if !found {
			continue
		}

		imageName := containerImage.(string)
		for _, img := range pt.images {
			if isImageMatched(imageName, img.Name) {
				name, tag := split(imageName)
				if img.NewName != "" {
					name = img.NewName
				}
				if img.NewTag != "" {
					tag = ":" + img.NewTag
				}
				if img.Digest != "" {
					tag = "@" + img.Digest
				}
				container["image"] = name + tag
				break
			}
		}
	}
	return nil
}

func (pt *imageTransformer) findContainers(obj map[string]interface{}) error {
	for key := range obj {
		switch typedV := obj[key].(type) {
		case map[string]interface{}:
			err := pt.findAndReplaceImage(typedV)
			if err != nil {
				return err
			}
		case []interface{}:
			for i := range typedV {
				item := typedV[i]
				typedItem, ok := item.(map[string]interface{})
				if ok {
					err := pt.findAndReplaceImage(typedItem)
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

// split separates and returns the name and tag parts
// from the image string using either colon `:` or at `@` separators.
// Note that the returned tag keeps its separator.
func split(imageName string) (name string, tag string) {
	ic := strings.LastIndex(imageName, ":")
	ia := strings.LastIndex(imageName, "@")
	if ic < 0 && ia < 0 {
		return imageName, ""
	}

	i := ic
	if ic < 0 {
		i = ia
	}

	name = imageName[:i]
	tag = imageName[i:]
	return
}
