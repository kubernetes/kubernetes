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
	"fmt"
	"regexp"
	"strings"

	"sigs.k8s.io/kustomize/pkg/image"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/transformers/config"
)

// imageTransformer replace image names and tags
type imageTransformer struct {
	images     []image.Image
	fieldSpecs []config.FieldSpec
}

var _ Transformer = &imageTransformer{}

// NewImageTransformer constructs an imageTransformer.
func NewImageTransformer(slice []image.Image, fs []config.FieldSpec) (Transformer, error) {
	return &imageTransformer{slice, fs}, nil
}

// Transform finds the matching images and replaces name, tag and/or digest
func (pt *imageTransformer) Transform(m resmap.ResMap) error {
	if len(pt.images) == 0 {
		return nil
	}
	for _, r := range m.Resources() {
		for _, path := range pt.fieldSpecs {
			if !r.OrgId().IsSelected(&path.Gvk) {
				continue
			}
			err := MutateField(r.Map(), path.PathSlice(), false, pt.mutateImage)
			if err != nil {
				return err
			}
		}
		// Kept for backward compatibility
		if err := pt.findAndReplaceImage(r.Map()); err != nil && r.OrgId().Kind != `CustomResourceDefinition` {
			return err
		}
	}
	return nil
}

func (pt *imageTransformer) mutateImage(in interface{}) (interface{}, error) {
	original, ok := in.(string)
	if !ok {
		return nil, fmt.Errorf("image path is not of type string but %T", in)
	}

	for _, img := range pt.images {
		if !isImageMatched(original, img.Name) {
			continue
		}
		name, tag := split(original)
		if img.NewName != "" {
			name = img.NewName
		}
		if img.NewTag != "" {
			tag = ":" + img.NewTag
		}
		if img.Digest != "" {
			tag = "@" + img.Digest
		}
		return name + tag, nil
	}
	return original, nil
}

/*
 findAndReplaceImage replaces the image name and tags inside one object
 It searches the object for container session
 then loops though all images inside containers session,
 finds matched ones and update the image name and tag name
*/
func (pt *imageTransformer) findAndReplaceImage(obj map[string]interface{}) error {
	paths := []string{"containers", "initContainers"}
	updated := false
	for _, path := range paths {
		containers, found := obj[path]
		if found {
			if _, err := pt.updateContainers(containers); err != nil {
				return err
			}
			updated = true
		}
	}
	if !updated {
		return pt.findContainers(obj)
	}
	return nil
}

func (pt *imageTransformer) updateContainers(in interface{}) (interface{}, error) {
	containers, ok := in.([]interface{})
	if !ok {
		return nil, fmt.Errorf("containers path is not of type []interface{} but %T", in)
	}
	for i := range containers {
		container := containers[i].(map[string]interface{})
		containerImage, found := container["image"]
		if !found {
			continue
		}

		imageName := containerImage.(string)
		for _, img := range pt.images {
			if !isImageMatched(imageName, img.Name) {
				continue
			}
			newImage, err := pt.mutateImage(imageName)
			if err != nil {
				return nil, err
			}
			container["image"] = newImage
			break
		}
	}
	return containers, nil
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
	// check if image name contains a domain
	// if domain is present, ignore domain and check for `:`
	ic := -1
	if slashIndex := strings.Index(imageName, "/"); slashIndex < 0 {
		ic = strings.LastIndex(imageName, ":")
	} else {
		lastIc := strings.LastIndex(imageName[slashIndex:], ":")
		// set ic only if `:` is present
		if lastIc > 0 {
			ic = slashIndex + lastIc
		}
	}
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
