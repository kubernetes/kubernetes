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

package tags

import "testing"

const TestAnnotationAdditionalTags = "volume.beta.kubernetes.io/additional-tags-test"

func TestGetAdditionalTagsFromAnnotation(t *testing.T) {
	tagTests := []struct {
		Annotations map[string]string
		Tags        map[string]string
	}{
		{
			Annotations: map[string]string{
				TestAnnotationAdditionalTags: "Key=Val",
			},
			Tags: map[string]string{
				"Key": "Val",
			},
		},
		{
			Annotations: map[string]string{
				TestAnnotationAdditionalTags: "Key1=Val1, Key2=Val2",
			},
			Tags: map[string]string{
				"Key1": "Val1",
				"Key2": "Val2",
			},
		},
		{
			Annotations: map[string]string{
				TestAnnotationAdditionalTags: "Key1=, Key2=Val2",
				"anotherKey":                 "anotherValue",
			},
			Tags: map[string]string{
				"Key1": "",
				"Key2": "Val2",
			},
		},
		{
			Annotations: map[string]string{
				"Nothing": "Key1=, Key2=Val2, Key3",
			},
			Tags: map[string]string{},
		},
		{
			Annotations: map[string]string{
				TestAnnotationAdditionalTags: "K=V K1=V2,Key1========, =====, ======Val, =Val, , 234,",
			},
			Tags: map[string]string{
				"K":    "V K1",
				"Key1": "",
				"234":  "",
			},
		},
	}

	for _, tagTest := range tagTests {
		result := GetAdditionalTagsFromAnnotation(tagTest.Annotations, TestAnnotationAdditionalTags)
		for k, v := range result {
			if len(result) != len(tagTest.Tags) {
				t.Errorf("incorrect expected length: %v != %v", result, tagTest.Tags)
				continue
			}
			if tagTest.Tags[k] != v {
				t.Errorf("%s != %s", tagTest.Tags[k], v)
				continue
			}
		}
	}
}
