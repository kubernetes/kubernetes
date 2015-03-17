/*
Copyright 2015 Google Inc. All rights reserved.

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

package autoscaler

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

func TestMatchAutoScaler(t *testing.T) {
	autoScaler := &api.AutoScaler{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
			Labels: map[string]string{
				"labelTest": "bar",
			},
		},
	}

	//invalid object type
	matcher := MatchAutoScaler(labels.Everything(), fields.Everything())
	if match, _ := matcher.Matches(&api.Pod{}); match {
		t.Errorf("Did not expect match")
	}

	//invalid field match
	nameSelector, _ := fields.ParseSelector("name=foobar")
	matcher = MatchAutoScaler(labels.Everything(), nameSelector)
	if match, _ := matcher.Matches(autoScaler); match {
		t.Errorf("Did not expect match")
	}

	//valid field match
	nameSelector, _ = fields.ParseSelector("name=foo")
	matcher = MatchAutoScaler(labels.Everything(), nameSelector)
	if match, _ := matcher.Matches(autoScaler); !match {
		t.Errorf("Expected match")
	}

	//invalid label match
	labelSelector, _ := labels.ParseSelector("labelTest=foo")
	matcher = MatchAutoScaler(labelSelector, fields.Everything())
	if match, _ := matcher.Matches(autoScaler); match {
		t.Errorf("Did not expect match")
	}

	//valid label match
	labelSelector, _ = labels.ParseSelector("labelTest=bar")
	matcher = MatchAutoScaler(labelSelector, fields.Everything())
	if match, _ := matcher.Matches(autoScaler); !match {
		t.Errorf("Expected match")
	}
}
