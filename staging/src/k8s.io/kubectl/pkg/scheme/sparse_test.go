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

package scheme

import (
	"testing"

	"k8s.io/api/batch/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
)

func TestCronJob(t *testing.T) {
	src := &v1.CronJob{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}

	encoder := Codecs.LegacyCodec(v1.SchemeGroupVersion)
	cronjobBytes, err := runtime.Encode(encoder, src)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(string(cronjobBytes))
	t.Log(Scheme.PrioritizedVersionsAllGroups())

	decoder := Codecs.UniversalDecoder(Scheme.PrioritizedVersionsAllGroups()...)

	uncastDst, err := runtime.Decode(decoder, cronjobBytes)
	if err != nil {
		t.Fatal(err)
	}

	// clear typemeta
	uncastDst.(*v1.CronJob).TypeMeta = metav1.TypeMeta{}

	if !equality.Semantic.DeepEqual(src, uncastDst) {
		t.Fatal(diff.ObjectDiff(src, uncastDst))
	}
}
