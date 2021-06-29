/*
Copyright 2021 The Kubernetes Authors.

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

package podsecurity

import (
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/core"
	podsecurityadmission "k8s.io/pod-security-admission/admission"
)

func TestConvert(t *testing.T) {
	extractor := podsecurityadmission.DefaultPodSpecExtractor{}
	internalTypes := map[schema.GroupResource]runtime.Object{
		core.Resource("pods"):                   &core.Pod{},
		core.Resource("replicationcontrollers"): &core.ReplicationController{},
		core.Resource("podtemplates"):           &core.PodTemplate{},
		apps.Resource("replicasets"):            &apps.ReplicaSet{},
		apps.Resource("deployments"):            &apps.Deployment{},
		apps.Resource("statefulsets"):           &apps.StatefulSet{},
		apps.Resource("daemonsets"):             &apps.DaemonSet{},
		batch.Resource("jobs"):                  &batch.Job{},
		batch.Resource("cronjobs"):              &batch.CronJob{},
	}
	for _, r := range extractor.PodSpecResources() {
		internalType, ok := internalTypes[r]
		if !ok {
			t.Errorf("no internal type registered for %s", r.String())
			continue
		}
		externalType, err := convert(internalType)
		if err != nil {
			t.Errorf("error converting %T: %v", internalType, err)
			continue
		}
		_, _, err = extractor.ExtractPodSpec(externalType)
		if err != nil {
			t.Errorf("error extracting from %T: %v", externalType, err)
			continue
		}
	}
}
