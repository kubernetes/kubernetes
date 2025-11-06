/*
Copyright 2025 The Kubernetes Authors.

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

package kubelet

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/podcertificate"
)

type recordingPodCertificateManager struct {
	Namespace   string
	PodName     string
	PodUID      string
	VolumeName  string
	SourceIndex int
}

func (f *recordingPodCertificateManager) GetPodCertificateCredentialBundle(ctx context.Context, namespace, podName, podUID, volumeName string, sourceIndex int) ([]byte, []byte, error) {
	f.Namespace = namespace
	f.PodName = podName
	f.PodUID = podUID
	f.VolumeName = volumeName
	f.SourceIndex = sourceIndex

	return nil, nil, nil
}

func (f *recordingPodCertificateManager) TrackPod(ctx context.Context, pod *corev1.Pod) {}

func (f *recordingPodCertificateManager) ForgetPod(ctx context.Context, pod *corev1.Pod) {}

func (f *recordingPodCertificateManager) MetricReport() *podcertificate.MetricReport {
	return &podcertificate.MetricReport{}
}

// Check that GetPodCertificateCredentialBundle forwards its arguments in the
// correct order.  Seems excessive, but we got here because I put the arguments
// in the wrong order...
func TestGetPodCertificateCredentialBundle(t *testing.T) {
	recorder := &recordingPodCertificateManager{}

	kvh := &kubeletVolumeHost{
		podCertificateManager: recorder,
	}

	_, _, err := kvh.GetPodCertificateCredentialBundle(context.Background(), "namespace", "pod-name", "pod-uid", "volume-name", 10)
	if err != nil {
		t.Fatalf("Unexpected error calling GetPodCertificateCredentialBundle: %v", err)
	}

	want := &recordingPodCertificateManager{
		Namespace:   "namespace",
		PodName:     "pod-name",
		PodUID:      "pod-uid",
		VolumeName:  "volume-name",
		SourceIndex: 10,
	}

	if diff := cmp.Diff(recorder, want); diff != "" {
		t.Errorf("Wrong input to GetPodCertificateCredentialBundle; diff (-got +want)\n%s", diff)
	}
}
