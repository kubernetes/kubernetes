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

package collectors

import (
	"context"
	"fmt"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/kubernetes/pkg/kubelet/podcertificate"
)

type fakePodCertificateManager struct {
	report *podcertificate.MetricReport
}

func (m *fakePodCertificateManager) TrackPod(ctx context.Context, pod *corev1.Pod)  {}
func (m *fakePodCertificateManager) ForgetPod(ctx context.Context, pod *corev1.Pod) {}
func (m *fakePodCertificateManager) GetPodCertificateCredentialBundle(ctx context.Context, namespace, podName, podUID, volumeName string, sourceIndex int) (privKey []byte, certChain []byte, err error) {
	return nil, nil, fmt.Errorf("unimplemented")
}
func (m *fakePodCertificateManager) MetricReport() *podcertificate.MetricReport {
	return m.report
}

func TestPodCertificateCollector(t *testing.T) {
	collector := &podCertificateCollector{
		manager: &fakePodCertificateManager{
			report: &podcertificate.MetricReport{
				PodCertificateStates: map[podcertificate.SignerAndState]int{
					{SignerName: "example.com/foo", State: "fresh"}:               1,
					{SignerName: "example.com/foo", State: "overdue_for_refresh"}: 1,
					{SignerName: "example.com/foo", State: "expired"}:             0,
					{SignerName: "example.com/bar", State: "fresh"}:               2,
					{SignerName: "example.com/bar", State: "overdue_for_refresh"}: 0,
					{SignerName: "example.com/bar", State: "expired"}:             1,
				},
			},
		},
	}

	// Fixed metadata on type and help text. We prepend this to every expected
	// output so we only have to modify a single place when doing adjustments.
	const metadata = `
		# HELP kubelet_podcertificate_states [ALPHA] Gauge vector reporting the number of pod certificate projected volume sources, faceted by signer_name and state.
		# TYPE kubelet_podcertificate_states gauge
		`

	want := metadata + `
			kubelet_podcertificate_states{signer_name="example.com/foo",state="fresh"} 1.0
			kubelet_podcertificate_states{signer_name="example.com/foo",state="overdue_for_refresh"} 1.0
			kubelet_podcertificate_states{signer_name="example.com/foo",state="expired"} 0.0
			kubelet_podcertificate_states{signer_name="example.com/bar",state="fresh"} 2.0
			kubelet_podcertificate_states{signer_name="example.com/bar",state="overdue_for_refresh"} 0.0
			kubelet_podcertificate_states{signer_name="example.com/bar",state="expired"} 1.0
			`

	metrics := []string{
		"kubelet_podcertificate_states",
	}

	if err := testutil.CustomCollectAndCompare(collector, strings.NewReader(want), metrics...); err != nil {
		t.Errorf("unexpected collecting result:\n%s", err)
	}
}
