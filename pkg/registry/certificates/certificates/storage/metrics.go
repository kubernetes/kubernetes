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

package storage

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/util/dryrun"
	"k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/certificate/csr"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/kubernetes/pkg/apis/certificates"
)

const (
	namespace = "apiserver"
	subsystem = "certificates_registry"
)

var (
	// csrDurationRequested counts and categorizes how many certificates were issued when the client requested a duration.
	csrDurationRequested = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "csr_requested_duration_total",
			Help:           "Total number of issued CSRs with a requested duration, sliced by signer (only kubernetes.io signer names are specifically identified)",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"signerName"},
	)

	// csrDurationHonored counts and categorizes how many certificates were issued when the client requested a duration and the signer honored it.
	csrDurationHonored = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "csr_honored_duration_total",
			Help:           "Total number of issued CSRs with a requested duration that was honored, sliced by signer (only kubernetes.io signer names are specifically identified)",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"signerName"},
	)
)

func init() {
	registerMetricsOnce.Do(func() {
		legacyregistry.MustRegister(csrDurationRequested)
		legacyregistry.MustRegister(csrDurationHonored)
	})
}

var registerMetricsOnce sync.Once

type counterVecMetric interface {
	WithLabelValues(...string) metrics.CounterMetric
}

func countCSRDurationMetric(requested, honored counterVecMetric) genericregistry.BeginUpdateFunc {
	return func(ctx context.Context, obj, old runtime.Object, options *metav1.UpdateOptions) (genericregistry.FinishFunc, error) {
		return func(ctx context.Context, success bool) {
			if !success {
				return // ignore failures
			}

			if dryrun.IsDryRun(options.DryRun) {
				return // ignore things that would not get persisted
			}

			oldCSR, ok := old.(*certificates.CertificateSigningRequest)
			if !ok {
				return
			}

			// if the old CSR already has a certificate, do not double count it
			if len(oldCSR.Status.Certificate) > 0 {
				return
			}

			if oldCSR.Spec.ExpirationSeconds == nil {
				return // ignore CSRs that are not using the CSR duration feature
			}

			newCSR, ok := obj.(*certificates.CertificateSigningRequest)
			if !ok {
				return
			}
			issuedCert := newCSR.Status.Certificate

			// new CSR has no issued certificate yet so do not count it.
			// note that this means that we will ignore CSRs that set a duration
			// but never get approved/signed.  this is fine because the point
			// of these metrics is to understand if the duration is honored
			// by the signer.  we are not checking the behavior of the approver.
			if len(issuedCert) == 0 {
				return
			}

			signer := compressSignerName(oldCSR.Spec.SignerName)

			// at this point we know that this CSR is going to be persisted and
			// the cert was just issued and the client requested a duration
			requested.WithLabelValues(signer).Inc()

			certs, err := cert.ParseCertsPEM(issuedCert)
			if err != nil {
				utilruntime.HandleError(fmt.Errorf("metrics recording failed to parse certificate for CSR %s: %w", oldCSR.Name, err))
				return
			}

			// now we check to see if the signer honored the requested duration
			certificate := certs[0]
			wantDuration := csr.ExpirationSecondsToDuration(*oldCSR.Spec.ExpirationSeconds)
			actualDuration := certificate.NotAfter.Sub(certificate.NotBefore)
			if isDurationHonored(wantDuration, actualDuration) {
				honored.WithLabelValues(signer).Inc()
			}
		}, nil
	}
}

func isDurationHonored(want, got time.Duration) bool {
	delta := want - got
	if delta < 0 {
		delta = -delta
	}

	// short-lived cert backdating + 5% of want
	maxDelta := 5*time.Minute + (want / 20)

	return delta < maxDelta
}

func compressSignerName(name string) string {
	if strings.HasPrefix(name, "kubernetes.io/") {
		return name
	}

	return "other"
}
