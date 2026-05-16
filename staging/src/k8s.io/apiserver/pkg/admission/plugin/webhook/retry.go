/*
Copyright The Kubernetes Authors.

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

package webhook

import (
	"context"
	"time"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	webhookutil "k8s.io/apiserver/pkg/util/webhook"
)

// admissionWebhookRetryBackoff matches the default authentication and
// authorization webhook retry backoff. The admission webhook timeout context
// bounds the total retry duration.
var admissionWebhookRetryBackoff = webhookutil.DefaultRetryBackoffWithInitialDelay(500 * time.Millisecond)

// WithAdmissionWebhookTransportErrorRetry retries admission webhook calls for
// transport-level failures where no admission response was received.
func WithAdmissionWebhookTransportErrorRetry(ctx context.Context, webhookFn func() error) error {
	return webhookutil.WithExponentialBackoff(ctx, admissionWebhookRetryBackoff, webhookFn, shouldRetryAdmissionWebhookCall)
}

func shouldRetryAdmissionWebhookCall(err error) bool {
	return utilnet.IsConnectionReset(err) || utilnet.IsProbableEOF(err) || utilnet.IsHTTP2ConnectionLost(err)
}
