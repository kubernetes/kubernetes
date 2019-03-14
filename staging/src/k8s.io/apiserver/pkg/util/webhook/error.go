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

package webhook

import "fmt"

// ErrCallingWebhook is returned for transport-layer errors calling webhooks. It
// represents a failure to talk to the webhook, not the webhook rejecting a
// request.
type ErrCallingWebhook struct {
	WebhookName string
	Reason      error
}

func (e *ErrCallingWebhook) Error() string {
	if e.Reason != nil {
		return fmt.Sprintf("failed calling webhook %q: %v", e.WebhookName, e.Reason)
	}
	return fmt.Sprintf("failed calling webhook %q; no further details available", e.WebhookName)
}
