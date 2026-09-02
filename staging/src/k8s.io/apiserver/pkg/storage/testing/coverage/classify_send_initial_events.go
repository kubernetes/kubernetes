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

package coverage

// SendInitialEventsMode classifies ListOptions.SendInitialEvents.
type SendInitialEventsMode string

const (
	SendInitialEventsUnset SendInitialEventsMode = "SendInitialEventsUnset"
	SendInitialEventsTrue  SendInitialEventsMode = "SendInitialEventsTrue"
	SendInitialEventsFalse SendInitialEventsMode = "SendInitialEventsFalse"
)

func classifySendInitialEvents(b *bool) SendInitialEventsMode {
	switch {
	case b == nil:
		return SendInitialEventsUnset
	case *b:
		return SendInitialEventsTrue
	default:
		return SendInitialEventsFalse
	}
}
