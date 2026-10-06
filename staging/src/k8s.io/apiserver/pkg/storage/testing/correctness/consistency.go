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

package correctness

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/storage"
)

// ReadConsistency describes the consistency guarantee of a read operation.
type ReadConsistency string

const (
	// ConsistencyConsistent requires the read to be served from the latest resource version (quorum read).
	ConsistencyConsistent ReadConsistency = "Consistent"
	// ConsistencyExact requires the read to be served at the exact requested resource version (exact stale).
	ConsistencyExact ReadConsistency = "Exact"
	// ConsistencyNotOlderThan requires the read to be served at a resource version not older than requested (bounded stale).
	ConsistencyNotOlderThan ReadConsistency = "NotOlderThan"
)

// GetReadConsistency qualifies the read consistency of a Get operation.
func GetReadConsistency(opts storage.GetOptions) ReadConsistency {
	if opts.ResourceVersion == "" {
		return ConsistencyConsistent
	}
	return ConsistencyNotOlderThan
}

// ListReadConsistency qualifies the read consistency of a List operation.
func ListReadConsistency(opts storage.ListOptions) (ReadConsistency, error) {
	if opts.SendInitialEvents != nil {
		return "", fmt.Errorf("sendInitialEvents is forbidden for list")
	}
	if opts.ResourceVersionMatch != "" {
		if opts.ResourceVersion == "" {
			return "", fmt.Errorf("resourceVersionMatch is forbidden unless resourceVersion is provided")
		}
		if opts.Predicate.Continue != "" {
			return "", fmt.Errorf("resourceVersionMatch is forbidden when continue is provided")
		}
	}
	switch opts.ResourceVersionMatch {
	case metav1.ResourceVersionMatchExact:
		if opts.ResourceVersion == "0" {
			return "", fmt.Errorf("resourceVersionMatch \"exact\" is forbidden for resourceVersion \"0\"")
		}
		return ConsistencyExact, nil
	case metav1.ResourceVersionMatchNotOlderThan:
		return ConsistencyNotOlderThan, nil
	case "":
		if opts.Recursive && opts.Predicate.Continue != "" {
			if opts.ResourceVersion != "" && opts.ResourceVersion != "0" {
				return "", fmt.Errorf("specifying resource version is not allowed when using continue")
			}
			_, continueRV, err := storage.DecodeContinue(opts.Predicate.Continue, "")
			if err != nil {
				return "", fmt.Errorf("invalid continue token: %w", err)
			}
			if continueRV > 0 {
				return ConsistencyExact, nil
			}
			return ConsistencyConsistent, nil
		}
		// Legacy exact match for paginated recursive lists.
		if opts.Recursive && opts.Predicate.Limit > 0 && opts.ResourceVersion != "" && opts.ResourceVersion != "0" {
			return ConsistencyExact, nil
		}
		if opts.ResourceVersion == "" {
			return ConsistencyConsistent, nil
		}
		return ConsistencyNotOlderThan, nil
	default:
		return "", fmt.Errorf("unknown ResourceVersionMatch: %q", opts.ResourceVersionMatch)
	}
}
