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

	apierrors "k8s.io/apimachinery/pkg/api/errors"
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

// ListReadConsistency qualifies the read consistency of a List operation,
// validates ListOptions, and parses ResourceVersion and Continue token.
func ListReadConsistency(keyPrefix string, versioner storage.Versioner, opts storage.ListOptions) (consistency ReadConsistency, rv uint64, continueKey string, err error) {
	if opts.SendInitialEvents != nil {
		return "", 0, "", fmt.Errorf("sendInitialEvents is forbidden for list")
	}
	if opts.ResourceVersionMatch != "" {
		if opts.ResourceVersion == "" {
			return "", 0, "", fmt.Errorf("resourceVersionMatch is forbidden unless resourceVersion is provided")
		}
		if opts.Predicate.Continue != "" {
			return "", 0, "", fmt.Errorf("resourceVersionMatch is forbidden when continue is provided")
		}
	}
	if opts.Recursive && len(opts.Predicate.Continue) > 0 {
		continueKey, continueRV, err := storage.DecodeContinue(opts.Predicate.Continue, keyPrefix)
		if err != nil {
			return "", 0, "", apierrors.NewBadRequest(fmt.Sprintf("invalid continue token: %v", err))
		}
		if len(opts.ResourceVersion) > 0 && opts.ResourceVersion != "0" {
			return "", 0, "", apierrors.NewBadRequest("specifying resource version is not allowed when using continue")
		}
		// If continueRV > 0, the LIST request needs a specific resource version.
		// continueRV==0 is invalid.
		// If continueRV < 0, the request is for the latest resource version.
		if continueRV > 0 {
			return ConsistencyExact, uint64(continueRV), continueKey, nil
		}
		return ConsistencyConsistent, 0, continueKey, nil
	}
	if len(opts.ResourceVersion) == 0 {
		return ConsistencyConsistent, 0, "", nil
	}
	parsedRV, err := versioner.ParseResourceVersion(opts.ResourceVersion)
	if err != nil {
		return "", 0, "", apierrors.NewBadRequest(fmt.Sprintf("invalid resource version: %v", err))
	}
	switch opts.ResourceVersionMatch {
	case metav1.ResourceVersionMatchNotOlderThan:
		return ConsistencyNotOlderThan, parsedRV, "", nil
	case metav1.ResourceVersionMatchExact:
		if parsedRV == 0 {
			return "", 0, "", fmt.Errorf("resourceVersionMatch \"exact\" is forbidden for resourceVersion \"0\"")
		}
		return ConsistencyExact, parsedRV, "", nil
	case "": // legacy case
		if opts.Recursive && opts.Predicate.Limit > 0 && parsedRV > 0 {
			return ConsistencyExact, parsedRV, "", nil
		}
		return ConsistencyNotOlderThan, parsedRV, "", nil
	default:
		return "", 0, "", fmt.Errorf("unknown ResourceVersionMatch value: %v", opts.ResourceVersionMatch)
	}
}
