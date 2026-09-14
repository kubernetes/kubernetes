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

package workloadbuilder

import (
	"fmt"
	"hash/fnv"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/utils/dump"
)

// GenerateWorkloadName returns a deterministic, DNS-compatible name for a
// Workload. ownerName is used as a prefix when it is non-empty and should be a
// DNS-1123 subdomain. The UID is included in the hash because object names can
// be reused after deletion, while the name remains useful to humans as a
// prefix.
func GenerateWorkloadName(ownerName string, ownerUID types.UID) string {
	hasher := fnv.New32a()
	// Use the same canonical representation as Kubernetes' DeepHashObject so
	// moving this helper does not change names already persisted by controllers.
	_, _ = fmt.Fprintf(hasher, "%v", dump.ForHash(ownerUID))
	hash := rand.SafeEncodeString(fmt.Sprint(hasher.Sum32()))

	if ownerName == "" {
		return hash
	}

	maxPrefixLen := validation.DNS1123SubdomainMaxLength - len(hash) - 1
	if len(ownerName) > maxPrefixLen {
		ownerName = ownerName[:maxPrefixLen]
	}
	return fmt.Sprintf("%s-%s", ownerName, hash)
}

// GeneratePodGroupName returns a deterministic, DNS-compatible name derived
// from a Workload and its PodGroupTemplate. workloadName and templateName must
// be non-empty DNS-1123 subdomains. Both names are retained in the prefix where
// possible; the hash prevents collisions when truncation makes different
// inputs share a prefix.
func GeneratePodGroupName(workloadName, templateName string) string {
	hasher := fnv.New32a()
	_, _ = hasher.Write([]byte(workloadName))
	_, _ = hasher.Write([]byte(templateName))
	hash := rand.SafeEncodeString(fmt.Sprint(hasher.Sum32()))

	maxPrefixLen := validation.DNS1123SubdomainMaxLength - len(hash) - 1
	if workloadName == "" {
		if templateName == "" {
			return hash
		}
		if len(templateName) > maxPrefixLen {
			templateName = templateName[:maxPrefixLen]
		}
		return fmt.Sprintf("%s-%s", templateName, hash)
	}
	if templateName == "" {
		if len(workloadName) > maxPrefixLen {
			workloadName = workloadName[:maxPrefixLen]
		}
		return fmt.Sprintf("%s-%s", workloadName, hash)
	}

	maxAvailable := validation.DNS1123SubdomainMaxLength - len(hash) - 2
	wl, tpl := workloadName, templateName
	if len(wl)+len(tpl) > maxAvailable {
		half := maxAvailable / 2
		switch {
		case len(wl) <= half:
			tpl = tpl[:maxAvailable-len(wl)]
		case len(tpl) <= half:
			wl = wl[:maxAvailable-len(tpl)]
		default:
			wl = wl[:maxAvailable-half]
			tpl = tpl[:half]
		}
	}
	return fmt.Sprintf("%s-%s-%s", wl, tpl, hash)
}
