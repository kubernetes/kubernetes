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

package images

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/utils/clock"
)

// DefaultSecurityProfileMaxAge is how long a pulled security profile may stay
// unused before it is removed, unless the image maximum GC age is set.
const DefaultSecurityProfileMaxAge = 24 * time.Hour

// SecurityProfileGCManager removes pulled security profiles that no pod uses.
type SecurityProfileGCManager interface {
	// GarbageCollect removes the security profiles that no pod has used for
	// longer than the maximum age.
	GarbageCollect(ctx context.Context) error
}

type securityProfileGCManager struct {
	runtime kubecontainer.ImageService
	// getPods returns the pods whose security profiles are in use.
	getPods func() []*v1.Pod
	// maxAge is how long a profile may stay unused before it is removed.
	maxAge time.Duration
	clock  clock.PassiveClock

	// lastUsed maps the digests of the listed profiles to the last time a pod
	// used them, or to the time they were first listed. It is only accessed
	// by GarbageCollect, which is not called concurrently.
	lastUsed map[string]time.Time
}

// NewSecurityProfileGCManager returns a SecurityProfileGCManager that removes
// profiles none of the pods returned by getPods have used for longer than
// maxAge.
func NewSecurityProfileGCManager(runtime kubecontainer.ImageService, getPods func() []*v1.Pod, maxAge time.Duration) SecurityProfileGCManager {
	return &securityProfileGCManager{
		runtime:  runtime,
		getPods:  getPods,
		maxAge:   maxAge,
		clock:    clock.RealClock{},
		lastUsed: map[string]time.Time{},
	}
}

func (m *securityProfileGCManager) GarbageCollect(ctx context.Context) error {
	logger := klog.FromContext(ctx)

	profiles, err := m.runtime.ListSecurityProfiles(ctx)
	if err != nil {
		return fmt.Errorf("failed to list security profiles: %w", err)
	}

	// The pods are read after listing, so a profile pulled for a pod added
	// in between is either not listed yet or seen in use. A profile is in use
	// if a pod references one of its refs or its digest; the runtime may
	// store it under a digest of another algorithm than the pod's reference.
	inUse := sets.New[string]()
	for _, pod := range m.getPods() {
		for _, ref := range kubecontainer.SecurityProfileOCIRefs(pod) {
			inUse.Insert(ref)
			// References are canonical and digest-pinned.
			if _, digest, ok := strings.Cut(ref, "@"); ok {
				inUse.Insert(digest)
			}
		}
	}

	now := m.clock.Now()
	listed := sets.New[string]()
	var errs []error
	for _, profile := range profiles {
		digest := profile.GetDigest()
		lastUsed, known := m.lastUsed[digest]
		if inUse.Has(digest) || slices.ContainsFunc(profile.GetRefs(), inUse.Has) || !known {
			m.lastUsed[digest] = now
			listed.Insert(digest)
			continue
		}
		if now.Sub(lastUsed) <= m.maxAge {
			listed.Insert(digest)
			continue
		}
		if err := m.runtime.RemoveSecurityProfile(ctx, digest); err != nil {
			errs = append(errs, fmt.Errorf("failed to remove security profile %q: %w", digest, err))
			listed.Insert(digest)
			continue
		}
		logger.V(2).Info("Removed unused security profile", "digest", digest, "refs", profile.GetRefs(), "size", profile.GetSize())
	}

	// Forget the profiles that are gone.
	for digest := range m.lastUsed {
		if !listed.Has(digest) {
			delete(m.lastUsed, digest)
		}
	}
	return utilerrors.NewAggregate(errs)
}
