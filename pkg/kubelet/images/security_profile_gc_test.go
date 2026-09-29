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
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	ctest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
	testingclock "k8s.io/utils/clock/testing"
)

func TestSecurityProfileGarbageCollect(t *testing.T) {
	const (
		maxAge  = time.Hour
		digestA = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
		digestB = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
		digestC = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
		// The runtime stores the profile of refC under digestC.
		refC = "registry.example.com/c@sha512:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
	)
	podWithProfile := func(ref string) *v1.Pod {
		return &v1.Pod{Spec: v1.PodSpec{
			SecurityContext: &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{
				Type: v1.SeccompProfileTypeOCI,
				OCI:  &v1.SecurityProfileOCI{Ref: ref},
			}},
			Containers: []v1.Container{{Name: "c"}},
		}}
	}
	digests := func(runtime *ctest.FakeRuntime) []string {
		runtime.Lock()
		defer runtime.Unlock()
		var out []string
		for _, p := range runtime.SecurityProfiles {
			out = append(out, p.Digest)
		}
		return out
	}

	tCtx := ktesting.Init(t)
	fakeRuntime := &ctest.FakeRuntime{T: t, SecurityProfiles: []*runtimeapi.SecurityProfileInfo{
		{Digest: digestA, Refs: []string{"registry.example.com/a@" + digestA}},
		{Digest: digestB, Refs: []string{"registry.example.com/b@" + digestB}},
		{Digest: digestC, Refs: []string{refC}},
	}}
	// One pod uses digestA through a reference the runtime has not listed,
	// so it matches by digest; the other uses refC, whose digest differs
	// from the one the runtime lists, so it matches by reference.
	pods := []*v1.Pod{podWithProfile("mirror.example.com/a@" + digestA), podWithProfile(refC)}
	fakeClock := testingclock.NewFakeClock(time.Now())
	manager := NewSecurityProfileGCManager(fakeRuntime, func() []*v1.Pod { return pods }, maxAge).(*securityProfileGCManager)
	manager.clock = fakeClock

	// Profiles seen for the first time are kept for maxAge, even if unused.
	require.NoError(t, manager.GarbageCollect(tCtx))
	assert.Equal(t, []string{digestA, digestB, digestC}, digests(fakeRuntime))

	fakeClock.Step(maxAge)
	require.NoError(t, manager.GarbageCollect(tCtx))
	assert.Equal(t, []string{digestA, digestB, digestC}, digests(fakeRuntime))

	// digestB is now unused for longer than maxAge and removed; the others
	// are in use.
	fakeClock.Step(time.Second)
	require.NoError(t, manager.GarbageCollect(tCtx))
	assert.Equal(t, []string{digestA, digestC}, digests(fakeRuntime))
	assert.NotContains(t, manager.lastUsed, digestB)

	// Once no pod uses them, they are removed after maxAge.
	pods = nil
	fakeClock.Step(maxAge)
	require.NoError(t, manager.GarbageCollect(tCtx))
	assert.Equal(t, []string{digestA, digestC}, digests(fakeRuntime))
	fakeClock.Step(time.Second)
	require.NoError(t, manager.GarbageCollect(tCtx))
	assert.Empty(t, digests(fakeRuntime))
	assert.Empty(t, manager.lastUsed)

	// Removal errors are returned and the profile is retried later.
	fakeRuntime.SecurityProfiles = []*runtimeapi.SecurityProfileInfo{{Digest: digestC}}
	require.NoError(t, manager.GarbageCollect(tCtx))
	fakeClock.Step(maxAge + time.Second)
	failing := &failingRemoveRuntime{FakeRuntime: fakeRuntime}
	manager.runtime = failing
	require.ErrorContains(t, manager.GarbageCollect(tCtx), "remove failed")
	assert.Equal(t, []string{digestC}, digests(fakeRuntime))
	manager.runtime = fakeRuntime
	require.NoError(t, manager.GarbageCollect(tCtx))
	assert.Empty(t, digests(fakeRuntime))
}

type failingRemoveRuntime struct {
	*ctest.FakeRuntime
}

func (*failingRemoveRuntime) RemoveSecurityProfile(context.Context, string) error {
	return errors.New("remove failed")
}

func TestSecurityProfileGarbageCollectListError(t *testing.T) {
	tCtx := ktesting.Init(t)
	fakeRuntime := &ctest.FakeRuntime{T: t, Err: errors.New("list failed")}
	manager := NewSecurityProfileGCManager(fakeRuntime, func() []*v1.Pod { return nil }, time.Minute)
	require.ErrorContains(t, manager.GarbageCollect(tCtx), "list failed")
}
