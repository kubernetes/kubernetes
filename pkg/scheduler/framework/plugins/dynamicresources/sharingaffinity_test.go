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

package dynamicresources

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/cel/environment"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	draapi "k8s.io/dynamic-resource-allocation/api"
	"k8s.io/dynamic-resource-allocation/cel/sharingaffinity"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

const (
	sharingAffinityTestDriver = "dra.example.com"
	sharingAffinityTestPool   = "pool"
)

type sharingAffinityTestLister struct {
	mutex  sync.Mutex
	slices []*resourceapi.ResourceSlice
	err    error
	calls  int
}

func (l *sharingAffinityTestLister) ListPatchedResourceSlices() ([]*resourceapi.ResourceSlice, error) {
	l.mutex.Lock()
	defer l.mutex.Unlock()
	l.calls++
	return slices.Clone(l.slices), l.err
}

func (l *sharingAffinityTestLister) setSlices(s ...*resourceapi.ResourceSlice) {
	l.mutex.Lock()
	defer l.mutex.Unlock()
	l.slices = slices.Clone(s)
}

func (l *sharingAffinityTestLister) callCount() int {
	l.mutex.Lock()
	defer l.mutex.Unlock()
	return l.calls
}

func sharingAffinityTestExtractor(name, expression string) resourceapi.SharingAffinityExtractor {
	return resourceapi.SharingAffinityExtractor{Name: name, APIGroup: "*", Kind: "*", Expression: expression}
}

func sharingAffinityTestSlice(name, driver, pool string, generation, count int64, extractors ...resourceapi.SharingAffinityExtractor) *resourceapi.ResourceSlice {
	return &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: resourceapi.ResourceSliceSpec{
			Driver: driver,
			Pool: resourceapi.ResourcePool{
				Name: pool, Generation: generation, ResourceSliceCount: count,
			},
			SharingAffinity: extractors,
		},
	}
}

func sharingAffinityTestConfig(driver, raw string, requests ...string) resourceapi.DeviceClaimConfiguration {
	return resourceapi.DeviceClaimConfiguration{
		Requests: requests,
		DeviceConfiguration: resourceapi.DeviceConfiguration{
			Opaque: &resourceapi.OpaqueDeviceConfiguration{
				Driver: driver, Parameters: runtime.RawExtension{Raw: []byte(raw)},
			},
		},
	}
}

func sharingAffinityTestResult(request, device string, shareID types.UID) resourceapi.DeviceRequestAllocationResult {
	return resourceapi.DeviceRequestAllocationResult{
		Request: request, Driver: sharingAffinityTestDriver, Pool: sharingAffinityTestPool,
		Device: device, ShareID: ptr.To(shareID),
	}
}

func sharingAffinityTestClaim(uid types.UID, configs []resourceapi.DeviceClaimConfiguration, results ...resourceapi.DeviceRequestAllocationResult) *resourceapi.ResourceClaim {
	return &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: string(uid), UID: uid},
		Spec:       resourceapi.ResourceClaimSpec{Devices: resourceapi.DeviceClaim{Config: configs}},
		Status: resourceapi.ResourceClaimStatus{
			Allocation: &resourceapi.AllocationResult{
				Devices: resourceapi.DeviceAllocationResult{Results: results},
			},
		},
	}
}

func sharingAffinityTestLogContext(t *testing.T, verbosity int) (context.Context, ktesting.Buffer) {
	t.Helper()
	_, ctx := ktesting.NewTestContext(t)
	logger := ktesting.NewLogger(t, ktesting.NewConfig(ktesting.BufferLogs(true), ktesting.Verbosity(verbosity)))
	return klog.NewContext(ctx, logger), logger.GetSink().(ktesting.Underlier).GetBuffer()
}

func sharingAffinityTestLogValues(entry ktesting.LogEntry) map[string]any {
	values := make(map[string]any)
	for _, kv := range [][]any{entry.WithKVList, entry.ParameterKVList} {
		for i := 0; i+1 < len(kv); i += 2 {
			values[fmt.Sprint(kv[i])] = kv[i+1]
		}
	}
	return values
}

func sharingAffinityTestErrorLog(t *testing.T, buffer ktesting.Buffer, want error) ktesting.LogEntry {
	t.Helper()
	for _, entry := range buffer.Data() {
		if entry.Type == ktesting.LogError && errors.Is(entry.Err, want) {
			return entry
		}
	}
	t.Fatalf("no Error log wrapping %v; logs:\n%s", want, buffer.String())
	return ktesting.LogEntry{}
}

func sharingAffinityTestAssertOwnership(t *testing.T, devices *allocatedDevices, device structured.DeviceID, status structured.AffinityStatus, values map[string]string) {
	t.Helper()
	owners, _ := devices.affinitySnapshot()
	record, found := owners[device]
	if status == structured.AffinityStatusClean {
		if found {
			t.Errorf("drained device still has affinity ownership: %#v", record)
		}
		return
	}
	if !found || record == nil {
		t.Fatalf("active device %s has no affinity ownership", device)
	}
	if len(record.owners) != 1 {
		t.Errorf("active device has %d owners, want one normalized owner", len(record.owners))
	}
	state, present := foldDeviceAffinity(record)
	if !present || state.Status != status || !reflect.DeepEqual(state.LockedAffinity, values) {
		t.Errorf("active affinity = %#v, present = %t; want %v with values %v", state, present, status, values)
	}
}

func sharingAffinityTestUnreconstructableThenDrain(t *testing.T, ctx context.Context, acct *affinityAccounting, claim *resourceapi.ResourceClaim) {
	t.Helper()
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAConsumableCapacity, true)
	devices := newAllocatedDevices(klog.FromContext(ctx), acct)
	result := claim.Status.Allocation.Devices.Results[0]
	device := structured.MakeDeviceID(result.Driver, result.Pool, result.Device)
	devices.onAdd(claim)
	sharingAffinityTestAssertOwnership(t, devices, device, structured.AffinityStatusUnreconstructable, nil)
	devices.onDelete(claim)
	sharingAffinityTestAssertOwnership(t, devices, device, structured.AffinityStatusClean, nil)
}

// Pool completeness is established before an empty extractor set
// can be interpreted as unconstrained.
func TestSharingAffinityLookupPoolExtractors(t *testing.T) {
	subnet := sharingAffinityTestExtractor("subnet", `parameters.subnet`)
	pkey := sharingAffinityTestExtractor("pkey", `parameters.pkey`)
	makeSlice := func(name string, generation, count int64, entries ...resourceapi.SharingAffinityExtractor) *resourceapi.ResourceSlice {
		return sharingAffinityTestSlice(name, sharingAffinityTestDriver, sharingAffinityTestPool, generation, count, entries...)
	}
	listErr := errors.New("slice cache unavailable")
	for _, tc := range []struct {
		name       string
		slices     []*resourceapi.ResourceSlice
		listErr    error
		wantErr    error
		generation int64
		extractors []resourceapi.SharingAffinityExtractor
	}{
		{
			name: "all metadata slices in name order and device-only slice",
			slices: []*resourceapi.ResourceSlice{
				makeSlice("z-metadata", 7, 3, pkey),
				makeSlice("m-devices", 7, 3),
				makeSlice("a-metadata", 7, 3, subnet),
			},
			generation: 7, extractors: []resourceapi.SharingAffinityExtractor{subnet, pkey},
		},
		{
			name: "highest generation only",
			slices: []*resourceapi.ResourceSlice{
				makeSlice("old", 4, 1, pkey), makeSlice("new", 5, 1, subnet),
			},
			generation: 5, extractors: []resourceapi.SharingAffinityExtractor{subnet},
		},
		{
			name: "other drivers and pool names cannot affect generation or completeness",
			slices: []*resourceapi.ResourceSlice{
				sharingAffinityTestSlice("other-driver", "other.example.com", sharingAffinityTestPool, 99, 2, pkey),
				sharingAffinityTestSlice("other-pool", sharingAffinityTestDriver, "other", 99, 2, pkey),
				makeSlice("selected", 3, 1, subnet),
			},
			generation: 3, extractors: []resourceapi.SharingAffinityExtractor{subnet},
		},
		{name: "complete pool without affinity metadata", slices: []*resourceapi.ResourceSlice{makeSlice("a", 0, 2), makeSlice("b", 0, 2)}},
		{name: "absent pool", wantErr: errAffinityPoolIncomplete},
		{name: "only another driver", slices: []*resourceapi.ResourceSlice{sharingAffinityTestSlice("other", "other.example.com", sharingAffinityTestPool, 0, 1)}, wantErr: errAffinityPoolIncomplete},
		{name: "incomplete", slices: []*resourceapi.ResourceSlice{makeSlice("a", 2, 2, subnet)}, wantErr: errAffinityPoolIncomplete},
		{name: "incomplete without metadata is not unconstrained", slices: []*resourceapi.ResourceSlice{makeSlice("a", 2, 2)}, wantErr: errAffinityPoolIncomplete},
		{name: "inconsistent slice counts", slices: []*resourceapi.ResourceSlice{makeSlice("a", 2, 2, subnet), makeSlice("b", 2, 3, pkey)}, wantErr: errAffinityPoolIncomplete},
		{name: "too many slices", slices: []*resourceapi.ResourceSlice{makeSlice("a", 2, 1, subnet), makeSlice("b", 2, 1, pkey)}, wantErr: errAffinityPoolIncomplete},
		{name: "zero advertised count", slices: []*resourceapi.ResourceSlice{makeSlice("a", 2, 0)}, wantErr: errAffinityPoolIncomplete},
		{name: "complete older generation cannot fill newer generation", slices: []*resourceapi.ResourceSlice{makeSlice("old", 1, 1, subnet), makeSlice("new", 2, 2, pkey)}, wantErr: errAffinityPoolIncomplete},
		{name: "list failure", listErr: listErr, wantErr: listErr},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lister := &sharingAffinityTestLister{slices: tc.slices, err: tc.listErr}
			before := make([]*resourceapi.ResourceSlice, len(tc.slices))
			for i, slice := range tc.slices {
				before[i] = slice.DeepCopy()
			}
			got, err := lookupPoolExtractors(lister, sharingAffinityTestDriver, sharingAffinityTestPool)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("lookupPoolExtractors() error = %v, want %v", err, tc.wantErr)
			}
			if lister.callCount() != 1 {
				t.Errorf("listed slices %d times, want once", lister.callCount())
			}
			for i := range before {
				if !reflect.DeepEqual(before[i], tc.slices[i]) {
					t.Errorf("lookup mutated input slice %d", i)
				}
			}
			if err != nil {
				for _, detail := range []string{sharingAffinityTestDriver, sharingAffinityTestPool} {
					if !strings.Contains(err.Error(), detail) {
						t.Errorf("lookup error lacks pool identity %q: %v", detail, err)
					}
				}
				return
			}
			wantKey := sharingaffinity.PoolKey{Driver: sharingAffinityTestDriver, Name: sharingAffinityTestPool, Generation: tc.generation}
			if got.Key != wantKey || !slices.Equal(got.Extractors, tc.extractors) {
				t.Errorf("lookupPoolExtractors() = %#v, want key %#v and extractors %#v", got, wantKey, tc.extractors)
			}
		})
	}
}

func TestSharingAffinityParseResultRequest(t *testing.T) {
	for _, tc := range []struct {
		name string
		want sharingaffinity.Request
	}{
		{"request", sharingaffinity.Request{Name: "request"}},
		{"parent/sub", sharingaffinity.Request{ParentName: "parent", Name: "sub"}},
		{"parent/sub/nested", sharingaffinity.Request{ParentName: "parent", Name: "sub/nested"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := parseResultRequest(tc.name); got != tc.want {
				t.Errorf("parseResultRequest(%q) = %#v, want %#v", tc.name, got, tc.want)
			}
		})
	}
}

// Poison configs must be excluded before JSON decoding or CEL execution.
func TestSharingAffinityComputeClaimAffinity(t *testing.T) {
	for _, tc := range []struct {
		name, request, group, kind, raw string
		scope                           []string
		matches                         bool
	}{
		{name: "wildcards accept absent metadata", request: "req", group: "*", kind: "*", raw: `{"subnet":"blue"}`, matches: true},
		{name: "exact group and kind v1", request: "req", group: "example.com", kind: "Config", raw: `{"apiVersion":"example.com/v1","kind":"Config","subnet":"blue"}`, matches: true},
		{name: "exact group ignores version", request: "req", group: "example.com", kind: "Config", raw: `{"apiVersion":"example.com/v2","kind":"Config","subnet":"blue"}`, matches: true},
		{name: "wildcard group accepts absent apiVersion", request: "req", group: "*", kind: "Config", raw: `{"kind":"Config","subnet":"blue"}`, matches: true},
		{name: "wildcard kind accepts absent kind", request: "req", group: "example.com", kind: "*", raw: `{"apiVersion":"example.com/v1","subnet":"blue"}`, matches: true},
		// Groupless apiVersion matches only the wildcard, not an empty APIGroup selector.
		{name: "core group", request: "req", group: "*", kind: "Config", raw: `{"apiVersion":"v1","kind":"Config","subnet":"blue"}`, matches: true},
		{name: "core group excludes empty selector", request: "req", group: "", kind: "Config", raw: `{"apiVersion":"v1","kind":"Config","subnet":"blue"}`},
		{name: "core group excludes exact selector", request: "req", group: "example.com", kind: "Config", raw: `{"apiVersion":"v1","kind":"Config","subnet":"blue"}`},
		{name: "core group wildcard does not loosen kind", request: "req", group: "*", kind: "Config", raw: `{"apiVersion":"v1","kind":"Other","subnet":"blue"}`},
		// Non-string values must not be coerced into an exact match.
		{name: "non-string apiVersion with exact group", request: "req", group: "example.com", kind: "*", raw: `{"apiVersion":5,"kind":"Config"}`},
		{name: "non-string kind with exact kind", request: "req", group: "*", kind: "Config", raw: `{"apiVersion":"example.com/v1","kind":5}`},
		{name: "different group", request: "req", group: "example.com", kind: "Config", raw: `{"apiVersion":"other.example.com/v1","kind":"Config"}`},
		{name: "different kind", request: "req", group: "example.com", kind: "Config", raw: `{"apiVersion":"example.com/v1","kind":"Other"}`},
		{name: "exact matcher requires metadata", request: "req", group: "example.com", kind: "Config", raw: `{}`},
		{name: "top-level scope", request: "req", scope: []string{"req"}, group: "*", kind: "*", raw: `{"subnet":"blue"}`, matches: true},
		{name: "subrequest whole claim scope", request: "parent/sub", group: "*", kind: "*", raw: `{"subnet":"blue"}`, matches: true},
		{name: "subrequest parent scope", request: "parent/sub", scope: []string{"parent"}, group: "*", kind: "*", raw: `{"subnet":"blue"}`, matches: true},
		{name: "qualified subrequest scope", request: "parent/sub", scope: []string{"parent/sub"}, group: "*", kind: "*", raw: `{"subnet":"blue"}`, matches: true},
		{name: "multiple scope targets", request: "parent/sub", scope: []string{"other", "parent/sub"}, group: "*", kind: "*", raw: `{"subnet":"blue"}`, matches: true},
		{name: "bare subrequest excluded", request: "parent/sub", scope: []string{"sub"}, group: "*", kind: "*", raw: `{`},
		{name: "sibling subrequest excluded", request: "parent/sub", scope: []string{"parent/other"}, group: "*", kind: "*", raw: `{`},
		{name: "other parent excluded", request: "parent/sub", scope: []string{"other/sub"}, group: "*", kind: "*", raw: `{`},
		{name: "top level excludes qualified target", request: "parent", scope: []string{"parent/sub"}, group: "*", kind: "*", raw: `{`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			entry := sharingAffinityTestExtractor("subnet", `parameters.subnet`)
			entry.APIGroup, entry.Kind = tc.group, tc.kind
			acct := &affinityAccounting{
				extractor: sharingaffinity.NewExtractor(256),
				slices: &sharingAffinityTestLister{slices: []*resourceapi.ResourceSlice{
					sharingAffinityTestSlice("metadata", sharingAffinityTestDriver, sharingAffinityTestPool, 7, 1, entry),
				}},
			}
			result := sharingAffinityTestResult(tc.request, "device", "share")
			claim := sharingAffinityTestClaim("claim", []resourceapi.DeviceClaimConfiguration{
				sharingAffinityTestConfig(sharingAffinityTestDriver, tc.raw, tc.scope...),
				sharingAffinityTestConfig("other.example.com", `{`),
				sharingAffinityTestConfig(sharingAffinityTestDriver, `{`, "unrelated"),
				{},
			}, result)
			got := acct.computeClaimAffinity(ctx, claim)
			if len(got) != 1 {
				t.Fatalf("computeClaimAffinity() returned %d contributions, want 1", len(got))
			}
			device := structured.MakeDeviceID(result.Driver, result.Pool, result.Device)
			wantID := affinityContributionID{claimUID: claim.UID, device: device, shareID: *result.ShareID}
			if got[0].device != device || got[0].id != wantID || got[0].pool != (sharingaffinity.PoolKey{Driver: result.Driver, Name: result.Pool, Generation: 7}) {
				t.Errorf("contribution lost claim/result/pool identity: %#v", got[0])
			}
			wantStatus := structured.AffinityStatusUnreconstructable
			var wantValues map[string]string
			if tc.matches {
				wantStatus, wantValues = structured.AffinityStatusLocked, map[string]string{"subnet": "blue"}
			}
			if got[0].status != wantStatus || !reflect.DeepEqual(got[0].values, wantValues) {
				t.Errorf("contribution status/values = %v/%v, want %v/%v", got[0].status, got[0].values, wantStatus, wantValues)
			}
		})
	}
}

func TestSharingAffinityComputeClaimAffinityUnconstrainedAndIgnored(t *testing.T) {
	for _, name := range []string{"complete unconstrained pool", "unallocated", "no results", "dedicated", "admin access"} {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			result := sharingAffinityTestResult("req", "device", "share")
			claim := sharingAffinityTestClaim("claim", []resourceapi.DeviceClaimConfiguration{sharingAffinityTestConfig(sharingAffinityTestDriver, `{`)}, result)
			lister := &sharingAffinityTestLister{err: errors.New("ignored results must not list slices")}
			wantCalls := 0
			switch name {
			case "complete unconstrained pool":
				lister.err = nil
				lister.slices = []*resourceapi.ResourceSlice{sharingAffinityTestSlice("devices", sharingAffinityTestDriver, sharingAffinityTestPool, 1, 1)}
				wantCalls = 1
			case "unallocated":
				claim.Status.Allocation = nil
			case "no results":
				claim.Status.Allocation.Devices.Results = nil
			case "dedicated":
				claim.Status.Allocation.Devices.Results[0].ShareID = nil
			case "admin access":
				claim.Status.Allocation.Devices.Results[0].AdminAccess = ptr.To(true)
			}
			acct := &affinityAccounting{extractor: sharingaffinity.NewExtractor(256), slices: lister}
			if got := acct.computeClaimAffinity(ctx, claim); len(got) != 0 {
				t.Errorf("computeClaimAffinity() = %#v, want no affinity contribution", got)
			}
			if got := lister.callCount(); got != wantCalls {
				t.Errorf("pool lookup calls = %d, want %d", got, wantCalls)
			}
		})
	}
}

func TestSharingAffinityComputeClaimAffinityConfigIsolation(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	lister := &sharingAffinityTestLister{slices: []*resourceapi.ResourceSlice{
		sharingAffinityTestSlice("metadata", sharingAffinityTestDriver, sharingAffinityTestPool, 1, 1, sharingAffinityTestExtractor("subnet", `parameters.subnet`)),
		sharingAffinityTestSlice("other-driver", "other.example.com", sharingAffinityTestPool, 4, 1, sharingAffinityTestExtractor("other-key", `parameters.subnet`)),
		sharingAffinityTestSlice("other-pool", sharingAffinityTestDriver, "other-pool", 8, 1, sharingAffinityTestExtractor("pool-key", `parameters.subnet`)),
	}}
	acct := &affinityAccounting{extractor: sharingaffinity.NewExtractor(256), slices: lister}
	otherResult := sharingAffinityTestResult("second", "device", "other-share")
	otherResult.Driver = "other.example.com"
	otherPoolResult := sharingAffinityTestResult("first", "device", "other-pool-share")
	otherPoolResult.Pool = "other-pool"
	claim := sharingAffinityTestClaim("claim", []resourceapi.DeviceClaimConfiguration{
		sharingAffinityTestConfig(sharingAffinityTestDriver, `{"subnet":"blue"}`, "first"),
		sharingAffinityTestConfig(sharingAffinityTestDriver, `{"subnet":"green"}`, "second"),
		sharingAffinityTestConfig("other.example.com", `{"subnet":"orange"}`, "second"),
	}, sharingAffinityTestResult("first", "a", "share-a"), sharingAffinityTestResult("second", "b", "share-b"), otherResult, otherPoolResult)
	claim.Status.Allocation.Devices.Config = []resourceapi.DeviceAllocationConfiguration{{
		Source:              resourceapi.AllocationConfigSourceClass,
		DeviceConfiguration: sharingAffinityTestConfig(sharingAffinityTestDriver, `{"subnet":"class-must-not-leak"}`).DeviceConfiguration,
	}}
	before := claim.DeepCopy()
	got := acct.computeClaimAffinity(ctx, claim)
	if len(got) != 4 {
		t.Fatalf("computeClaimAffinity() returned %d contributions, want 4", len(got))
	}
	want := map[structured.DeviceID]map[string]string{
		structured.MakeDeviceID(sharingAffinityTestDriver, sharingAffinityTestPool, "a"): {"subnet": "blue"},
		structured.MakeDeviceID(sharingAffinityTestDriver, sharingAffinityTestPool, "b"): {"subnet": "green"},
		structured.MakeDeviceID("other.example.com", sharingAffinityTestPool, "device"):  {"other-key": "orange"},
		structured.MakeDeviceID(sharingAffinityTestDriver, "other-pool", "device"):       {"pool-key": "blue"},
	}
	for _, contribution := range got {
		values, ok := want[contribution.device]
		if !ok || contribution.status != structured.AffinityStatusLocked || !maps.Equal(contribution.values, values) {
			t.Errorf("request/driver isolation failed: %#v", contribution)
		}
		delete(want, contribution.device)
	}
	if len(want) != 0 {
		t.Errorf("missing contributions: %v", want)
	}
	if !reflect.DeepEqual(claim, before) {
		t.Error("extraction mutated the claim")
	}
	if lister.callCount() != 3 {
		t.Errorf("pool lookups = %d, want one per driver/pool", lister.callCount())
	}
	independent := sharingAffinityTestClaim("independent", []resourceapi.DeviceClaimConfiguration{
		sharingAffinityTestConfig(sharingAffinityTestDriver, `{"subnet":"independent"}`),
	}, sharingAffinityTestResult("first", "a", "share-a"))
	next := acct.computeClaimAffinity(ctx, independent)
	if len(next) != 1 || next[0].status != structured.AffinityStatusLocked || !maps.Equal(next[0].values, map[string]string{"subnet": "independent"}) {
		t.Fatalf("independent claim inherited another claim's extraction: %#v", next)
	}
	next[0].values["subnet"] = "caller-mutation"
	if got[0].values["subnet"] == "caller-mutation" {
		t.Error("independent contributions share a values map")
	}
	if lister.callCount() != 4 {
		t.Errorf("pool lookup cache leaked across claims: calls = %d, want 4", lister.callCount())
	}
}

func TestSharingAffinityComputeClaimAffinityCombinedMetadata(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	metadata := []*resourceapi.ResourceSlice{
		sharingAffinityTestSlice("a", sharingAffinityTestDriver, sharingAffinityTestPool, 1, 2),
		sharingAffinityTestSlice("b", sharingAffinityTestDriver, sharingAffinityTestPool, 1, 2),
	}
	want := make(map[string]string)
	for i := range 9 {
		key := fmt.Sprintf("key-%d", i)
		metadata[i/8].Spec.SharingAffinity = append(metadata[i/8].Spec.SharingAffinity, sharingAffinityTestExtractor(key, `"value"`))
		want[key] = "value"
	}
	lister := &sharingAffinityTestLister{slices: metadata}
	acct := &affinityAccounting{extractor: sharingaffinity.NewExtractor(256), slices: lister}
	claim := sharingAffinityTestClaim("claim", []resourceapi.DeviceClaimConfiguration{sharingAffinityTestConfig(sharingAffinityTestDriver, `{}`)}, sharingAffinityTestResult("req", "device", "share"))
	got := acct.computeClaimAffinity(ctx, claim)
	if len(got) != 1 || got[0].status != structured.AffinityStatusLocked || !maps.Equal(got[0].values, want) {
		t.Fatalf("eight-per-slice limit incorrectly applied to complete pool: %#v", got)
	}
	lister.setSlices(
		sharingAffinityTestSlice("old-a", sharingAffinityTestDriver, sharingAffinityTestPool, 1, 2, sharingAffinityTestExtractor("old", `"old"`)),
		sharingAffinityTestSlice("old-b", sharingAffinityTestDriver, sharingAffinityTestPool, 1, 2),
		sharingAffinityTestSlice("new", sharingAffinityTestDriver, sharingAffinityTestPool, 2, 1, sharingAffinityTestExtractor("new", `"new"`)),
	)
	next := acct.computeClaimAffinity(ctx, claim)
	if len(next) != 1 || next[0].pool.Generation != 2 || next[0].status != structured.AffinityStatusLocked || !maps.Equal(next[0].values, map[string]string{"new": "new"}) {
		t.Errorf("new computation used stale pool metadata: %#v", next)
	}
}

// Inspect the accounting Error log, not merely the extractor's V(5) log,
// to prove wrapping preserves errors.Is/As and actionable claim/result context.
func TestSharingAffinityComputeClaimAffinityErrors(t *testing.T) {
	subnet := sharingAffinityTestExtractor("subnet", `parameters.subnet`)
	empty := sharingAffinityTestExtractor("pkey", `""`)
	config := sharingAffinityTestConfig(sharingAffinityTestDriver, `{"subnet":"secret-blue","private":"opaque-secret"}`)
	listErr := errors.New("slice cache unavailable")
	for _, tc := range []struct {
		name       string
		extractors []resourceapi.SharingAffinityExtractor
		configs    []resourceapi.DeviceClaimConfiguration
		split      bool
		count      int64
		listErr    error
		wantErr    error
		outcome    sharingaffinity.Outcome
		reason     string
	}{
		{name: "unsatisfied", extractors: []resourceapi.SharingAffinityExtractor{empty}, configs: []resourceapi.DeviceClaimConfiguration{config}, reason: "unsatisfied"},
		{name: "no claim configs", extractors: []resourceapi.SharingAffinityExtractor{subnet}, reason: "unsatisfied"},
		{name: "partial", extractors: []resourceapi.SharingAffinityExtractor{subnet, empty}, configs: []resourceapi.DeviceClaimConfiguration{config}, reason: "unsatisfied"},
		{name: "self inconsistent", extractors: []resourceapi.SharingAffinityExtractor{subnet}, configs: []resourceapi.DeviceClaimConfiguration{config, sharingAffinityTestConfig(sharingAffinityTestDriver, `{"subnet":"secret-green"}`)}, wantErr: sharingaffinity.ErrClaimSelfInconsistent, outcome: sharingaffinity.OutcomeClaimInconsistent},
		{name: "duplicate names across metadata slices", extractors: []resourceapi.SharingAffinityExtractor{subnet, subnet}, split: true, configs: []resourceapi.DeviceClaimConfiguration{config}, wantErr: sharingaffinity.ErrInvalidDefinition, outcome: sharingaffinity.OutcomeInvalidDefinition},
		{name: "invalid CEL", extractors: []resourceapi.SharingAffinityExtractor{sharingAffinityTestExtractor("subnet", `parameters.[`)}, configs: []resourceapi.DeviceClaimConfiguration{config}, wantErr: sharingaffinity.ErrInvalidDefinition, outcome: sharingaffinity.OutcomeInvalidDefinition},
		{name: "static nonstring", extractors: []resourceapi.SharingAffinityExtractor{sharingAffinityTestExtractor("subnet", `42`)}, configs: []resourceapi.DeviceClaimConfiguration{config}, wantErr: sharingaffinity.ErrInvalidDefinition, outcome: sharingaffinity.OutcomeInvalidDefinition},
		{name: "dynamic nonstring", extractors: []resourceapi.SharingAffinityExtractor{subnet}, configs: []resourceapi.DeviceClaimConfiguration{sharingAffinityTestConfig(sharingAffinityTestDriver, `{"subnet":42,"private":"opaque-secret"}`)}, wantErr: sharingaffinity.ErrNonStringReturn, outcome: sharingaffinity.OutcomeRuntime},
		{name: "runtime division by zero", extractors: []resourceapi.SharingAffinityExtractor{sharingAffinityTestExtractor("subnet", `string(1 / parameters.zero)`)}, configs: []resourceapi.DeviceClaimConfiguration{sharingAffinityTestConfig(sharingAffinityTestDriver, `{"zero":0,"private":"opaque-secret"}`)}, wantErr: sharingaffinity.ErrCELEval, outcome: sharingaffinity.OutcomeRuntime},
		{name: "missing field discards earlier values", extractors: []resourceapi.SharingAffinityExtractor{subnet, sharingAffinityTestExtractor("pkey", `parameters.missing`)}, configs: []resourceapi.DeviceClaimConfiguration{config}, wantErr: sharingaffinity.ErrMissingField, outcome: sharingaffinity.OutcomeRuntime},
		{name: "malformed JSON", extractors: []resourceapi.SharingAffinityExtractor{subnet}, configs: []resourceapi.DeviceClaimConfiguration{sharingAffinityTestConfig(sharingAffinityTestDriver, `{"private":"opaque-secret","subnet":}`)}, wantErr: sharingaffinity.ErrParameters, outcome: sharingaffinity.OutcomeRuntime},
		{name: "nonobject JSON", extractors: []resourceapi.SharingAffinityExtractor{subnet}, configs: []resourceapi.DeviceClaimConfiguration{sharingAffinityTestConfig(sharingAffinityTestDriver, `["opaque-secret"]`)}, wantErr: sharingaffinity.ErrParameters, outcome: sharingaffinity.OutcomeRuntime},
		{name: "dynamic missing key redaction", extractors: []resourceapi.SharingAffinityExtractor{sharingAffinityTestExtractor("subnet", `parameters[parameters.private]`)}, configs: []resourceapi.DeviceClaimConfiguration{config}, wantErr: sharingaffinity.ErrMissingField, outcome: sharingaffinity.OutcomeRuntime},
		{name: "conversion error redaction", extractors: []resourceapi.SharingAffinityExtractor{sharingAffinityTestExtractor("subnet", `string(int(parameters.private))`)}, configs: []resourceapi.DeviceClaimConfiguration{config}, wantErr: sharingaffinity.ErrCELEval, outcome: sharingaffinity.OutcomeRuntime},
		{name: "incomplete pool", extractors: []resourceapi.SharingAffinityExtractor{subnet}, count: 2, configs: []resourceapi.DeviceClaimConfiguration{config}, reason: "poolIncomplete"},
		{name: "slice list failure", extractors: []resourceapi.SharingAffinityExtractor{subnet}, listErr: listErr, configs: []resourceapi.DeviceClaimConfiguration{config}, wantErr: listErr},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, buffer := sharingAffinityTestLogContext(t, 5)
			count := tc.count
			if count == 0 {
				count = 1
			}
			metadata := []*resourceapi.ResourceSlice{sharingAffinityTestSlice("a", sharingAffinityTestDriver, sharingAffinityTestPool, 7, count, tc.extractors...)}
			if tc.split {
				metadata = []*resourceapi.ResourceSlice{
					sharingAffinityTestSlice("a", sharingAffinityTestDriver, sharingAffinityTestPool, 7, 2, tc.extractors[0]),
					sharingAffinityTestSlice("b", sharingAffinityTestDriver, sharingAffinityTestPool, 7, 2, tc.extractors[1]),
				}
			}
			acct := &affinityAccounting{extractor: sharingaffinity.NewExtractor(256), slices: &sharingAffinityTestLister{slices: metadata, err: tc.listErr}}
			claim := sharingAffinityTestClaim("claim", tc.configs, sharingAffinityTestResult("parent/sub", "device", "share"))
			got := acct.computeClaimAffinity(ctx, claim)
			if len(got) != 1 || got[0].status != structured.AffinityStatusUnreconstructable || got[0].values != nil {
				t.Fatalf("failed extraction produced usable or absent contribution: %#v", got)
			}
			if tc.reason != "" {
				found := false
				for _, entry := range buffer.Data() {
					values := sharingAffinityTestLogValues(entry)
					if entry.Type == ktesting.LogInfo && entry.Verbosity == 2 && values["reason"] == tc.reason {
						found = true
						if tc.reason == "unsatisfied" && len(fmt.Sprint(values["unsatisfied"])) <= 2 {
							t.Errorf("unsatisfied diagnostic lacks missing extractor names: %#v", entry)
						}
					}
				}
				if !found {
					t.Errorf("no V(2) reason=%s diagnostic:\n%s", tc.reason, buffer.String())
				}
			} else {
				entry := sharingAffinityTestErrorLog(t, buffer, tc.wantErr)
				values := sharingAffinityTestLogValues(entry)
				for _, key := range []string{"claim", "claimUID", "device", "request", "pool", "generation", "outcome"} {
					if _, ok := values[key]; !ok {
						t.Errorf("accounting error log lacks %q: %#v", key, entry)
					}
				}
				if values["request"] != "parent/sub" {
					t.Errorf("error log request = %v, want parent/sub", values["request"])
				}
				if tc.listErr == nil {
					var detail *sharingaffinity.Error
					if !errors.As(entry.Err, &detail) {
						t.Errorf("wrapped error lost *sharingaffinity.Error: %v", entry.Err)
					}
					if outcome := sharingaffinity.Classify(entry.Err); outcome != tc.outcome {
						t.Errorf("Classify(logged error) = %v, want %v", outcome, tc.outcome)
					}
				}
			}
			if tc.reason == "poolIncomplete" || tc.listErr != nil {
				// The legacy extraction helper reports unknown contributions, but
				// accounting must keep transient metadata failures recoverable.
				c, _ := newPoolEventsUnitTracker(t, metadata...)
				c.publishMetadata(true, nil)
				lister := &sharingAffinityTestLister{slices: metadata, err: tc.listErr}
				c.affinity = &affinityAccounting{extractor: acct.extractor, slices: lister}
				c.metadataSlices = lister
				c.allocatedDevices.affinityAcct = c.affinity
				c.allocatedDevices.onAdd(claim)
				result := claim.Status.Allocation.Devices.Results[0]
				key := poolName{result.Driver, result.Pool}
				device := structured.MakeDeviceID(result.Driver, result.Pool, result.Device)
				poolID := schedulerapi.PoolID{Driver: result.Driver, Pool: result.Pool}
				state := gatherSharingAffinityAccounting(t, c)
				if !state.DeviceSharingAffinity.StalePools.Has(poolID) {
					t.Fatal("transient metadata failure did not mark the allocated pool stale")
				}
				if _, found := state.DeviceSharingAffinity.DeviceStates[device]; found {
					t.Fatalf("transient failure installed sticky affinity: %+v", state.DeviceSharingAffinity.DeviceStates[device])
				}
				if !state.AllocatedSharedDeviceIDs.Has(device) {
					t.Fatal("pending reconstruction lost share-ID-only allocation")
				}
				c.allocatedDevices.mutex.RLock()
				record := c.allocatedDevices.owners[ownerPoolKey{uid: claim.UID, pool: key}]
				pending := record != nil && record.pending
				fenced := c.allocatedDevices.fenced.Has(device)
				c.allocatedDevices.mutex.RUnlock()
				if !pending || fenced {
					t.Fatalf("transient failure pending=%t fenced=%t, want pending without a sticky fence", pending, fenced)
				}

				ready := metadata[0].DeepCopy()
				ready.Spec.Pool.ResourceSliceCount = 1
				lister.mutex.Lock()
				lister.slices, lister.err = []*resourceapi.ResourceSlice{ready}, nil
				lister.mutex.Unlock()
				c.publishMetadata(false, nil)
				// This unit tracker has no worker. Replay the pending callback;
				// event-only worker recovery is covered by reconstruction tests.
				c.allocatedDevices.onUpdate(claim, claim)
				state = gatherSharingAffinityAccounting(t, c)
				if state.DeviceSharingAffinity.StalePools.Has(poolID) {
					t.Fatal("coherent metadata and owner replay left the pool stale")
				}
				sharingAffinityTestAssertOwnership(t, c.allocatedDevices, device, structured.AffinityStatusLocked, map[string]string{"subnet": "secret-blue"})
				c.allocatedDevices.onDelete(claim)
				sharingAffinityTestAssertOwnership(t, c.allocatedDevices, device, structured.AffinityStatusClean, nil)
			} else {
				sharingAffinityTestUnreconstructableThenDrain(t, ctx, acct, claim)
			}
			for _, secret := range []string{"secret-blue", "secret-green", "opaque-secret"} {
				if strings.Contains(buffer.String(), secret) {
					t.Errorf("accounting logs disclosed opaque/extracted secret %q", secret)
				}
			}
			highCtx, highBuffer := sharingAffinityTestLogContext(t, 10)
			acct.computeClaimAffinity(highCtx, claim)
			if strings.Contains(highBuffer.String(), "opaque-secret") {
				t.Error("high-verbosity accounting logs disclosed raw opaque parameters")
			}
		})
	}
}

func TestSharingAffinityComputeClaimAffinityCombinedCost(t *testing.T) {
	// Derive the threshold from the public validator instead of relying on
	// CEL-version-specific cost constants.
	expression := fmt.Sprintf("'%s'.matches('%s') ? 'x' : 'x'", strings.Repeat("a", 4000), strings.Repeat("a", 2000))
	cost := sharingaffinity.ValidateExpression(expression, environment.StoredExpressions)
	limit := uint64(resourceapi.SharingAffinityCELMaxCost)
	if cost.Error != nil || cost.MaxCost == 0 || cost.MaxCost > limit {
		t.Fatalf("invalid individual-cost fixture: %#v", cost)
	}
	count := int(limit/cost.MaxCost) + 1
	if count > 64 {
		t.Fatalf("cost fixture requires %d metadata slices; choose a higher-cost expression", count)
	}
	metadata := make([]*resourceapi.ResourceSlice, 0, count)
	for i := range count {
		extractor := sharingAffinityTestExtractor(fmt.Sprintf("key-%d", i), expression)
		if err := sharingaffinity.ValidateExtractorSet([]resourceapi.SharingAffinityExtractor{extractor}); err != nil {
			t.Fatalf("individual slice must be valid: %v", err)
		}
		metadata = append(metadata, sharingAffinityTestSlice(fmt.Sprintf("slice-%02d", i), sharingAffinityTestDriver, sharingAffinityTestPool, 7, int64(count), extractor))
	}
	ctx, buffer := sharingAffinityTestLogContext(t, 5)
	acct := &affinityAccounting{extractor: sharingaffinity.NewExtractor(256), slices: &sharingAffinityTestLister{slices: metadata}}
	claim := sharingAffinityTestClaim("claim", []resourceapi.DeviceClaimConfiguration{sharingAffinityTestConfig(sharingAffinityTestDriver, `{}`)}, sharingAffinityTestResult("req", "device", "share"))
	got := acct.computeClaimAffinity(ctx, claim)
	if len(got) != 1 || got[0].status != structured.AffinityStatusUnreconstructable || got[0].values != nil {
		t.Fatalf("combined cost overrun produced usable contribution: %#v", got)
	}
	entry := sharingAffinityTestErrorLog(t, buffer, sharingaffinity.ErrInvalidDefinition)
	if !errors.Is(entry.Err, sharingaffinity.ErrCostExceeded) || sharingaffinity.Classify(entry.Err) != sharingaffinity.OutcomeInvalidDefinition {
		t.Errorf("aggregate error lost cost/definition identity: %v", entry.Err)
	}
	sharingAffinityTestUnreconstructableThenDrain(t, ctx, acct, claim)
}

func TestSharingAffinityComputeClaimAffinityRuntimeBudget(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	entry := sharingAffinityTestExtractor("subnet", `parameters.s.matches("a{1,3}") ? "x" : "x"`)
	if err := sharingaffinity.ValidateExtractorSet([]resourceapi.SharingAffinityExtractor{entry}); err != nil {
		t.Fatalf("runtime-budget fixture must pass static validation: %v", err)
	}
	extractor := sharingaffinity.NewExtractor(256)
	pool := sharingaffinity.PoolExtractors{
		Key:        sharingaffinity.PoolKey{Driver: sharingAffinityTestDriver, Name: sharingAffinityTestPool, Generation: 1},
		Extractors: []resourceapi.SharingAffinityExtractor{entry},
	}
	config := sharingAffinityTestConfig(sharingAffinityTestDriver, `{"s":"`+strings.Repeat("a", 300000)+`"}`)
	var configs []resourceapi.DeviceClaimConfiguration
	// Find a shared-budget failure through the existing public extractor. The
	// scheduler test must not depend on a particular CEL release's unit costs.
	for count := 1; count <= 32; count++ {
		configs = append(configs, config)
		_, err := extractor.Extract(ctx, pool, sharingaffinity.Request{Name: "req"}, configs)
		if errors.Is(err, sharingaffinity.ErrCostExceeded) {
			break
		}
		if err != nil {
			t.Fatalf("unexpected runtime-budget fixture error: %v", err)
		}
		if count == 32 {
			t.Fatal("fixture did not exhaust the shared runtime budget within 32 configs")
		}
	}
	if len(configs) < 2 {
		t.Fatal("one config must succeed before testing shared-budget exhaustion")
	}
	acct := &affinityAccounting{extractor: extractor, slices: &sharingAffinityTestLister{slices: []*resourceapi.ResourceSlice{
		sharingAffinityTestSlice("metadata", sharingAffinityTestDriver, sharingAffinityTestPool, 1, 1, entry),
	}}}
	errorCtx, buffer := sharingAffinityTestLogContext(t, 5)
	claim := sharingAffinityTestClaim("over-budget", configs, sharingAffinityTestResult("req", "device", "share"))
	got := acct.computeClaimAffinity(errorCtx, claim)
	if len(got) != 1 || got[0].status != structured.AffinityStatusUnreconstructable || got[0].values != nil {
		t.Fatalf("runtime budget exhaustion produced usable contribution: %#v", got)
	}
	log := sharingAffinityTestErrorLog(t, buffer, sharingaffinity.ErrCostExceeded)
	if sharingaffinity.Classify(log.Err) != sharingaffinity.OutcomeRuntime {
		t.Errorf("runtime cost failure classified as %v", sharingaffinity.Classify(log.Err))
	}
	sharingAffinityTestUnreconstructableThenDrain(t, errorCtx, acct, claim)
	var scoped []resourceapi.DeviceClaimConfiguration
	for _, request := range []string{"first", "second"} {
		for _, cfg := range configs[:len(configs)-1] {
			cfg.Requests = []string{request}
			scoped = append(scoped, cfg)
		}
	}
	for _, uid := range []types.UID{"independent-a", "independent-b"} {
		claim := sharingAffinityTestClaim(uid, scoped,
			sharingAffinityTestResult("first", "a", "share-a"),
			sharingAffinityTestResult("second", "b", "share-b"))
		got := acct.computeClaimAffinity(ctx, claim)
		if len(got) != 2 {
			t.Fatalf("independent requests returned %d contributions, want 2", len(got))
		}
		for _, contribution := range got {
			if contribution.status != structured.AffinityStatusLocked || !maps.Equal(contribution.values, map[string]string{"subnet": "x"}) {
				t.Errorf("runtime budget leaked across requests or claims: %#v", contribution)
			}
		}
	}
}

func TestSharingAffinityRepeatedShareID(t *testing.T) {
	for _, tc := range []struct {
		name, secondRaw string
		wantStatus      structured.AffinityStatus
		wantConflict    bool
	}{
		{"identical contributions collapse", `{"subnet":"secret-blue"}`, structured.AffinityStatusLocked, false},
		{"different scopes conflict", `{"subnet":"secret-green"}`, structured.AffinityStatusUnreconstructable, true},
		{"partial duplicate cannot hide behind satisfied result", `{}`, structured.AffinityStatusUnreconstructable, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, reverse := range []bool{false, true} {
				t.Run(fmt.Sprintf("reverse-%t", reverse), func(t *testing.T) {
					ctx, buffer := sharingAffinityTestLogContext(t, 5)
					acct := &affinityAccounting{
						extractor: sharingaffinity.NewExtractor(256),
						slices: &sharingAffinityTestLister{slices: []*resourceapi.ResourceSlice{
							sharingAffinityTestSlice("metadata", sharingAffinityTestDriver, sharingAffinityTestPool, 7, 1, sharingAffinityTestExtractor("subnet", `parameters.?subnet.orValue("")`)),
						}},
					}
					results := []resourceapi.DeviceRequestAllocationResult{
						sharingAffinityTestResult("first", "device", "share"),
						sharingAffinityTestResult("second", "device", "share"),
					}
					for i := range results {
						results[i].ConsumedCapacity = map[resourceapi.QualifiedName]resource.Quantity{"units": resource.MustParse("5")}
					}
					if reverse {
						slices.Reverse(results)
					}
					claim := sharingAffinityTestClaim("claim", []resourceapi.DeviceClaimConfiguration{
						sharingAffinityTestConfig(sharingAffinityTestDriver, `{"subnet":"secret-blue"}`, "first"),
						sharingAffinityTestConfig(sharingAffinityTestDriver, tc.secondRaw, "second"),
					}, results...)
					got := acct.computeClaimAffinity(ctx, claim)
					if len(got) != 1 || got[0].status != tc.wantStatus {
						t.Fatalf("repeated ShareID normalization = %#v, want one %v contribution", got, tc.wantStatus)
					}
					if tc.wantConflict {
						if got[0].values != nil {
							t.Errorf("conflicting duplicate retained lock values: %v", got[0].values)
						}
						sharingAffinityTestErrorLog(t, buffer, errAffinityConflict)
					} else if !maps.Equal(got[0].values, map[string]string{"subnet": "secret-blue"}) {
						t.Errorf("identical duplicate lost complete values: %v", got[0].values)
					}
					featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAConsumableCapacity, true)
					enabled := newAllocatedDevices(klog.FromContext(ctx), acct)
					baseline := newAllocatedDevices(klog.FromContext(ctx), nil)
					device := structured.MakeDeviceID(sharingAffinityTestDriver, sharingAffinityTestPool, "device")
					for _, devices := range []*allocatedDevices{enabled, baseline} {
						devices.onAdd(claim)
					}
					enabledIDs, _ := enabled.GetSharedDeviceIDs()
					baselineIDs, _ := baseline.GetSharedDeviceIDs()
					if !enabledIDs.Equal(baselineIDs) || len(enabledIDs) != 1 || !enabledIDs.Has(device) {
						t.Errorf("repeated ShareID changed legacy shared devices: enabled %v, baseline %v", enabledIDs, baselineIDs)
					}
					enabledCapacity, _ := enabled.Capacities()
					baselineCapacity, _ := baseline.Capacities()
					if !reflect.DeepEqual(enabledCapacity, baselineCapacity) {
						t.Errorf("affinity normalization changed legacy capacity: enabled %v, baseline %v", enabledCapacity, baselineCapacity)
					}
					total := enabledCapacity[device][draapi.MakeFullyQualifiedName("units", sharingAffinityTestDriver)]
					if total == nil || total.Cmp(resource.MustParse("10")) != 0 {
						t.Errorf("legacy repeated-result capacity = %v, want 5+5=10 (not deduplicated)", total)
					}
					var wantValues map[string]string
					if tc.wantStatus == structured.AffinityStatusLocked {
						wantValues = map[string]string{"subnet": "secret-blue"}
					}
					sharingAffinityTestAssertOwnership(t, enabled, device, tc.wantStatus, wantValues)
					for _, devices := range []*allocatedDevices{enabled, baseline} {
						devices.onDelete(claim)
						ids, _ := devices.GetSharedDeviceIDs()
						capacity, _ := devices.Capacities()
						if len(ids) != 0 || len(capacity) != 0 {
							t.Errorf("single claim release did not drain legacy accounting: devices %v, capacity %v", ids, capacity)
						}
					}
					sharingAffinityTestAssertOwnership(t, enabled, device, structured.AffinityStatusClean, nil)
					for _, secret := range []string{"secret-blue", "secret-green"} {
						if strings.Contains(buffer.String(), secret) {
							t.Errorf("normalization logs disclosed extracted value %q", secret)
						}
					}
				})
			}
		})
	}
}
