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

package resourcepoolstatusrequest

import (
	"fmt"
	"slices"
	"sort"
	"strings"

	resourcev1 "k8s.io/api/resource/v1"
	resourcev1alpha3 "k8s.io/api/resource/v1alpha3"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/utils/ptr"
)

// maxStatusListItems matches the +k8s:maxItems=32 marker on the status lists.
// Exceeding it fails status validation, so views report a validationError
// rather than truncating.
const maxStatusListItems = 32

// Machine-readable prefixes for pool validationError messages.
const (
	// prefixPoolIncomplete is the only transient error; syncRequest requeues on it.
	prefixPoolIncomplete = "PoolIncomplete:"

	prefixPartitionTypeMissing    = "PartitionTypeMissing:"
	prefixPartitionCostMismatch   = "PartitionCostMismatch:"
	prefixPartitionSummaryOverCap = "PartitionSummaryOverCap:"
	prefixShareableOverCap        = "ShareableSummaryOverCap:"
)

// deviceRecord is the per-device data the advanced views are computed from.
type deviceRecord struct {
	name string
	// partitionAttr is the grouping attribute resolved for this device: the
	// PartitionTypeAttribute its own slice declared, or the request default
	// when the slice declares none. Empty when neither names one.
	partitionAttr string
	// partitionType is the value of partitionAttr on this device;
	// hasPartitionType is false when the device lacks it as a string.
	partitionType    string
	hasPartitionType bool
	consumesCounters []resourcev1.DeviceCounterConsumption
	allowMultiple    bool
	capacity         map[resourcev1.QualifiedName]resourcev1.DeviceCapacity
	// attributes lets the partition type be resolved once partitionAttr is known.
	attributes map[resourcev1.QualifiedName]resourcev1.DeviceAttribute
}

// sliceDeviceRecords builds the per-device records for one slice, seeding each
// with the slice's own PartitionTypeAttribute (empty when the slice declares none).
func sliceDeviceRecords(slice *resourcev1.ResourceSlice) []deviceRecord {
	var sliceAttr string
	if slice.Spec.PartitionTypeAttribute != nil {
		sliceAttr = string(*slice.Spec.PartitionTypeAttribute)
	}
	recs := make([]deviceRecord, 0, len(slice.Spec.Devices))
	for i := range slice.Spec.Devices {
		d := &slice.Spec.Devices[i]
		recs = append(recs, deviceRecord{
			name:             d.Name,
			partitionAttr:    sliceAttr,
			consumesCounters: d.ConsumesCounters,
			allowMultiple:    d.AllowMultipleAllocations != nil && *d.AllowMultipleAllocations,
			capacity:         d.Capacity,
			attributes:       d.Attributes,
		})
	}
	return recs
}

// resolvePartitionType returns the string value of attribute fqn on a device.
// An attribute key's domain defaults to the driver name (as CEL selectors do),
// so an attribute in the driver's domain may be declared explicitly or bare;
// the explicit form wins, keeping the result deterministic when both are present.
// A non-string or absent attribute yields ("", false).
func resolvePartitionType(driver string, attrs map[resourcev1.QualifiedName]resourcev1.DeviceAttribute, fqn string) (string, bool) {
	attr, ok := attrs[resourcev1.QualifiedName(fqn)]
	if !ok {
		if bare, cut := strings.CutPrefix(fqn, driver+"/"); cut {
			attr, ok = attrs[resourcev1.QualifiedName(bare)]
		}
	}
	if !ok || attr.StringValue == nil {
		return "", false
	}
	return *attr.StringValue, true
}

// poolViewInput carries everything needed to compute the advanced views for one
// complete pool.
type poolViewInput struct {
	driver   string
	poolName string
	devices  []deviceRecord
	// sharedCounters merged from all slices in the pool (names are unique per pool).
	sharedCounters []resourcev1.CounterSet
	// inUse holds device names with at least one non-AdminAccess claim.
	inUse map[string]struct{}
	// consumedCapacity is per-key consumption summed over non-AdminAccess claims.
	consumedCapacity map[resourcev1.QualifiedName]resource.Quantity
}

// resolveDevicePartitions resolves each device's partition type against its
// grouping attribute. A device's own slice declaration takes precedence, and
// slices may declare differing attributes: a pool that mixes partitions is
// grouped per device, not per pool. The request's attribute is a pool-wide
// fallback used only when no slice in the pool declares one, for drivers that
// have not been updated to declare it themselves.
func resolveDevicePartitions(requestAttr *string, in *poolViewInput) {
	poolDeclares := false
	for i := range in.devices {
		if in.devices[i].partitionAttr != "" {
			poolDeclares = true
			break
		}
	}
	for i := range in.devices {
		d := &in.devices[i]
		if d.partitionAttr == "" {
			if poolDeclares || requestAttr == nil {
				continue // no grouping attribute for this device
			}
			d.partitionAttr = *requestAttr
		}
		d.partitionType, d.hasPartitionType = resolvePartitionType(in.driver, d.attributes, d.partitionAttr)
	}
}

// computePoolViews returns the partition and shareable views for a pool.
// Uncomputable views are nil; validationError holds the first structural problem
// (prefixed). Basic device counts stay valid regardless of validationError.
// A partitionable pool with no resolved grouping attribute reports no partition
// view (neither the request nor the driver named one).
func computePoolViews(in poolViewInput) (partitionSummary []resourcev1alpha3.PartitionTypeStatus, shareable *resourcev1alpha3.ShareableSummaryStatus, validationError string) {
	hasCounters := len(in.sharedCounters) > 0

	grouped := false
	for i := range in.devices {
		if in.devices[i].partitionAttr != "" {
			grouped = true
			break
		}
	}

	switch {
	case grouped && !hasCounters:
		validationError = fmt.Sprintf("%s pool %s/%s declares partitionTypeAttribute but publishes no sharedCounters",
			prefixPartitionTypeMissing, in.driver, in.poolName)
	case grouped:
		partitionSummary, validationError = computePartitionSummary(in)
	}

	// shareableSummary is independent of the partition view. Keep the first error.
	sh, shErr := computeShareableSummary(in)
	if validationError == "" {
		validationError = shErr
	}
	if shErr == "" {
		shareable = sh
	}
	if validationError != "" {
		// Drop any partially-built partition view; the pool is flagged instead.
		partitionSummary = nil
	}
	return partitionSummary, shareable, validationError
}

// computePartitionSummary groups partitionable (counter-consuming) devices by
// (grouping attribute, partition type) and reports, per group, the total device
// count and how many additional devices still fit given current shared-counter
// consumption. Devices that consume no counters are not partitions and are
// skipped; so are devices for which no grouping attribute was resolved.
func computePartitionSummary(in poolViewInput) ([]resourcev1alpha3.PartitionTypeStatus, string) {
	type groupKey struct{ attr, typ string }
	type group struct {
		total   int32
		fresh   []deviceRecord
		costKey string
	}
	groups := map[groupKey]*group{}
	for i := range in.devices {
		d := in.devices[i]
		if len(d.consumesCounters) == 0 || d.partitionAttr == "" {
			continue
		}
		if !d.hasPartitionType {
			return nil, fmt.Sprintf("%s device %q in pool %s/%s does not carry partition type attribute %q",
				prefixPartitionTypeMissing, d.name, in.driver, in.poolName, d.partitionAttr)
		}
		k := groupKey{d.partitionAttr, d.partitionType}
		g := groups[k]
		if g == nil {
			g = &group{}
			groups[k] = g
		}
		key := consumesCountersKey(d.consumesCounters)
		if g.total == 0 {
			g.costKey = key
		} else if key != g.costKey {
			return nil, fmt.Sprintf("%s devices of partition type %q in pool %s/%s consume different counters",
				prefixPartitionCostMismatch, d.partitionType, in.driver, in.poolName)
		}
		g.total++
		if _, used := in.inUse[d.name]; !used {
			g.fresh = append(g.fresh, d)
		}
	}

	// Headroom after in-use consumption. Each group is measured independently
	// against this baseline: allocatable[G] = min(fresh[G], floor(avail/cost[G])).
	baseline := availableCounters(in.sharedCounters, in.devices, in.inUse)

	result := make([]resourcev1alpha3.PartitionTypeStatus, 0, len(groups))
	for k, g := range groups {
		avail := cloneCounters(baseline)
		allocatable := int32(0)
		for _, d := range g.fresh {
			if deviceFits(d.consumesCounters, avail) {
				allocatable++
				deductCounters(d.consumesCounters, avail)
			}
		}
		result = append(result, resourcev1alpha3.PartitionTypeStatus{
			Attribute:   k.attr,
			Type:        k.typ,
			Total:       ptr.To(g.total),
			Allocatable: ptr.To(allocatable),
		})
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].Attribute != result[j].Attribute {
			return result[i].Attribute < result[j].Attribute
		}
		return result[i].Type < result[j].Type
	})

	if len(result) > maxStatusListItems {
		return nil, fmt.Sprintf("%s pool %s/%s has %d partition types, exceeding the maximum of %d",
			prefixPartitionSummaryOverCap, in.driver, in.poolName, len(result), maxStatusListItems)
	}
	return result, ""
}

// computeShareableSummary reports aggregate capacity for pools containing devices
// with allowMultipleAllocations. It returns nil when the pool has no shareable
// devices.
func computeShareableSummary(in poolViewInput) (*resourcev1alpha3.ShareableSummaryStatus, string) {
	var full, partial int32
	total := map[resourcev1.QualifiedName]resource.Quantity{}
	any := false
	for i := range in.devices {
		d := in.devices[i]
		if !d.allowMultiple {
			continue
		}
		any = true
		if _, used := in.inUse[d.name]; used {
			partial++
		} else {
			full++
		}
		for key, capacity := range d.capacity {
			cur := total[key].DeepCopy()
			cur.Add(capacity.Value)
			total[key] = cur
		}
	}
	if !any {
		return nil, ""
	}

	keys := make([]resourcev1.QualifiedName, 0, len(total))
	for key := range total {
		keys = append(keys, key)
	}
	slices.Sort(keys)

	capacities := make([]resourcev1alpha3.ShareableCapacityStatus, 0, len(keys))
	for _, key := range keys {
		t := total[key]
		cons := resource.Quantity{}
		if q, ok := in.consumedCapacity[key]; ok {
			cons = q.DeepCopy()
		}
		avail := nonNegativeDiff(t, cons)
		capacities = append(capacities, resourcev1alpha3.ShareableCapacityStatus{
			Name:      string(key),
			Total:     &t,
			Consumed:  &cons,
			Available: &avail,
		})
	}
	if len(capacities) > maxStatusListItems {
		return nil, fmt.Sprintf("%s pool %s/%s has %d shareable capacity keys, exceeding the maximum of %d",
			prefixShareableOverCap, in.driver, in.poolName, len(capacities), maxStatusListItems)
	}

	return &resourcev1alpha3.ShareableSummaryStatus{
		FullyAvailableDevices:     ptr.To(full),
		PartiallyAvailableDevices: ptr.To(partial),
		Capacity:                  capacities,
	}, ""
}

// availableCounters returns [set][counter] capacity minus in-use consumption.
// A counter is debited once per in-use device (not per claim), matching the
// scheduler.
func availableCounters(sharedCounters []resourcev1.CounterSet, devices []deviceRecord, inUse map[string]struct{}) map[string]map[string]resource.Quantity {
	avail := map[string]map[string]resource.Quantity{}
	for _, cs := range sharedCounters {
		m := avail[cs.Name]
		if m == nil {
			m = map[string]resource.Quantity{}
			avail[cs.Name] = m
		}
		for name, c := range cs.Counters {
			m[name] = c.Value.DeepCopy()
		}
	}
	deductInUse(avail, devices, inUse)
	return avail
}

// deductInUse subtracts each in-use device's counter consumption from avail.
func deductInUse(avail map[string]map[string]resource.Quantity, devices []deviceRecord, inUse map[string]struct{}) {
	for i := range devices {
		d := devices[i]
		if _, used := inUse[d.name]; !used {
			continue
		}
		deductCounters(d.consumesCounters, avail)
	}
}

// deviceFits reports whether a device's counter cost is satisfied by avail.
func deviceFits(cost []resourcev1.DeviceCounterConsumption, avail map[string]map[string]resource.Quantity) bool {
	for _, cc := range cost {
		set := avail[cc.CounterSet]
		for name, c := range cc.Counters {
			have, ok := set[name]
			if !ok {
				return false
			}
			if have.Cmp(ptr.Deref(c.Value, resource.Quantity{})) < 0 {
				return false
			}
		}
	}
	return true
}

// deductCounters subtracts a device's counter cost from avail in place.
func deductCounters(cost []resourcev1.DeviceCounterConsumption, avail map[string]map[string]resource.Quantity) {
	for _, cc := range cost {
		set := avail[cc.CounterSet]
		if set == nil {
			continue
		}
		for name, c := range cc.Counters {
			have, ok := set[name]
			if !ok {
				continue
			}
			have.Sub(ptr.Deref(c.Value, resource.Quantity{}))
			set[name] = have
		}
	}
}

// cloneCounters deep-copies a [set][counter] quantity map so per-type greedy fit
// does not mutate the shared baseline.
func cloneCounters(src map[string]map[string]resource.Quantity) map[string]map[string]resource.Quantity {
	out := make(map[string]map[string]resource.Quantity, len(src))
	for set, counters := range src {
		m := make(map[string]resource.Quantity, len(counters))
		for name, q := range counters {
			m[name] = q.DeepCopy()
		}
		out[set] = m
	}
	return out
}

// consumesCountersKey builds a stable string identifying a device's counter cost,
// used to detect heterogeneous costs within a partition type.
func consumesCountersKey(cost []resourcev1.DeviceCounterConsumption) string {
	parts := make([]string, 0, len(cost))
	for _, cc := range cost {
		names := make([]string, 0, len(cc.Counters))
		for name := range cc.Counters {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			q := cc.Counters[name]
			quantity := ptr.Deref(q.Value, resource.Quantity{})
			parts = append(parts, cc.CounterSet+"/"+name+"="+quantity.String())
		}
	}
	sort.Strings(parts)
	return strings.Join(parts, ",")
}

// nonNegativeDiff returns max(0, a-b) as a new Quantity, preserving a's format.
func nonNegativeDiff(a, b resource.Quantity) resource.Quantity {
	out := a.DeepCopy()
	out.Sub(b)
	if out.Sign() < 0 {
		return *resource.NewQuantity(0, a.Format)
	}
	return out
}
