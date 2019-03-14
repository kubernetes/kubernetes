package mesos

import (
	"bytes"
	"fmt"
	"strconv"

	"github.com/gogo/protobuf/proto"
	"github.com/mesos/mesos-go/api/v1/lib/roles"
)

const DefaultRole = "*"

type (
	Resources         []Resource
	resourceErrorType int

	resourceError struct {
		errorType resourceErrorType
		reason    string
		spec      Resource
	}
)

const (
	resourceErrorTypeIllegalName resourceErrorType = iota
	resourceErrorTypeIllegalType
	resourceErrorTypeUnsupportedType
	resourceErrorTypeIllegalScalar
	resourceErrorTypeIllegalRanges
	resourceErrorTypeIllegalSet
	resourceErrorTypeIllegalDisk
	resourceErrorTypeIllegalReservation
	resourceErrorTypeIllegalShare

	noReason = "" // make error generation code more readable
)

var (
	resourceErrorMessages = map[resourceErrorType]string{
		resourceErrorTypeIllegalName:        "missing or illegal resource name",
		resourceErrorTypeIllegalType:        "missing or illegal resource type",
		resourceErrorTypeUnsupportedType:    "unsupported resource type",
		resourceErrorTypeIllegalScalar:      "illegal scalar resource",
		resourceErrorTypeIllegalRanges:      "illegal ranges resource",
		resourceErrorTypeIllegalSet:         "illegal set resource",
		resourceErrorTypeIllegalDisk:        "illegal disk resource",
		resourceErrorTypeIllegalReservation: "illegal resource reservation",
		resourceErrorTypeIllegalShare:       "illegal shared resource",
	}
)

func (t resourceErrorType) Generate(reason string) error {
	msg := resourceErrorMessages[t]
	if reason != noReason {
		if msg != "" {
			msg += ": " + reason
		} else {
			msg = reason
		}
	}
	return &resourceError{errorType: t, reason: msg}
}

func (err *resourceError) Reason() string          { return err.reason }
func (err *resourceError) Resource() Resource      { return err.spec }
func (err *resourceError) WithResource(r Resource) { err.spec = r }

func (err *resourceError) Error() string {
	// TODO(jdef) include additional context here? (type, resource)
	if err.reason != "" {
		return "resource error: " + err.reason
	}
	return "resource error"
}

func IsResourceError(err error) (ok bool) {
	_, ok = err.(*resourceError)
	return
}

func (r *Resource_ReservationInfo) Assign() func(interface{}) {
	return func(v interface{}) {
		type reserver interface {
			WithReservation(*Resource_ReservationInfo)
		}
		if ri, ok := v.(reserver); ok {
			ri.WithReservation(r)
		}
	}
}

func (resources Resources) Clone() Resources {
	if resources == nil {
		return nil
	}
	clone := make(Resources, 0, len(resources))
	for i := range resources {
		rr := proto.Clone(&resources[i]).(*Resource)
		clone = append(clone, *rr)
	}
	return clone
}

// Minus calculates and returns the result of `resources - that` without modifying either
// the receiving `resources` or `that`.
func (resources Resources) Minus(that ...Resource) Resources {
	x := resources.Clone()
	return x.Subtract(that...)
}

// Subtract subtracts `that` from the receiving `resources` and returns the result (the modified
// `resources` receiver).
func (resources *Resources) Subtract(that ...Resource) (rs Resources) {
	if resources != nil {
		if len(that) > 0 {
			x := make(Resources, len(that))
			copy(x, that)
			that = x

			for i := range that {
				resources.Subtract1(that[i])
			}
		}
		rs = *resources
	}
	return
}

// Plus calculates and returns the result of `resources + that` without modifying either
// the receiving `resources` or `that`.
func (resources Resources) Plus(that ...Resource) Resources {
	x := resources.Clone()
	return x.Add(that...)
}

// Add adds `that` to the receiving `resources` and returns the result (the modified
// `resources` receiver).
func (resources *Resources) Add(that ...Resource) (rs Resources) {
	if resources != nil {
		rs = *resources
	}
	for i := range that {
		rs = rs._add(that[i])
	}
	if resources != nil {
		*resources = rs
	}
	return
}

// Add1 adds `that` to the receiving `resources` and returns the result (the modified
// `resources` receiver).
func (resources *Resources) Add1(that Resource) (rs Resources) {
	if resources != nil {
		rs = *resources
	}
	rs = rs._add(that)
	if resources != nil {
		*resources = rs
	}
	return
}

func (resources Resources) _add(that Resource) Resources {
	if that.Validate() != nil || that.IsEmpty() {
		return resources
	}
	for i := range resources {
		r := &resources[i]
		if r.Addable(that) {
			r.Add(that)
			return resources
		}
	}
	// cannot be combined with an existing resource
	r := proto.Clone(&that).(*Resource)
	return append(resources, *r)
}

// Minus1 calculates and returns the result of `resources - that` without modifying either
// the receiving `resources` or `that`.
func (resources *Resources) Minus1(that Resource) Resources {
	x := resources.Clone()
	return x.Subtract1(that)
}

// Subtract1 subtracts `that` from the receiving `resources` and returns the result (the modified
// `resources` receiver).
func (resources *Resources) Subtract1(that Resource) Resources {
	if resources == nil {
		return nil
	}
	if that.Validate() == nil && !that.IsEmpty() {
		for i := range *resources {
			r := &(*resources)[i]
			if r.Subtractable(that) {
				r.Subtract(that)
				// remove the resource if it becomes invalid or zero.
				// need to do validation in order to strip negative scalar
				// resource objects.
				if r.Validate() != nil || r.IsEmpty() {
					// delete resource at i, without leaking an uncollectable Resource
					// a, a[len(a)-1] = append(a[:i], a[i+1:]...), nil
					(*resources), (*resources)[len((*resources))-1] = append((*resources)[:i], (*resources)[i+1:]...), Resource{}
				}
				break
			}
		}
	}
	return *resources
}

// String returns a human-friendly representation of the resource collection using default formatting
// options (e.g. allocation-info is not rendered). For additional control over resource formatting see
// the Format func.
func (resources Resources) String() string {
	return resources.Format()
}

type ResourcesFormatOptions struct {
	ShowAllocated bool // ShowAllocated when true will not display resource allocation info
}

func (resources Resources) Format(options ...func(*ResourcesFormatOptions)) string {
	if len(resources) == 0 {
		return ""
	}
	var f ResourcesFormatOptions
	for _, o := range options {
		if o != nil {
			o(&f)
		}
	}
	// TODO(jdef) use a string.Builder once we can rely on a more modern golang version
	buf := bytes.Buffer{}
	for i := range resources {
		if i > 0 {
			buf.WriteString(";")
		}
		r := &resources[i]
		buf.WriteString(r.Name)
		if r.AllocationInfo != nil && f.ShowAllocated {
			buf.WriteString("(allocated: ")
			buf.WriteString(r.AllocationInfo.GetRole())
			buf.WriteString(")")
		}
		if res := r.Reservations; len(res) > 0 || (r.Role != nil && *r.Role != "*") {
			if len(res) == 0 {
				res = make([]Resource_ReservationInfo, 0, 1)
				if r.Reservation == nil {
					res = append(res, Resource_ReservationInfo{
						Type: Resource_ReservationInfo_STATIC.Enum(),
						Role: r.Role,
					})
				} else {
					res = append(res, *r.Reservation) // copy!
					res[0].Type = Resource_ReservationInfo_DYNAMIC.Enum()
					res[0].Role = r.Role
				}
			}
			buf.WriteString("(reservations: [")
			for j := range res {
				if j > 0 {
					buf.WriteString(",")
				}
				rr := &res[j]
				buf.WriteString("(")
				buf.WriteString(rr.GetType().String())
				buf.WriteString(",")
				buf.WriteString(rr.GetRole())
				if rr.Principal != nil {
					buf.WriteString(",")
					buf.WriteString(*rr.Principal)
				}
				if rr.Labels != nil {
					buf.WriteString(",{")
					rr.GetLabels().writeTo(&buf)
					buf.WriteString("}")
				}
				buf.WriteString(")")
			}
			buf.WriteString("])")
		}
		if d := r.GetDisk(); d != nil {
			buf.WriteString("[")
			if s := d.GetSource(); s != nil {
				switch s.GetType() {
				case Resource_DiskInfo_Source_BLOCK:
					buf.WriteString("BLOCK")
					if id, profile := s.GetID(), s.GetProfile(); id != "" || profile != "" {
						buf.WriteByte('(')
						buf.WriteString(id)
						buf.WriteByte(',')
						buf.WriteString(profile)
						buf.WriteByte(')')
					}
				case Resource_DiskInfo_Source_RAW:
					buf.WriteString("RAW")
					if id, profile := s.GetID(), s.GetProfile(); id != "" || profile != "" {
						buf.WriteByte('(')
						buf.WriteString(id)
						buf.WriteByte(',')
						buf.WriteString(profile)
						buf.WriteByte(')')
					}
				case Resource_DiskInfo_Source_PATH:
					buf.WriteString("PATH")
					if id, profile := s.GetID(), s.GetProfile(); id != "" || profile != "" {
						buf.WriteByte('(')
						buf.WriteString(id)
						buf.WriteByte(',')
						buf.WriteString(profile)
						buf.WriteByte(')')
					} else if root := s.GetPath().GetRoot(); root != "" {
						buf.WriteByte(':')
						buf.WriteString(root)
					}
				case Resource_DiskInfo_Source_MOUNT:
					buf.WriteString("MOUNT")
					if id, profile := s.GetID(), s.GetProfile(); id != "" || profile != "" {
						buf.WriteByte('(')
						buf.WriteString(id)
						buf.WriteByte(',')
						buf.WriteString(profile)
						buf.WriteByte(')')
					} else if root := s.GetMount().GetRoot(); root != "" {
						buf.WriteByte(':')
						buf.WriteString(root)
					}
				}
			}
			if p := d.GetPersistence(); p != nil {
				if d.GetSource() != nil {
					buf.WriteString(",")
				}
				buf.WriteString(p.GetID())
			}
			if v := d.GetVolume(); v != nil {
				buf.WriteString(":")
				vconfig := v.GetContainerPath()
				if h := v.GetHostPath(); h != "" {
					vconfig = h + ":" + vconfig
				}
				if m := v.Mode; m != nil {
					switch *m {
					case RO:
						vconfig += ":ro"
					case RW:
						vconfig += ":rw"
					default:
						panic("unrecognized volume mode: " + m.String())
					}
				}
				buf.WriteString(vconfig)
			}
			buf.WriteString("]")
		}
		if r.Revocable != nil {
			buf.WriteString("{REV}")
		}
		if r.Shared != nil {
			buf.WriteString("<SHARED>")
		}
		buf.WriteString(":")
		switch r.GetType() {
		case SCALAR:
			buf.WriteString(strconv.FormatFloat(r.GetScalar().GetValue(), 'f', -1, 64))
		case RANGES:
			buf.WriteString("[")
			ranges := Ranges(r.GetRanges().GetRange())
			for j := range ranges {
				if j > 0 {
					buf.WriteString(",")
				}
				if b, e := ranges[j].Begin, ranges[j].End; b == e {
					buf.WriteString(strconv.FormatUint(b, 10))
				} else {
					buf.WriteString(strconv.FormatUint(b, 10))
					buf.WriteString("-")
					buf.WriteString(strconv.FormatUint(e, 10))
				}
			}
			buf.WriteString("]")
		case SET:
			buf.WriteString("{")
			items := r.GetSet().GetItem()
			for j := range items {
				if j > 0 {
					buf.WriteString(",")
				}
				buf.WriteString(items[j])
			}
			buf.WriteString("}")
		}
	}
	return buf.String()
}

func (left *Resource) Validate() error {
	if left.GetName() == "" {
		return resourceErrorTypeIllegalName.Generate(noReason)
	}
	if _, ok := Value_Type_name[int32(left.GetType())]; !ok {
		return resourceErrorTypeIllegalType.Generate(noReason)
	}
	switch left.GetType() {
	case SCALAR:
		if s := left.GetScalar(); s == nil || left.GetRanges() != nil || left.GetSet() != nil {
			return resourceErrorTypeIllegalScalar.Generate(noReason)
		} else if s.GetValue() < 0 {
			return resourceErrorTypeIllegalScalar.Generate("value < 0")
		}
	case RANGES:
		r := left.GetRanges()
		if left.GetScalar() != nil || r == nil || left.GetSet() != nil {
			return resourceErrorTypeIllegalRanges.Generate(noReason)
		}
		for i, rr := range r.GetRange() {
			// ensure that ranges are not inverted
			if rr.Begin > rr.End {
				return resourceErrorTypeIllegalRanges.Generate("begin > end")
			}
			// ensure that ranges don't overlap (but not necessarily squashed)
			for j := i + 1; j < len(r.GetRange()); j++ {
				r2 := r.GetRange()[j]
				if rr.Begin <= r2.Begin && r2.Begin <= rr.End {
					return resourceErrorTypeIllegalRanges.Generate("overlapping ranges")
				}
			}
		}
	case SET:
		s := left.GetSet()
		if left.GetScalar() != nil || left.GetRanges() != nil || s == nil {
			return resourceErrorTypeIllegalSet.Generate(noReason)
		}
		unique := make(map[string]struct{}, len(s.GetItem()))
		for _, x := range s.GetItem() {
			if _, found := unique[x]; found {
				return resourceErrorTypeIllegalSet.Generate("duplicated elements")
			}
			unique[x] = struct{}{}
		}
	default:
		return resourceErrorTypeUnsupportedType.Generate(noReason)
	}

	// check for disk resource
	if disk := left.GetDisk(); disk != nil {
		if left.GetName() != "disk" {
			return resourceErrorTypeIllegalDisk.Generate("DiskInfo should not be set for \"" + left.GetName() + "\" resource")
		}
		if s := disk.GetSource(); s != nil {
			switch s.GetType() {
			case Resource_DiskInfo_Source_PATH,
				Resource_DiskInfo_Source_MOUNT:
				// these only contain optional members
			case Resource_DiskInfo_Source_BLOCK,
				Resource_DiskInfo_Source_RAW:
				// TODO(jdef): update w/ validation once the format of BLOCK and RAW
				// disks is known.
			case Resource_DiskInfo_Source_UNKNOWN:
				return resourceErrorTypeIllegalDisk.Generate(fmt.Sprintf("unsupported DiskInfo.Source.Type in %q", s))
			}
		}
	}

	if rs := left.GetReservations(); len(rs) == 0 {
		// check for "pre-reservation-refinement" format
		if _, err := roles.Parse(left.GetRole()); err != nil {
			return resourceErrorTypeIllegalReservation.Generate(err.Error())
		}

		if r := left.GetReservation(); r != nil {
			if r.Type != nil {
				return resourceErrorTypeIllegalReservation.Generate(
					"Resource.ReservationInfo.type must not be set for the Resource.reservation field")
			}
			if r.Role != nil {
				return resourceErrorTypeIllegalReservation.Generate(
					"Resource.ReservationInfo.role must not be set for the Resource.reservation field")
			}
			// check for invalid state of (role,reservation) pair
			if left.GetRole() == "*" {
				return resourceErrorTypeIllegalReservation.Generate("default role cannot be dynamically reserved")
			}
		}
	} else {
		// check for "post-reservation-refinement" format
		for i := range rs {
			r := &rs[i]
			if r.Type == nil {
				return resourceErrorTypeIllegalReservation.Generate(
					"Resource.ReservationInfo.type must be set")
			}
			if r.Role == nil {
				return resourceErrorTypeIllegalReservation.Generate(
					"Resource.ReservationInfo.role must be set")
			}
			if _, err := roles.Parse(r.GetRole()); err != nil {
				return resourceErrorTypeIllegalReservation.Generate(err.Error())
			}
			if r.GetRole() == "*" {
				return resourceErrorTypeIllegalReservation.Generate(
					"role '*' cannot be reserved")
			}
		}
		// check that reservations are correctly refined
		ancestor := rs[0].GetRole()
		for i := 1; i < len(rs); i++ {
			r := &rs[i]
			if r.GetType() == Resource_ReservationInfo_STATIC {
				return resourceErrorTypeIllegalReservation.Generate(
					"a refined reservation cannot be STATIC")
			}
			child := r.GetRole()
			if !roles.IsStrictSubroleOf(child, ancestor) {
				return resourceErrorTypeIllegalReservation.Generate(fmt.Sprintf(
					"role %q is not a refinement of %q", child, ancestor))
			}
		}

		// Additionally, we allow the "pre-reservation-refinement" format to be set
		// as long as there is only one reservation, and the `Resource.role` and
		// `Resource.reservation` fields are consistent with the reservation.
		if len(rs) == 1 {
			if r := left.Role; r != nil && *r != rs[0].GetRole() {
				return resourceErrorTypeIllegalReservation.Generate(fmt.Sprintf(
					"'Resource.role' field with %q does not match the role %q in 'Resource.reservations'",
					*r, rs[0].GetRole()))
			}

			switch rs[0].GetType() {
			case Resource_ReservationInfo_STATIC:
				if left.Reservation != nil {
					return resourceErrorTypeIllegalReservation.Generate(
						"'Resource.reservation' must not be set if the single reservation in 'Resource.reservations' is STATIC")
				}
			case Resource_ReservationInfo_DYNAMIC:
				if (left.Role == nil) != (left.GetReservation() == nil) {
					return resourceErrorTypeIllegalReservation.Generate(
						"'Resource.role' and 'Resource.reservation' must both be set or both not be set if the single reservation in 'Resource.reservations' is DYNAMIC")
				}
				if r := left.GetReservation(); r != nil && r.GetPrincipal() != rs[0].GetPrincipal() {
					return resourceErrorTypeIllegalReservation.Generate(fmt.Sprintf(
						"'Resource.reservation.principal' with %q does not match the principal %q in 'Resource.reservations'",
						r.GetPrincipal(), rs[0].GetPrincipal()))
				}
				if r := left.GetReservation(); r != nil && !r.GetLabels().Equivalent(rs[0].GetLabels()) {
					return resourceErrorTypeIllegalReservation.Generate(fmt.Sprintf(
						"'Resource.reservation.labels' with %q does not match the labels %q in 'Resource.reservations'",
						r.GetLabels(), rs[0].GetLabels()))
				}
			case Resource_ReservationInfo_UNKNOWN:
				return resourceErrorTypeIllegalReservation.Generate("Unsupported 'Resource.ReservationInfo.type'")
			}
		} else {
			if r := left.Role; r != nil {
				return resourceErrorTypeIllegalReservation.Generate(
					"'Resource.role' must not be set if there is more than one reservation in 'Resource.reservations'")
			}
			if r := left.GetReservation(); r != nil {
				return resourceErrorTypeIllegalReservation.Generate(
					"'Resource.reservation' must not be set if there is more than one reservation in 'Resource.reservations'")
			}
		}
	}

	// Check that shareability is enabled for supported resource types.
	// For now, it is for persistent volumes only.
	// NOTE: We need to modify this once we extend shareability to other
	// resource types.
	if s := left.GetShared(); s != nil {
		if left.GetName() != "disk" {
			return resourceErrorTypeIllegalShare.Generate(fmt.Sprintf(
				"Resource %q cannot be shared", left.GetName()))
		}
		if p := left.GetDisk().GetPersistence(); p == nil {
			return resourceErrorTypeIllegalShare.Generate("only persistent volumes can be shared")
		}
	}

	return nil
}

func (left *Resource_AllocationInfo) Equivalent(right *Resource_AllocationInfo) bool {
	if (left == nil) != (right == nil) {
		return false
	} else if left == nil {
		return true
	}
	if (left.Role == nil) != (right.Role == nil) {
		return false
	}
	if left.Role != nil && *left.Role != *right.Role {
		return false
	}
	return true
}

func (r *Resource_ReservationInfo) Equivalent(right *Resource_ReservationInfo) bool {
	// TODO(jdef) should we consider equivalency of both pre- and post-refinement formats,
	// such that a pre-refinement format could be the equivalent of a post-refinement format
	// if defined just the right way?
	if (r == nil) != (right == nil) {
		return false
	} else if r == nil {
		return true
	}
	if (r.Type == nil) != (right.Type == nil) {
		return false
	}
	if r.Type != nil && *r.Type != *right.Type {
		return false
	}
	if (r.Role == nil) != (right.Role == nil) {
		return false
	}
	if r.Role != nil && *r.Role != *right.Role {
		return false
	}
	if (r.Principal == nil) != (right.Principal == nil) {
		return false
	}
	if r.Principal != nil && *r.Principal != *right.Principal {
		return false
	}
	return r.Labels.Equivalent(right.Labels)
}

func (left *Resource_DiskInfo) Equivalent(right *Resource_DiskInfo) bool {
	// NOTE: We ignore 'volume' inside DiskInfo when doing comparison
	// because it describes how this resource will be used which has
	// nothing to do with the Resource object itself. A framework can
	// use this resource and specify different 'volume' every time it
	// uses it.
	// see https://github.com/apache/mesos/blob/0.25.0/src/common/resources.cpp#L67
	if (left == nil) != (right == nil) {
		return false
	}

	if a, b := left.GetSource(), right.GetSource(); (a == nil) != (b == nil) {
		return false
	} else if a != nil {
		if a.GetType() != b.GetType() {
			return false
		}
		if aa, bb := a.GetMount(), b.GetMount(); (aa == nil) != (bb == nil) {
			return false
		} else if aa.GetRoot() != bb.GetRoot() {
			return false
		}
		if aa, bb := a.GetPath(), b.GetPath(); (aa == nil) != (bb == nil) {
			return false
		} else if aa.GetRoot() != bb.GetRoot() {
			return false
		}
		if aa, bb := a.GetID(), b.GetID(); aa != bb {
			return false
		}
		if aa, bb := a.GetProfile(), b.GetProfile(); aa != bb {
			return false
		}
		if aa, bb := a.GetMetadata(), b.GetMetadata(); (aa == nil) != (bb == nil) {
			return false
		} else if !labelList(aa.GetLabels()).Equivalent(labelList(bb.GetLabels())) {
			return false
		}
	}

	if a, b := left.GetPersistence(), right.GetPersistence(); (a == nil) != (b == nil) {
		return false
	} else if a != nil {
		return a.GetID() == b.GetID()
	}

	return true
}

// Equivalent returns true if right is equivalent to left (differs from Equal in that
// deeply nested values are test for equivalence, not equality).
func (left *Resource) Equivalent(right Resource) bool {
	if left == nil {
		return right.IsEmpty()
	}
	if left.GetName() != right.GetName() ||
		left.GetType() != right.GetType() ||
		left.GetRole() != right.GetRole() {
		return false
	}
	if a, b := left.GetAllocationInfo(), right.GetAllocationInfo(); !a.Equivalent(b) {
		return false
	}
	if a, b := left.GetReservations(), right.GetReservations(); len(a) != len(b) {
		return false
	} else {
		for i := range a {
			ri := &a[i]
			if !ri.Equivalent(&b[i]) {
				return false
			}
		}
	}
	if !left.GetReservation().Equivalent(right.GetReservation()) {
		return false
	}
	if !left.GetDisk().Equivalent(right.GetDisk()) {
		return false
	}
	if (left.Revocable == nil) != (right.Revocable == nil) {
		return false
	}
	if a, b := left.ProviderID, right.ProviderID; (a == nil) != (b == nil) {
		return false
	} else if a != nil && a.Value != b.Value {
		return false
	}
	if a, b := left.Shared, right.Shared; (a == nil) != (b == nil) {
		return false
	}

	switch left.GetType() {
	case SCALAR:
		return left.GetScalar().Compare(right.GetScalar()) == 0
	case RANGES:
		return Ranges(left.GetRanges().GetRange()).Equivalent(right.GetRanges().GetRange())
	case SET:
		return left.GetSet().Compare(right.GetSet()) == 0
	default:
		return false
	}
}

// Addable tests if we can add two Resource objects together resulting in one
// valid Resource object. For example, two Resource objects with
// different name, type or role are not addable.
func (left *Resource) Addable(right Resource) bool {
	if left == nil {
		return true
	}
	if left.GetName() != right.GetName() ||
		left.GetType() != right.GetType() ||
		left.GetRole() != right.GetRole() {
		return false
	}

	if a, b := left.GetShared(), right.GetShared(); (a == nil) != (b == nil) {
		// shared has no fields
		return false
	}

	if a, b := left.GetAllocationInfo(), right.GetAllocationInfo(); !a.Equivalent(b) {
		return false
	}

	if !left.GetReservation().Equivalent(right.GetReservation()) {
		return false
	}

	if a, b := left.Reservations, right.Reservations; len(a) != len(b) {
		return false
	} else {
		for i := range a {
			aa := &a[i]
			if !aa.Equivalent(&b[i]) {
				return false
			}
		}
	}

	if !left.GetDisk().Equivalent(right.GetDisk()) {
		return false
	}

	if ls := left.GetDisk().GetSource(); ls != nil {
		switch ls.GetType() {
		case Resource_DiskInfo_Source_PATH:
			// Two PATH resources can be added if their disks are identical
		case Resource_DiskInfo_Source_BLOCK,
			Resource_DiskInfo_Source_MOUNT:
			// Two resources that represent exclusive 'MOUNT' or 'RAW' disks
			// cannot be added together; this would defeat the exclusivity.
			return false
		case Resource_DiskInfo_Source_RAW:
			// We can only add resources representing 'RAW' disks if
			// they have no identity or are identical.
			if ls.GetID() != "" {
				return false
			}
		case Resource_DiskInfo_Source_UNKNOWN:
			panic("unreachable")
		}
	}

	// from apache/mesos: src/common/resources.cpp
	// TODO(jieyu): Even if two Resource objects with DiskInfo have the
	// same persistence ID, they cannot be added together. In fact, this
	// shouldn't happen if we do not add resources from different
	// namespaces (e.g., across slave). Consider adding a warning.
	if left.GetDisk().GetPersistence() != nil {
		return false
	}
	if (left.GetRevocable() == nil) != (right.GetRevocable() == nil) {
		return false
	}
	if a, b := left.GetProviderID(), right.GetProviderID(); (a == nil) != (b == nil) {
		return false
	} else if a != nil && a.Value != b.Value {
		return false
	}
	return true
}

// Subtractable tests if we can subtract "right" from "left" resulting in one
// valid Resource object. For example, two Resource objects with different
// name, type or role are not subtractable.
// NOTE: Set subtraction is always well defined, it does not require
// 'right' to be contained within 'left'. For example, assuming that
// "left = {1, 2}" and "right = {2, 3}", "left" and "right" are
// subtractable because "left - right = {1}". However, "left" does not
// contain "right".
func (left *Resource) Subtractable(right Resource) bool {
	if left.GetName() != right.GetName() ||
		left.GetType() != right.GetType() ||
		left.GetRole() != right.GetRole() {
		return false
	}
	if a, b := left.GetShared(), right.GetShared(); (a == nil) != (b == nil) {
		// shared has no fields
		return false
	}

	if a, b := left.GetAllocationInfo(), right.GetAllocationInfo(); !a.Equivalent(b) {
		return false
	}

	if !left.GetReservation().Equivalent(right.GetReservation()) {
		return false
	}
	if a, b := left.Reservations, right.Reservations; len(a) != len(b) {
		return false
	} else {
		for i := range a {
			aa := &a[i]
			if !aa.Equivalent(&b[i]) {
				return false
			}
		}
	}

	if !left.GetDisk().Equivalent(right.GetDisk()) {
		return false
	}

	if ls := left.GetDisk().GetSource(); ls != nil {
		switch ls.GetType() {
		case Resource_DiskInfo_Source_PATH:
			// Two PATH resources can be subtracted if their disks are identical
		case Resource_DiskInfo_Source_BLOCK,
			Resource_DiskInfo_Source_MOUNT:
			// Two resources that represent exclusive 'MOUNT' or 'RAW' disks
			// cannot be substracted from each other if they are not the same;
			// this would defeat the exclusivity.
			if !left.Equivalent(right) {
				return false
			}
		case Resource_DiskInfo_Source_RAW:
			// We can only add resources representing 'RAW' disks if
			// they have no identity or refer to the same disk.
			if ls.GetID() != "" && !left.Equivalent(right) {
				return false
			}
		case Resource_DiskInfo_Source_UNKNOWN:
			panic("unreachable")
		}
	}

	// NOTE: For Resource objects that have DiskInfo, we can only do
	// subtraction if they are **equal**.
	if left.GetDisk().GetPersistence() != nil && !left.Equivalent(right) {
		return false
	}
	if (left.GetRevocable() == nil) != (right.GetRevocable() == nil) {
		return false
	}
	if a, b := left.GetProviderID(), right.GetProviderID(); (a == nil) != (b == nil) {
		return false
	} else if a != nil && a.Value != b.Value {
		return false
	}
	return true
}

// Contains tests if "right" is contained in "left".
func (left Resource) Contains(right Resource) bool {
	if !left.Subtractable(right) {
		return false
	}
	switch left.GetType() {
	case SCALAR:
		return right.GetScalar().Compare(left.GetScalar()) <= 0
	case RANGES:
		return right.GetRanges().Compare(left.GetRanges()) <= 0
	case SET:
		return right.GetSet().Compare(left.GetSet()) <= 0
	default:
		return false
	}
}

// Subtract removes right from left.
// This func panics if the resource types don't match.
func (left *Resource) Subtract(right Resource) {
	switch right.checkType(left.GetType()) {
	case SCALAR:
		left.Scalar = left.GetScalar().Subtract(right.GetScalar())
	case RANGES:
		left.Ranges = left.GetRanges().Subtract(right.GetRanges())
	case SET:
		left.Set = left.GetSet().Subtract(right.GetSet())
	}
}

// Add adds right to left.
// This func panics if the resource types don't match.
func (left *Resource) Add(right Resource) {
	switch right.checkType(left.GetType()) {
	case SCALAR:
		left.Scalar = left.GetScalar().Add(right.GetScalar())
	case RANGES:
		left.Ranges = left.GetRanges().Add(right.GetRanges())
	case SET:
		left.Set = left.GetSet().Add(right.GetSet())
	}
}

// checkType panics if the type of this resources != t
func (left *Resource) checkType(t Value_Type) Value_Type {
	if left != nil && left.GetType() != t {
		panic(fmt.Sprintf("expected type %v instead of %v", t, left.GetType()))
	}
	return t
}

// IsEmpty returns true if the value of this resource is equivalent to the zero-value,
// where a zero-length slice or map is equivalent to a nil reference to such.
func (left *Resource) IsEmpty() bool {
	if left == nil {
		return true
	}
	switch left.GetType() {
	case SCALAR:
		return left.GetScalar().GetValue() == 0
	case RANGES:
		return len(left.GetRanges().GetRange()) == 0
	case SET:
		return len(left.GetSet().GetItem()) == 0
	}
	return false
}

// IsUnreserved returns true if this resource neither statically or dynamically reserved.
// A resource is considered statically reserved if it has a non-default role.
func (left *Resource) IsUnreserved() bool {
	// role != RoleDefault     -> static reservation
	// GetReservation() != nil -> dynamic reservation
	// return {no-static-reservation} && {no-dynamic-reservation}
	return (left.Role == nil || left.GetRole() == "*") && left.GetReservation() == nil && len(left.GetReservations()) == 0
}

// IsReserved returns true if this resource has been reserved for the given role.
// If role=="" then return true if there are no static or dynamic reservations for this resource.
// It's expected that this Resource has already been validated (see Validate).
func (left *Resource) IsReserved(role string) bool {
	return !left.IsUnreserved() && (role == "" || role == left.ReservationRole())
}

// ReservationRole returns the role for which the resource is reserved. Callers should check the
// reservation status of the resource via IsReserved prior to invoking this func.
func (r *Resource) ReservationRole() string {
	// if using reservation refinement, return the role of the last refinement
	rs := r.GetReservations()
	if x := len(rs); x > 0 {
		return rs[x-1].GetRole()
	}
	// if using the old reservation API, role is a first class field of Resource
	// (and it's never stored in Resource.Reservation).
	return r.GetRole()
}

// IsAllocatableTo returns true if the resource may be allocated to the given role.
func (left *Resource) IsAllocatableTo(role string) bool {
	if left.IsUnreserved() {
		return true
	}
	r := left.ReservationRole()
	return role == r || roles.IsStrictSubroleOf(role, r)
}

// IsDynamicallyReserved returns true if this resource has a non-nil reservation descriptor
func (left *Resource) IsDynamicallyReserved() bool {
	if left.IsReserved("") {
		if left.GetReservation() != nil {
			return true
		}
		rs := left.GetReservations()
		return rs[len(rs)-1].GetType() == Resource_ReservationInfo_DYNAMIC
	}
	return false
}

// IsRevocable returns true if this resource has a non-nil revocable descriptor
func (left *Resource) IsRevocable() bool {
	return left.GetRevocable() != nil
}

// IsPersistentVolume returns true if this is a disk resource with a non-nil Persistence descriptor
func (left *Resource) IsPersistentVolume() bool {
	return left.GetDisk().GetPersistence() != nil
}

// IsDisk returns true if this is a disk resource of the specified type.
func (left *Resource) IsDisk(t Resource_DiskInfo_Source_Type) bool {
	if s := left.GetDisk().GetSource(); s != nil {
		return s.GetType() == t
	}
	return false
}

// HasResourceProvider returns true if the given Resource object is provided by a resource provider.
func (left *Resource) HasResourceProvider() bool {
	return left.GetProviderID() != nil
}

// ToUnreserved returns a (cloned) view of the Resources w/o any reservation data. It does not modify
// the receiver.
func (rs Resources) ToUnreserved() (result Resources) {
	if rs == nil {
		return nil
	}
	for i := range rs {
		r := rs[i] // intentionally shallow-copy
		r.Reservations = nil
		r.Reservation = nil
		r.Role = nil
		result.Add1(r)
	}
	return
}

// PushReservation returns a cloned set of Resources w/ the given resource refinement.
// Panics if resources become invalid as a result of pushing the reservation (e.g. pre- and post-
// refinement modes are mixed).
func (rs Resources) PushReservation(ri Resource_ReservationInfo) (result Resources) {
push_next:
	for i := range rs {
		if rs[i].IsEmpty() {
			continue
		}
		r := proto.Clone(&rs[i]).(*Resource) // we don't want to impact rs
		r.Reservations = append(r.Reservations, *(proto.Clone(&ri).(*Resource_ReservationInfo)))

		if err := r.Validate(); err != nil {
			panic(err)
		}

		// unroll Add1 to avoid additional calls to Clone
		rr := *r
		for j := range result {
			r2 := &result[j]
			if r2.Addable(rr) {
				r2.Add(rr)
				continue push_next
			}
		}
		// cannot be combined with an existing resource
		result = append(result, rr)
	}
	return
}

// PopReservation returns a cloned set of Resources wherein the most recent reservation refeinement has been
// removed. Panics if for any resource in the collection there is no "last refinement" to remove.
func (rs Resources) PopReservation() (result Resources) {
pop_next:
	for i := range rs {
		r := &rs[i]
		ls := len(r.Reservations)
		if ls == 0 {
			panic(fmt.Sprintf("no reservations exist for resource %q", r))
		}

		r = proto.Clone(r).(*Resource)                    // avoid modifying rs
		r.Reservations[ls-1] = Resource_ReservationInfo{} // don't leak nested pointers
		r.Reservations = r.Reservations[:ls-1]            // shrink the slice

		// unroll Add1 to avoid additional calls to Clone
		rr := *r
		for j := range result {
			r2 := &result[j]
			if r2.Addable(rr) {
				r2.Add(rr)
				continue pop_next
			}
		}

		// cannot be combined with an existing resource
		result = append(result, rr)
	}
	return
}

// Allocate sets the AllocationInfo for the resource, panics if role is "".
func (r *Resource) Allocate(role string) {
	if role == "" {
		panic(fmt.Sprintf("cannot allocate resource to an empty-string role: %q", r))
	}
	r.AllocationInfo = &Resource_AllocationInfo{Role: &role}
}

// Unallocate clears the AllocationInfo for the resource.
func (r *Resource) Unallocate() {
	r.AllocationInfo = nil
}

// Allocate sets the AllocationInfo for all the resources.
// Returns a reference to the receiver to allow for chaining.
func (rs Resources) Allocate(role string) Resources {
	if role == "" {
		panic(fmt.Sprintf("cannot allocate resources to an empty-string role: %q", rs))
	}
	for i := range rs {
		rs[i].AllocationInfo = &Resource_AllocationInfo{Role: &role}
	}
	return rs
}

// Unallocate clears the AllocationInfo for all the resources.
// Returns a reference to the receiver to allow for chaining.
func (rs Resources) Unallocate() Resources {
	for i := range rs {
		rs[i].AllocationInfo = nil
	}
	return rs
}
