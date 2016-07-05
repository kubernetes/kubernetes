/*
Copyright 2015 The Kubernetes Authors.

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

package resources

import (
	"github.com/gogo/protobuf/proto"
	mesos "github.com/mesos/mesos-go/mesosproto"
)

type Port struct {
	Port uint64
	Role string
}

// PortRanges creates a range resource for the spec ports.
func PortRanges(Ports []Port) []*mesos.Resource {
	rolePorts := make(map[string][]uint64, len(Ports))

	for _, p := range Ports {
		rolePorts[p.Role] = append(rolePorts[p.Role], p.Port)
	}

	resources := make([]*mesos.Resource, 0, len(rolePorts))
	for role, ports := range rolePorts {
		resources = append(
			resources,
			&mesos.Resource{
				Name:   proto.String("ports"),
				Type:   mesos.Value_RANGES.Enum(),
				Ranges: NewRanges(ports),
				Role:   StringPtrTo(role),
			},
		)
	}

	return resources
}

// NewRanges generates port ranges from the given list of ports. (naive implementation)
func NewRanges(ports []uint64) *mesos.Value_Ranges {
	r := make([]*mesos.Value_Range, 0, len(ports))
	for _, port := range ports {
		x := proto.Uint64(port)
		r = append(r, &mesos.Value_Range{Begin: x, End: x})
	}
	return &mesos.Value_Ranges{Range: r}
}

// ForeachPortsRange calls f for each resource that matches the given roles
// in the order of the given roles.
func ForeachPortsRange(rs []*mesos.Resource, roles []string, f func(begin, end uint64, role string)) {
	rs = Filter(rs, HasName("ports"))
	rs = ByRoles(roles...).Sort(rs)

	for _, resource := range rs {
		for _, r := range (*resource).GetRanges().Range {
			bp := r.GetBegin()
			ep := r.GetEnd()
			f(bp, ep, (*resource).GetRole())
		}
	}
}

// ByRolesSorter sorts resources according to the ordering of roles.
type ByRolesSorter struct {
	roles []string
}

// ByRoles returns a ByRolesSorter with the given roles.
func ByRoles(roles ...string) *ByRolesSorter {
	return &ByRolesSorter{roles: roles}
}

// sort sorts the given resources according to the order of roles in the ByRolesSorter
// and returns the sorted resources.
func (sorter *ByRolesSorter) Sort(resources []*mesos.Resource) []*mesos.Resource {
	rolesMap := map[string][]*mesos.Resource{} // maps roles to resources
	for _, res := range resources {
		role := CanonicalRole(res.GetRole())
		rolesMap[role] = append(rolesMap[role], res)
	}

	result := make([]*mesos.Resource, 0, len(resources))
	for _, role := range sorter.roles {
		for _, res := range rolesMap[role] {
			result = append(result, res)
		}
	}

	return result
}

// ResourcePredicate is a predicate function on *mesos.Resource structs.
type (
	ResourcePredicate  func(*mesos.Resource) bool
	ResourcePredicates []ResourcePredicate
)

// Filter filters the given slice of resources and returns a slice of resources
// matching all given predicates.
func Filter(res []*mesos.Resource, ps ...ResourcePredicate) []*mesos.Resource {
	return ResourcePredicates(ps).Filter(res)
}

// Filter filters the given slice of resources and returns a slice of resources
// matching all given predicates.
func (ps ResourcePredicates) Filter(res []*mesos.Resource) []*mesos.Resource {
	filtered := make([]*mesos.Resource, 0, len(res))

next:
	for _, r := range res {
		for _, p := range ps {
			if !p(r) {
				continue next
			}
		}

		filtered = append(filtered, r)
	}

	return filtered
}

// MatchesAll returns true if the given resource matches all given predicates ps.
func MatchesAll(res *mesos.Resource, ps ...ResourcePredicate) bool {
	return ResourcePredicates(ps).MatchesAll(res)
}

// MatchesAll returns true if the given resource matches all given predicates ps.
func (ps ResourcePredicates) MatchesAll(res *mesos.Resource) bool {
	for _, p := range ps {
		if !p(res) {
			return false
		}
	}

	return true
}

func Sum(res []*mesos.Resource) float64 {
	var sum float64

	for _, r := range res {
		sum += r.GetScalar().GetValue()
	}

	return sum
}

// IsScalar returns true if the given resource is a scalar type.
func IsScalar(r *mesos.Resource) bool {
	return r.GetType() == mesos.Value_SCALAR
}

// HasName returns a ResourcePredicate which returns true
// if the given resource has the given name.
func HasName(name string) ResourcePredicate {
	return func(r *mesos.Resource) bool {
		return r.GetName() == name
	}
}

// StringPtrTo returns a pointer to the given string
// or nil if it is empty string.
func StringPtrTo(s string) *string {
	var protos *string

	if s != "" {
		protos = &s
	}

	return protos
}

// CanonicalRole returns a "*" if the given role is empty else the role itself
func CanonicalRole(name string) string {
	if name == "" {
		return "*"
	}

	return name
}

func NewPorts(role string, ports ...uint64) *mesos.Resource {
	return &mesos.Resource{
		Name:   proto.String("ports"),
		Type:   mesos.Value_RANGES.Enum(),
		Ranges: NewRanges(ports),
		Role:   StringPtrTo(role),
	}
}
