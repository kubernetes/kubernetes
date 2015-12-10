/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package podtask

import (
	"github.com/gogo/protobuf/proto"
	mesos "github.com/mesos/mesos-go/mesosproto"
)

// portRangeResources creates a range resource for the spec ports.
func portRangeResources(Ports []Port) []*mesos.Resource {
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
				Ranges: newRanges(ports),
				Role:   stringPtrTo(role),
			},
		)
	}

	return resources
}

// newRanges generates port ranges from the given list of ports. (naive implementation)
func newRanges(ports []uint64) *mesos.Value_Ranges {
	r := make([]*mesos.Value_Range, 0, len(ports))
	for _, port := range ports {
		x := proto.Uint64(port)
		r = append(r, &mesos.Value_Range{Begin: x, End: x})
	}
	return &mesos.Value_Ranges{Range: r}
}

// foreachPortsRange calls f for each resource that matches the given roles
// in the order of the given roles.
func foreachPortsRange(rs []*mesos.Resource, roles []string, f func(begin, end uint64, role string)) {
	rs = filterResources(rs, hasName("ports"))
	rs = byRoles(roles...).sort(rs)

	for _, resource := range rs {
		for _, r := range (*resource).GetRanges().Range {
			bp := r.GetBegin()
			ep := r.GetEnd()
			f(bp, ep, (*resource).GetRole())
		}
	}
}

// byRolesSorter sorts resources according to the ordering of roles.
type byRolesSorter struct {
	roles []string
}

// byRoles returns a byRolesSorter with the given roles.
func byRoles(roles ...string) *byRolesSorter {
	return &byRolesSorter{roles: roles}
}

// sort sorts the given resources according to the order of roles in the byRolesSorter
// and returns the sorted resources.
func (sorter *byRolesSorter) sort(resources []*mesos.Resource) []*mesos.Resource {
	rolesMap := map[string][]*mesos.Resource{} // maps roles to resources
	for _, res := range resources {
		role := starredRole(res.GetRole())
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

// resourcePredicate is a predicate function on *mesos.Resource structs.
type resourcePredicate func(*mesos.Resource) bool

// filter filters the given slice of resources and returns a slice of resources
// matching all given predicates.
func filterResources(res []*mesos.Resource, ps ...resourcePredicate) []*mesos.Resource {
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

// resourceMatchesAll returns true if the given resource matches all given predicates ps.
func resourceMatchesAll(res *mesos.Resource, ps ...resourcePredicate) bool {
	for _, p := range ps {
		if !p(res) {
			return false
		}
	}

	return true
}

func sumResources(res []*mesos.Resource) float64 {
	var sum float64

	for _, r := range res {
		sum += r.GetScalar().GetValue()
	}

	return sum
}

// isScalar returns true if the given resource is a scalar type.
func isScalar(r *mesos.Resource) bool {
	return r.GetType() == mesos.Value_SCALAR
}

// hasName returns a resourcePredicate which returns true
// if the given resource has the given name.
func hasName(name string) resourcePredicate {
	return func(r *mesos.Resource) bool {
		return r.GetName() == name
	}
}
