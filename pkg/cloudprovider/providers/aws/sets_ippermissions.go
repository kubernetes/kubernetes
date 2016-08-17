/*
Copyright 2016 The Kubernetes Authors.

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

package aws

import (
	"encoding/json"
	"fmt"

	"github.com/aws/aws-sdk-go/service/ec2"
)

type IPPermissionSet map[string]*ec2.IpPermission

func NewIPPermissionSet(items ...*ec2.IpPermission) IPPermissionSet {
	s := make(IPPermissionSet)
	s.Insert(items...)
	return s
}

// Ungroup splits permissions out into individual permissions
// EC2 will combine permissions with the same port but different SourceRanges together, for example
// We ungroup them so we can process them
func (s IPPermissionSet) Ungroup() IPPermissionSet {
	l := []*ec2.IpPermission{}
	for _, p := range s.List() {
		if len(p.IpRanges) <= 1 {
			l = append(l, p)
			continue
		}
		for _, ipRange := range p.IpRanges {
			c := &ec2.IpPermission{}
			*c = *p
			c.IpRanges = []*ec2.IpRange{ipRange}
			l = append(l, c)
		}
	}

	l2 := []*ec2.IpPermission{}
	for _, p := range l {
		if len(p.UserIdGroupPairs) <= 1 {
			l2 = append(l2, p)
			continue
		}
		for _, u := range p.UserIdGroupPairs {
			c := &ec2.IpPermission{}
			*c = *p
			c.UserIdGroupPairs = []*ec2.UserIdGroupPair{u}
			l2 = append(l, c)
		}
	}

	l3 := []*ec2.IpPermission{}
	for _, p := range l2 {
		if len(p.PrefixListIds) <= 1 {
			l3 = append(l3, p)
			continue
		}
		for _, v := range p.PrefixListIds {
			c := &ec2.IpPermission{}
			*c = *p
			c.PrefixListIds = []*ec2.PrefixListId{v}
			l3 = append(l3, c)
		}
	}

	return NewIPPermissionSet(l3...)
}

// Insert adds items to the set.
func (s IPPermissionSet) Insert(items ...*ec2.IpPermission) {
	for _, p := range items {
		k := keyForIPPermission(p)
		s[k] = p
	}
}

// List returns the contents as a slice.  Order is not defined.
func (s IPPermissionSet) List() []*ec2.IpPermission {
	res := make([]*ec2.IpPermission, 0, len(s))
	for _, v := range s {
		res = append(res, v)
	}
	return res
}

// IsSuperset returns true if and only if s1 is a superset of s2.
func (s1 IPPermissionSet) IsSuperset(s2 IPPermissionSet) bool {
	for k := range s2 {
		_, found := s1[k]
		if !found {
			return false
		}
	}
	return true
}

// Equal returns true if and only if s1 is equal (as a set) to s2.
// Two sets are equal if their membership is identical.
// (In practice, this means same elements, order doesn't matter)
func (s1 IPPermissionSet) Equal(s2 IPPermissionSet) bool {
	return len(s1) == len(s2) && s1.IsSuperset(s2)
}

// Difference returns a set of objects that are not in s2
// For example:
// s1 = {a1, a2, a3}
// s2 = {a1, a2, a4, a5}
// s1.Difference(s2) = {a3}
// s2.Difference(s1) = {a4, a5}
func (s IPPermissionSet) Difference(s2 IPPermissionSet) IPPermissionSet {
	result := NewIPPermissionSet()
	for k, v := range s {
		_, found := s2[k]
		if !found {
			result[k] = v
		}
	}
	return result
}

// Len returns the size of the set.
func (s IPPermissionSet) Len() int {
	return len(s)
}

func keyForIPPermission(p *ec2.IpPermission) string {
	v, err := json.Marshal(p)
	if err != nil {
		panic(fmt.Sprintf("error building JSON representation of ec2.IpPermission: %v", err))
	}
	return string(v)
}
