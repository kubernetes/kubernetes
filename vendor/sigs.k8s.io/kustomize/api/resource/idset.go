// Copyright 2020 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package resource

import "sigs.k8s.io/kustomize/api/resid"

type IdSet struct {
	ids map[resid.ResId]bool
}

func MakeIdSet(slice []*Resource) *IdSet {
	set := make(map[resid.ResId]bool)
	for _, r := range slice {
		id := r.CurId()
		if _, ok := set[id]; !ok {
			set[id] = true
		}
	}
	return &IdSet{ids: set}
}

func (s IdSet) Contains(id resid.ResId) bool {
	_, ok := s.ids[id]
	return ok
}

func (s IdSet) Size() int {
	return len(s.ids)
}
