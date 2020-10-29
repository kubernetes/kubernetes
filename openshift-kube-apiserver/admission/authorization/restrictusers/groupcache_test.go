package restrictusers

import (
	userv1 "github.com/openshift/api/user/v1"
)

type fakeGroupCache struct {
	groups []userv1.Group
}

func (g fakeGroupCache) GroupsFor(user string) ([]*userv1.Group, error) {
	ret := []*userv1.Group{}
	for i := range g.groups {
		group := &g.groups[i]
		for _, currUser := range group.Users {
			if user == currUser {
				ret = append(ret, group)
				break
			}
		}

	}
	return ret, nil
}

func (g fakeGroupCache) HasSynced() bool {
	return true
}
