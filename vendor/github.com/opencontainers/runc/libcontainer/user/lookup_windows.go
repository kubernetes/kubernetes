// +build windows

package user

import (
	"fmt"
	"os/user"
)

func lookupUser(username string) (User, error) {
	u, err := user.Lookup(username)
	if err != nil {
		return User{}, err
	}
	return userFromOS(u)
}

func lookupUid(uid int) (User, error) {
	u, err := user.LookupId(fmt.Sprintf("%d", uid))
	if err != nil {
		return User{}, err
	}
	return userFromOS(u)
}

func lookupGroup(groupname string) (Group, error) {
	g, err := user.LookupGroup(groupname)
	if err != nil {
		return Group{}, err
	}
	return groupFromOS(g)
}

func lookupGid(gid int) (Group, error) {
	g, err := user.LookupGroupId(fmt.Sprintf("%d", gid))
	if err != nil {
		return Group{}, err
	}
	return groupFromOS(g)
}
