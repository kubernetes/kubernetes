// +build !windows

package listeners

import (
	"fmt"
	"strconv"

	"github.com/opencontainers/runc/libcontainer/user"
	"github.com/pkg/errors"
)

const defaultSocketGroup = "docker"

func lookupGID(name string) (int, error) {
	groupFile, err := user.GetGroupPath()
	if err != nil {
		return -1, errors.Wrap(err, "error looking up groups")
	}
	groups, err := user.ParseGroupFileFilter(groupFile, func(g user.Group) bool {
		return g.Name == name || strconv.Itoa(g.Gid) == name
	})
	if err != nil {
		return -1, errors.Wrapf(err, "error parsing groups for %s", name)
	}
	if len(groups) > 0 {
		return groups[0].Gid, nil
	}
	gid, err := strconv.Atoi(name)
	if err == nil {
		return gid, nil
	}
	return -1, fmt.Errorf("group %s not found", name)
}
