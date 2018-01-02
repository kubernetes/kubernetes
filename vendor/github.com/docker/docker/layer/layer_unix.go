// +build linux freebsd darwin openbsd solaris

package layer

import "github.com/docker/docker/pkg/stringid"

func (ls *layerStore) mountID(name string) string {
	return stringid.GenerateRandomID()
}
