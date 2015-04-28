package types

import (
	"errors"
	"fmt"
	"net/url"
	"strconv"
	"strings"
)

type MountPoint struct {
	Name     ACName `json:"name"`
	Path     string `json:"path"`
	ReadOnly bool   `json:"readOnly,omitempty"`
}

func (mount MountPoint) assertValid() error {
	if mount.Name.Empty() {
		return errors.New("name must be set")
	}
	if len(mount.Path) == 0 {
		return errors.New("path must be set")
	}
	return nil
}

// MountPointFromString takes a command line mountpoint parameter and returns a mountpoint
//
// It is useful for actool patch-manifest --mounts
//
// Example mountpoint parameters:
// 	database,path=/tmp,readOnly=true
func MountPointFromString(mp string) (*MountPoint, error) {
	var mount MountPoint

	mp = "name=" + mp
	v, err := url.ParseQuery(strings.Replace(mp, ",", "&", -1))
	if err != nil {
		return nil, err
	}
	for key, val := range v {
		if len(val) > 1 {
			return nil, fmt.Errorf("label %s with multiple values %q", key, val)
		}

		// TOOD(philips): make this less hardcoded
		switch key {
		case "name":
			acn, err := NewACName(val[0])
			if err != nil {
				return nil, err
			}
			mount.Name = *acn
		case "path":
			mount.Path = val[0]
		case "readOnly":
			ro, err := strconv.ParseBool(val[0])
			if err != nil {
				return nil, err
			}
			mount.ReadOnly = ro
		default:
			return nil, fmt.Errorf("unknown mountpoint parameter %q", key)
		}
	}
	err = mount.assertValid()
	if err != nil {
		return nil, err
	}

	return &mount, nil
}
