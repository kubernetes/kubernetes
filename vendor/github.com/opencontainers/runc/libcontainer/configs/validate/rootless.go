package validate

import (
	"fmt"
	"os"
	"reflect"
	"strings"

	"github.com/opencontainers/runc/libcontainer/configs"
)

var (
	geteuid = os.Geteuid
	getegid = os.Getegid
)

func (v *ConfigValidator) rootless(config *configs.Config) error {
	if err := rootlessMappings(config); err != nil {
		return err
	}
	if err := rootlessMount(config); err != nil {
		return err
	}

	// XXX: We currently can't verify the user config at all, because
	//      configs.Config doesn't store the user-related configs. So this
	//      has to be verified by setupUser() in init_linux.go.

	return nil
}

func hasIDMapping(id int, mappings []configs.IDMap) bool {
	for _, m := range mappings {
		if id >= m.ContainerID && id < m.ContainerID+m.Size {
			return true
		}
	}
	return false
}

func rootlessMappings(config *configs.Config) error {
	if euid := geteuid(); euid != 0 {
		if !config.Namespaces.Contains(configs.NEWUSER) {
			return fmt.Errorf("rootless containers require user namespaces")
		}
	}

	if len(config.UidMappings) == 0 {
		return fmt.Errorf("rootless containers requires at least one UID mapping")
	}
	if len(config.GidMappings) == 0 {
		return fmt.Errorf("rootless containers requires at least one UID mapping")
	}

	return nil
}

// cgroup verifies that the user isn't trying to set any cgroup limits or paths.
func rootlessCgroup(config *configs.Config) error {
	// Nothing set at all.
	if config.Cgroups == nil || config.Cgroups.Resources == nil {
		return nil
	}

	// Used for comparing to the zero value.
	left := reflect.ValueOf(*config.Cgroups.Resources)
	right := reflect.Zero(left.Type())

	// This is all we need to do, since specconv won't add cgroup options in
	// rootless mode.
	if !reflect.DeepEqual(left.Interface(), right.Interface()) {
		return fmt.Errorf("cannot specify resource limits in rootless container")
	}

	return nil
}

// mount verifies that the user isn't trying to set up any mounts they don't have
// the rights to do. In addition, it makes sure that no mount has a `uid=` or
// `gid=` option that doesn't resolve to root.
func rootlessMount(config *configs.Config) error {
	// XXX: We could whitelist allowed devices at this point, but I'm not
	//      convinced that's a good idea. The kernel is the best arbiter of
	//      access control.

	for _, mount := range config.Mounts {
		// Check that the options list doesn't contain any uid= or gid= entries
		// that don't resolve to root.
		for _, opt := range strings.Split(mount.Data, ",") {
			if strings.HasPrefix(opt, "uid=") {
				var uid int
				n, err := fmt.Sscanf(opt, "uid=%d", &uid)
				if n != 1 || err != nil {
					// Ignore unknown mount options.
					continue
				}
				if !hasIDMapping(uid, config.UidMappings) {
					return fmt.Errorf("cannot specify uid= mount options for unmapped uid in rootless containers")
				}
			}

			if strings.HasPrefix(opt, "gid=") {
				var gid int
				n, err := fmt.Sscanf(opt, "gid=%d", &gid)
				if n != 1 || err != nil {
					// Ignore unknown mount options.
					continue
				}
				if !hasIDMapping(gid, config.GidMappings) {
					return fmt.Errorf("cannot specify gid= mount options for unmapped gid in rootless containers")
				}
			}
		}
	}

	return nil
}
