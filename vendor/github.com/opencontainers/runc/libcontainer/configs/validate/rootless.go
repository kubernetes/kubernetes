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
	// Currently, cgroups cannot effectively be used in rootless containers.
	// The new cgroup namespace doesn't really help us either because it doesn't
	// have nice interactions with the user namespace (we're working with upstream
	// to fix this).
	if err := rootlessCgroup(config); err != nil {
		return err
	}

	// XXX: We currently can't verify the user config at all, because
	//      configs.Config doesn't store the user-related configs. So this
	//      has to be verified by setupUser() in init_linux.go.

	return nil
}

func rootlessMappings(config *configs.Config) error {
	rootuid, err := config.HostRootUID()
	if err != nil {
		return fmt.Errorf("failed to get root uid from uidMappings: %v", err)
	}
	if euid := geteuid(); euid != 0 {
		if !config.Namespaces.Contains(configs.NEWUSER) {
			return fmt.Errorf("rootless containers require user namespaces")
		}
		if rootuid != euid {
			return fmt.Errorf("rootless containers cannot map container root to a different host user")
		}
	}

	rootgid, err := config.HostRootGID()
	if err != nil {
		return fmt.Errorf("failed to get root gid from gidMappings: %v", err)
	}

	// Similar to the above test, we need to make sure that we aren't trying to
	// map to a group ID that we don't have the right to be.
	if rootgid != getegid() {
		return fmt.Errorf("rootless containers cannot map container root to a different host group")
	}

	// We can only map one user and group inside a container (our own).
	if len(config.UidMappings) != 1 || config.UidMappings[0].Size != 1 {
		return fmt.Errorf("rootless containers cannot map more than one user")
	}
	if len(config.GidMappings) != 1 || config.GidMappings[0].Size != 1 {
		return fmt.Errorf("rootless containers cannot map more than one group")
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
			if strings.HasPrefix(opt, "uid=") && opt != "uid=0" {
				return fmt.Errorf("cannot specify uid= mount options in rootless containers where argument isn't 0")
			}
			if strings.HasPrefix(opt, "gid=") && opt != "gid=0" {
				return fmt.Errorf("cannot specify gid= mount options in rootless containers where argument isn't 0")
			}
		}
	}

	return nil
}
