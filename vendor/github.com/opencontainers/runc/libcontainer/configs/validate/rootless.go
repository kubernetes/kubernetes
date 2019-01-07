package validate

import (
	"fmt"
	"strings"

	"github.com/opencontainers/runc/libcontainer/configs"
)

// rootlessEUID makes sure that the config can be applied when runc
// is being executed as a non-root user (euid != 0) in the current user namespace.
func (v *ConfigValidator) rootlessEUID(config *configs.Config) error {
	if err := rootlessEUIDMappings(config); err != nil {
		return err
	}
	if err := rootlessEUIDMount(config); err != nil {
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

func rootlessEUIDMappings(config *configs.Config) error {
	if !config.Namespaces.Contains(configs.NEWUSER) {
		return fmt.Errorf("rootless container requires user namespaces")
	}

	if len(config.UidMappings) == 0 {
		return fmt.Errorf("rootless containers requires at least one UID mapping")
	}
	if len(config.GidMappings) == 0 {
		return fmt.Errorf("rootless containers requires at least one GID mapping")
	}
	return nil
}

// mount verifies that the user isn't trying to set up any mounts they don't have
// the rights to do. In addition, it makes sure that no mount has a `uid=` or
// `gid=` option that doesn't resolve to root.
func rootlessEUIDMount(config *configs.Config) error {
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
