package validate

import (
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/opencontainers/runc/libcontainer/configs"
)

// rootlessEUIDCheck makes sure that the config can be applied when runc
// is being executed as a non-root user (euid != 0) in the current user namespace.
func rootlessEUIDCheck(config *configs.Config) error {
	if !config.RootlessEUID {
		return nil
	}
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

func rootlessEUIDMappings(config *configs.Config) error {
	if !config.Namespaces.Contains(configs.NEWUSER) {
		return errors.New("rootless container requires user namespaces")
	}
	// We only require mappings if we are not joining another userns.
	if path := config.Namespaces.PathOf(configs.NEWUSER); path == "" {
		if len(config.UIDMappings) == 0 {
			return errors.New("rootless containers requires at least one UID mapping")
		}
		if len(config.GIDMappings) == 0 {
			return errors.New("rootless containers requires at least one GID mapping")
		}
	}
	return nil
}

// rootlessEUIDMount verifies that all mounts have valid uid=/gid= options,
// i.e. their arguments has proper ID mappings.
func rootlessEUIDMount(config *configs.Config) error {
	// XXX: We could whitelist allowed devices at this point, but I'm not
	//      convinced that's a good idea. The kernel is the best arbiter of
	//      access control.

	for _, mount := range config.Mounts {
		// Check that the options list doesn't contain any uid= or gid= entries
		// that don't resolve to root.
		for _, opt := range strings.Split(mount.Data, ",") {
			if str := strings.TrimPrefix(opt, "uid="); len(str) < len(opt) {
				uid, err := strconv.Atoi(str)
				if err != nil {
					// Ignore unknown mount options.
					continue
				}
				if _, err := config.HostUID(uid); err != nil {
					return fmt.Errorf("cannot specify uid=%d mount option for rootless container: %w", uid, err)
				}
			}

			if str := strings.TrimPrefix(opt, "gid="); len(str) < len(opt) {
				gid, err := strconv.Atoi(str)
				if err != nil {
					// Ignore unknown mount options.
					continue
				}
				if _, err := config.HostGID(gid); err != nil {
					return fmt.Errorf("cannot specify gid=%d mount option for rootless container: %w", gid, err)
				}
			}
		}
	}

	return nil
}
