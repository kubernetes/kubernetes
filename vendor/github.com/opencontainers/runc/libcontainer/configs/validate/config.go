package validate

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/opencontainers/runc/libcontainer/configs"
)

type Validator interface {
	Validate(*configs.Config) error
}

func New() Validator {
	return &ConfigValidator{}
}

type ConfigValidator struct {
}

func (v *ConfigValidator) Validate(config *configs.Config) error {
	if err := v.rootfs(config); err != nil {
		return err
	}
	if err := v.network(config); err != nil {
		return err
	}
	if err := v.hostname(config); err != nil {
		return err
	}
	if err := v.security(config); err != nil {
		return err
	}
	if err := v.usernamespace(config); err != nil {
		return err
	}
	return nil
}

// rootfs validates the the rootfs is an absolute path and is not a symlink
// to the container's root filesystem.
func (v *ConfigValidator) rootfs(config *configs.Config) error {
	cleaned, err := filepath.Abs(config.Rootfs)
	if err != nil {
		return err
	}
	if cleaned, err = filepath.EvalSymlinks(cleaned); err != nil {
		return err
	}
	if config.Rootfs != cleaned {
		return fmt.Errorf("%s is not an absolute path or is a symlink", config.Rootfs)
	}
	return nil
}

func (v *ConfigValidator) network(config *configs.Config) error {
	if !config.Namespaces.Contains(configs.NEWNET) {
		if len(config.Networks) > 0 || len(config.Routes) > 0 {
			return fmt.Errorf("unable to apply network settings without a private NET namespace")
		}
	}
	return nil
}

func (v *ConfigValidator) hostname(config *configs.Config) error {
	if config.Hostname != "" && !config.Namespaces.Contains(configs.NEWUTS) {
		return fmt.Errorf("unable to set hostname without a private UTS namespace")
	}
	return nil
}

func (v *ConfigValidator) security(config *configs.Config) error {
	// restrict sys without mount namespace
	if (len(config.MaskPaths) > 0 || len(config.ReadonlyPaths) > 0) &&
		!config.Namespaces.Contains(configs.NEWNS) {
		return fmt.Errorf("unable to restrict sys entries without a private MNT namespace")
	}
	return nil
}

func (v *ConfigValidator) usernamespace(config *configs.Config) error {
	if config.Namespaces.Contains(configs.NEWUSER) {
		if _, err := os.Stat("/proc/self/ns/user"); os.IsNotExist(err) {
			return fmt.Errorf("USER namespaces aren't enabled in the kernel")
		}
	} else {
		if config.UidMappings != nil || config.GidMappings != nil {
			return fmt.Errorf("User namespace mappings specified, but USER namespace isn't enabled in the config")
		}
	}
	return nil
}
