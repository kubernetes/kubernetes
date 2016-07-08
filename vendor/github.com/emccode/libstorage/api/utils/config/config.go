package config

import (
	"path"
	"time"

	log "github.com/Sirupsen/logrus"

	"github.com/akutz/gofig"
	"github.com/akutz/gotil"
	"github.com/emccode/libstorage/api/types"
	"github.com/emccode/libstorage/api/utils"
)

// NewConfig returns a new configuration instance.
func NewConfig() (gofig.Config, error) {
	config := gofig.New()

	etcYML := types.Etc.Join("config.yml")
	etcYAML := types.Etc.Join("config.yaml")

	userHomeDir := gotil.HomeDir()
	usrYML := path.Join(userHomeDir, "config.yml")
	usrYAML := path.Join(userHomeDir, "config.yaml")

	if err := readConfigFile(config, etcYML); err != nil {
		return nil, err
	}
	if err := readConfigFile(config, etcYAML); err != nil {
		return nil, err
	}
	if err := readConfigFile(config, usrYML); err != nil {
		return nil, err
	}
	if err := readConfigFile(config, usrYAML); err != nil {
		return nil, err
	}

	types.BackCompat(config)

	return config, nil
}

// UpdateLogLevel updates the log level based on the config.
func UpdateLogLevel(config gofig.Config) {
	ll, err := log.ParseLevel(config.GetString(types.ConfigLogLevel))
	if err != nil {
		return
	}
	log.SetLevel(ll)
}

func readConfigFile(config gofig.Config, path string) error {
	if !gotil.FileExists(path) {
		return nil
	}
	return config.ReadConfigFile(path)
}

// DeviceAttachTimeout gets the configured device attach timeout.
func DeviceAttachTimeout(config gofig.Config) time.Duration {
	return utils.DeviceAttachTimeout(
		config.GetString(types.ConfigDeviceAttachTimeout))
}

// DeviceScanType gets the configured device scan type.
func DeviceScanType(config gofig.Config) types.DeviceScanType {
	return types.ParseDeviceScanType(config.GetInt(types.ConfigDeviceScanType))
}
