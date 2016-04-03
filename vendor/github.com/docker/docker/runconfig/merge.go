package runconfig

import (
	"strings"

	"github.com/docker/docker/pkg/nat"
)

func Merge(userConf, imageConf *Config) error {
	if userConf.User == "" {
		userConf.User = imageConf.User
	}
	if len(userConf.ExposedPorts) == 0 {
		userConf.ExposedPorts = imageConf.ExposedPorts
	} else if imageConf.ExposedPorts != nil {
		if userConf.ExposedPorts == nil {
			userConf.ExposedPorts = make(nat.PortSet)
		}
		for port := range imageConf.ExposedPorts {
			if _, exists := userConf.ExposedPorts[port]; !exists {
				userConf.ExposedPorts[port] = struct{}{}
			}
		}
	}

	if len(userConf.Env) == 0 {
		userConf.Env = imageConf.Env
	} else {
		for _, imageEnv := range imageConf.Env {
			found := false
			imageEnvKey := strings.Split(imageEnv, "=")[0]
			for _, userEnv := range userConf.Env {
				userEnvKey := strings.Split(userEnv, "=")[0]
				if imageEnvKey == userEnvKey {
					found = true
					break
				}
			}
			if !found {
				userConf.Env = append(userConf.Env, imageEnv)
			}
		}
	}

	if userConf.Entrypoint.Len() == 0 {
		if userConf.Cmd.Len() == 0 {
			userConf.Cmd = imageConf.Cmd
		}

		if userConf.Entrypoint == nil {
			userConf.Entrypoint = imageConf.Entrypoint
		}
	}
	if userConf.WorkingDir == "" {
		userConf.WorkingDir = imageConf.WorkingDir
	}
	if len(userConf.Volumes) == 0 {
		userConf.Volumes = imageConf.Volumes
	} else {
		for k, v := range imageConf.Volumes {
			userConf.Volumes[k] = v
		}
	}
	return nil
}
