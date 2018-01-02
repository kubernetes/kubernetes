// Copyright 2016 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//+build linux

package stage0

import (
	"errors"
	"fmt"

	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/common"
	pkgPod "github.com/coreos/rkt/pkg/pod"
	"github.com/hashicorp/errwrap"
)

type StopConfig struct {
	*CommonConfig
	PodPath string
	AppName *types.ACName
	PodPID  int
}

func StopApp(cfg StopConfig) error {
	pod, err := pkgPod.PodFromUUIDString(cfg.DataDir, cfg.UUID.String())
	if err != nil {
		return errwrap.Wrap(errors.New("error loading pod manifest"), err)
	}
	defer pod.Close()

	pm, err := pod.SandboxManifest()
	if err != nil {
		return errwrap.Wrap(errors.New("cannot stop application"), err)
	}

	app := pm.Apps.Get(*cfg.AppName)
	if app == nil {
		return fmt.Errorf("error: nonexistent app %q", *cfg.AppName)
	}

	args := []string{
		fmt.Sprintf("--debug=%t", cfg.Debug),
		fmt.Sprintf("--app=%s", cfg.AppName),
	}

	ce := CrossingEntrypoint{
		PodPath:        cfg.PodPath,
		PodPID:         cfg.PodPID,
		AppName:        cfg.AppName.String(),
		EntrypointName: appStopEntrypoint,
		EntrypointArgs: args,
		Interactive:    false,
	}

	if err := ce.Run(); err != nil {
		status, err := common.GetExitStatus(err)
		// exit status 5 comes from systemctl and means the unit doesn't exist
		if status == 5 {
			return fmt.Errorf("app %q is not running", app.Name)
		}

		return err
	}

	return nil
}
