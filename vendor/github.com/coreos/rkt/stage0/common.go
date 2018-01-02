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

package stage0

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/common/apps"
	stage1types "github.com/coreos/rkt/stage1/common/types"
	"github.com/hashicorp/errwrap"
	"strconv"
)

// CrossingEntrypoint represents a stage1 entrypoint whose execution
// needs to cross the stage0/stage1/stage2 boundary.
type CrossingEntrypoint struct {
	PodPath        string
	PodPID         int
	AppName        string
	EntrypointName string
	EntrypointArgs []string
	Interactive    bool
}

// Run wraps the execution of a stage1 entrypoint which
// requires crossing the stage0/stage1/stage2 boundary during its execution,
// by setting up proper environment variables for enter.
func (ce CrossingEntrypoint) Run() error {
	enterCmd, err := getStage1Entrypoint(ce.PodPath, enterEntrypoint)
	if err != nil {
		return errwrap.Wrap(errors.New("error determining 'enter' entrypoint"), err)
	}

	previousDir, err := os.Getwd()
	if err != nil {
		return err
	}

	if err := os.Chdir(ce.PodPath); err != nil {
		return errwrap.Wrap(errors.New("failed changing to dir"), err)
	}

	ep, err := getStage1Entrypoint(ce.PodPath, ce.EntrypointName)
	if err != nil {
		return fmt.Errorf("%q not implemented for pod's stage1: %v", ce.EntrypointName, err)
	}
	execArgs := []string{filepath.Join(common.Stage1RootfsPath(ce.PodPath), ep)}
	execArgs = append(execArgs, ce.EntrypointArgs...)

	pathEnv := os.Getenv("PATH")
	if pathEnv == "" {
		pathEnv = common.DefaultPath
	}
	execEnv := []string{
		fmt.Sprintf("%s=%s", common.CrossingEnterCmd, filepath.Join(common.Stage1RootfsPath(ce.PodPath), enterCmd)),
		fmt.Sprintf("%s=%d", common.CrossingEnterPID, ce.PodPID),
		fmt.Sprintf("PATH=%s", pathEnv),
	}

	c := exec.Cmd{
		Path: execArgs[0],
		Args: execArgs,
		Env:  execEnv,
	}

	if ce.Interactive {
		c.Stdin = os.Stdin
		c.Stdout = os.Stdout
		c.Stderr = os.Stderr
		if err := c.Run(); err != nil {
			return fmt.Errorf("error executing stage1 entrypoint: %v", err)
		}
	} else {
		out, err := c.CombinedOutput()
		if len(out) > 0 {
			debug("%s\n", out)
		}

		if err != nil {
			return errwrap.Wrapf("error executing stage1 entrypoint", err)
		}
	}

	if err := os.Chdir(previousDir); err != nil {
		return errwrap.Wrap(errors.New("failed changing to dir"), err)
	}

	return nil
}

// generateRuntimeApp merges runtime information from the image manifest and from
// runtime configuration overrides, returning a full configuration for a runtime app
func generateRuntimeApp(appRunConfig *apps.App, am *schema.ImageManifest, podMounts []schema.Mount) (schema.RuntimeApp, error) {

	ra := schema.RuntimeApp{
		App: am.App,
		Image: schema.RuntimeImage{
			Name:   &am.Name,
			ID:     appRunConfig.ImageID,
			Labels: am.Labels,
		},
		Mounts:         MergeMounts(podMounts, appRunConfig.Mounts),
		ReadOnlyRootFS: appRunConfig.ReadOnlyRootFS,
	}

	appName, err := types.NewACName(appRunConfig.Name)
	if err != nil {
		return ra, errwrap.Wrap(errors.New("invalid app name format"), err)
	}
	ra.Name = *appName

	if appRunConfig.Exec != "" {
		// Create a minimal App section if not present
		if am.App == nil {
			ra.App = &types.App{
				User:  strconv.Itoa(os.Getuid()),
				Group: strconv.Itoa(os.Getgid()),
			}
		}
		ra.App.Exec = []string{appRunConfig.Exec}
	}

	if appRunConfig.Args != nil {
		ra.App.Exec = append(ra.App.Exec, appRunConfig.Args...)
	}

	if appRunConfig.WorkingDir != "" {
		ra.App.WorkingDirectory = appRunConfig.WorkingDir
	}

	if err := prepareIsolators(appRunConfig, ra.App); err != nil {
		return ra, err
	}

	if appRunConfig.User != "" {
		ra.App.User = appRunConfig.User
	}

	if appRunConfig.Group != "" {
		ra.App.Group = appRunConfig.Group
	}

	if appRunConfig.SupplementaryGIDs != nil {
		ra.App.SupplementaryGIDs = appRunConfig.SupplementaryGIDs
	}

	if appRunConfig.UserAnnotations != nil {
		ra.App.UserAnnotations = appRunConfig.UserAnnotations
	}

	if appRunConfig.UserLabels != nil {
		ra.App.UserLabels = appRunConfig.UserLabels
	}

	if appRunConfig.Stdin != "" {
		ra.Annotations.Set(stage1types.AppStdinMode, appRunConfig.Stdin.String())
	}
	if appRunConfig.Stdout != "" {
		ra.Annotations.Set(stage1types.AppStdoutMode, appRunConfig.Stdout.String())
	}
	if appRunConfig.Stderr != "" {
		ra.Annotations.Set(stage1types.AppStderrMode, appRunConfig.Stderr.String())
	}

	if appRunConfig.Environments != nil {
		envs := make([]string, 0, len(appRunConfig.Environments))
		for name, value := range appRunConfig.Environments {
			envs = append(envs, fmt.Sprintf("%s=%s", name, value))
		}
		// Let the app level environment override the environment variables.
		mergeEnvs(&ra.App.Environment, envs, true)
	}

	return ra, nil
}
