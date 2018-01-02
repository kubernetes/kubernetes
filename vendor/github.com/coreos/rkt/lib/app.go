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

package rkt

import (
	"fmt"
	"io/ioutil"
	"os"

	"github.com/appc/spec/schema"
	"github.com/coreos/rkt/common"
	pkgPod "github.com/coreos/rkt/pkg/pod"
)

// AppsForPod returns the apps of the pod with the given uuid in the given data directory.
// If appName is non-empty, then only the app with the given name will be returned.
func AppsForPod(uuid, dataDir string, appName string) ([]*App, error) {
	p, err := pkgPod.PodFromUUIDString(dataDir, uuid)
	if err != nil {
		return nil, err
	}
	defer p.Close()

	_, podManifest, err := p.PodManifest()
	if err != nil {
		return nil, err
	}

	var apps []*App
	for _, ra := range podManifest.Apps {
		if appName != "" && appName != ra.Name.String() {
			continue
		}

		app, err := newApp(&ra, podManifest, p)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Cannot get app status: %v", err)
			continue
		}

		apps = append(apps, app)
	}

	return apps, nil
}

// newApp constructs the App object with the runtime app and pod manifest.
func newApp(ra *schema.RuntimeApp, podManifest *schema.PodManifest, pod *pkgPod.Pod) (*App, error) {
	app := &App{
		Name:            ra.Name.String(),
		ImageID:         ra.Image.ID.String(),
		UserAnnotations: ra.App.UserAnnotations,
		UserLabels:      ra.App.UserLabels,
	}

	for _, mnt := range ra.Mounts {
		app.Mounts = append(app.Mounts, &Mount{
			Name:          mnt.Volume.String(),
			ContainerPath: mnt.Path,
			HostPath:      mnt.AppVolume.Source,
			ReadOnly:      *mnt.AppVolume.ReadOnly,
		})
	}

	// Generate state.
	if err := appState(app, pod); err != nil {
		return nil, fmt.Errorf("error getting app's state: %v", err)
	}

	return app, nil
}

func appState(app *App, pod *pkgPod.Pod) error {
	app.State = AppStateUnknown

	defer func() {
		if pod.IsAfterRun() {
			// If the pod is hard killed, set the app to 'exited' state.
			// Other than this case, status file is guaranteed to be written.
			if app.State != AppStateExited {
				app.State = AppStateExited
				t, err := pod.GCMarkedTime()
				if err != nil {
					fmt.Fprintf(os.Stderr, "Cannot get GC marked time: %v", err)
				}
				if !t.IsZero() {
					finishedAt := t.UnixNano()
					app.FinishedAt = &finishedAt
				}
			}
		}
	}()

	// Check if the app is created.
	fi, err := os.Stat(common.AppCreatedPath(pod.Path(), app.Name))
	if err != nil {
		if !os.IsNotExist(err) {
			return fmt.Errorf("cannot stat app creation file: %v", err)
		}
		return nil
	}

	app.State = AppStateCreated
	createdAt := fi.ModTime().UnixNano()
	app.CreatedAt = &createdAt

	// Check if the app is started.
	fi, err = os.Stat(common.AppStartedPath(pod.Path(), app.Name))
	if err != nil {
		if !os.IsNotExist(err) {
			return fmt.Errorf("cannot stat app started file: %v", err)
		}
		return nil
	}

	app.State = AppStateRunning
	startedAt := fi.ModTime().UnixNano()
	app.StartedAt = &startedAt

	// Check if the app is exited.
	appStatusFile := common.AppStatusPath(pod.Path(), app.Name)
	fi, err = os.Stat(appStatusFile)
	if err != nil {
		if !os.IsNotExist(err) {
			return fmt.Errorf("cannot stat app exited file: %v", err)
		}
		return nil
	}

	app.State = AppStateExited
	finishedAt := fi.ModTime().UnixNano()
	app.FinishedAt = &finishedAt

	// Read exit code.
	exitCode, err := readExitCode(appStatusFile)
	if err != nil {
		return err
	}
	app.ExitCode = &exitCode

	return nil
}

func readExitCode(path string) (int32, error) {
	var exitCode int32

	b, err := ioutil.ReadFile(path)
	if err != nil {
		return -1, fmt.Errorf("cannot read app exited file: %v", err)
	}
	if _, err := fmt.Sscanf(string(b), "%d", &exitCode); err != nil {
		return -1, fmt.Errorf("cannot parse exit code: %v", err)
	}
	return exitCode, nil
}
