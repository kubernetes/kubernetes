/*
Copyright 2015 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package rkt

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/capabilities"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/credentialprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	appcschema "github.com/appc/spec/schema"
	appctypes "github.com/appc/spec/schema/types"
	"github.com/coreos/go-systemd/dbus"
	"github.com/coreos/rkt/store"
	"github.com/golang/glog"
)

const (
	rktBinName = "rkt"

	acversion             = "0.5.1"
	rktMinimumVersion     = "0.5.4"
	systemdMinimumVersion = "215"

	systemdServiceDir      = "/run/systemd/system"
	rktDataDir             = "/var/lib/rkt"
	rktLocalConfigDir      = "/etc/rkt"
	rktMetadataServiceFile = "rkt-metadata.service"
	rktMetadataSocketFile  = "rkt-metadata.socket"

	kubernetesUnitPrefix  = "k8s"
	unitKubernetesSection = "X-Kubernetes"
	unitPodName           = "POD"
	unitRktID             = "RktID"

	dockerPrefix = "docker://"
)

// Runtime implements the ContainerRuntime for rkt. The implementation
// uses systemd, so in order to run this runtime, systemd must be installed
// on the machine.
type Runtime struct {
	systemd *dbus.Conn
	absPath string
	config  *Config
	// TODO(yifan): Refactor this to be generic keyring.
	dockerKeyring credentialprovider.DockerKeyring
}

// New creates the rkt container runtime which implements the container runtime interface.
// It will test if the rkt binary is in the $PATH, and whether we can get the
// version of it. If so, creates the rkt container runtime, otherwise returns an error.
func New(config *Config) (*Runtime, error) {
	systemdVersion, err := getSystemdVersion()
	if err != nil {
		return nil, err
	}
	result, err := systemdVersion.Compare(systemdMinimumVersion)
	if err != nil {
		return nil, err
	}
	if result < 0 {
		return nil, fmt.Errorf("rkt: systemd version is too old, requires at least %v", systemdMinimumVersion)
	}

	systemd, err := dbus.New()
	if err != nil {
		return nil, fmt.Errorf("cannot connect to dbus: %v", err)
	}

	// Test if rkt binary is in $PATH.
	absPath, err := exec.LookPath(rktBinName)
	if err != nil {
		return nil, fmt.Errorf("cannot find rkt binary: %v", err)
	}

	rkt := &Runtime{
		systemd:       systemd,
		absPath:       absPath,
		config:        config,
		dockerKeyring: credentialprovider.NewDockerKeyring(),
	}

	// Test the rkt version.
	version, err := rkt.Version()
	if err != nil {
		return nil, err
	}
	result, err = version.Compare(rktMinimumVersion)
	if err != nil {
		return nil, err
	}
	if result < 0 {
		return nil, fmt.Errorf("rkt: Version is too old, requires at least %v", rktMinimumVersion)
	}
	return rkt, nil
}

func (r *Runtime) buildCommand(args ...string) *exec.Cmd {
	cmd := exec.Command(rktBinName)
	cmd.Args = append(cmd.Args, r.config.buildGlobalOptions()...)
	cmd.Args = append(cmd.Args, args...)
	return cmd
}

// runCommand invokes rkt binary with arguments and returns the result
// from stdout in a list of strings.
func (r *Runtime) runCommand(args ...string) ([]string, error) {
	glog.V(4).Info("rkt: Run command:", args)

	output, err := r.buildCommand(args...).Output()
	if err != nil {
		return nil, err
	}
	return strings.Split(strings.TrimSpace(string(output)), "\n"), nil
}

// makePodServiceFileName constructs the unit file name for a pod using its UID.
func makePodServiceFileName(uid types.UID) string {
	// TODO(yifan): Revisit this later, decide whether we want to use UID.
	return fmt.Sprintf("%s_%s.service", kubernetesUnitPrefix, uid)
}

type resource struct {
	limit   string
	request string
}

// rawValue converts a string to *json.RawMessage
func rawValue(value string) *json.RawMessage {
	msg := json.RawMessage(value)
	return &msg
}

// setIsolators overrides the isolators of the pod manifest if necessary.
func setIsolators(app *appctypes.App, c *api.Container) error {
	if len(c.Capabilities.Add) > 0 || len(c.Capabilities.Drop) > 0 || len(c.Resources.Limits) > 0 || len(c.Resources.Requests) > 0 {
		app.Isolators = []appctypes.Isolator{}
	}

	// Retained capabilities/privileged.
	privileged := false
	if capabilities.Get().AllowPrivileged {
		privileged = c.Privileged
	} else if c.Privileged {
		return fmt.Errorf("privileged is disallowed globally")
	}
	var addCaps string
	if privileged {
		addCaps = getAllCapabilities()
	} else {
		addCaps = getCapabilities(c.Capabilities.Add)
	}
	if len(addCaps) > 0 {
		// TODO(yifan): Replace with constructor, see:
		// https://github.com/appc/spec/issues/268
		isolator := appctypes.Isolator{
			Name:     "os/linux/capabilities-retain-set",
			ValueRaw: rawValue(fmt.Sprintf(`{"set":[%s]}`, addCaps)),
		}
		app.Isolators = append(app.Isolators, isolator)
	}

	// Removed capabilities.
	dropCaps := getCapabilities(c.Capabilities.Drop)
	if len(dropCaps) > 0 {
		// TODO(yifan): Replace with constructor, see:
		// https://github.com/appc/spec/issues/268
		isolator := appctypes.Isolator{
			Name:     "os/linux/capabilities-remove-set",
			ValueRaw: rawValue(fmt.Sprintf(`{"set":[%s]}`, dropCaps)),
		}
		app.Isolators = append(app.Isolators, isolator)
	}

	// Resources.
	resources := make(map[api.ResourceName]resource)
	for name, quantity := range c.Resources.Limits {
		resources[name] = resource{limit: quantity.String()}
	}
	for name, quantity := range c.Resources.Requests {
		r, ok := resources[name]
		if !ok {
			r = resource{}
		}
		r.request = quantity.String()
		resources[name] = r
	}
	var acName appctypes.ACName
	for name, res := range resources {
		switch name {
		case api.ResourceCPU:
			acName = "resource/cpu"
		case api.ResourceMemory:
			acName = "resource/memory"
		default:
			return fmt.Errorf("resource type not supported: %v", name)
		}
		// TODO(yifan): Replace with constructor, see:
		// https://github.com/appc/spec/issues/268
		isolator := appctypes.Isolator{
			Name:     acName,
			ValueRaw: rawValue(fmt.Sprintf(`{"request":%q,"limit":%q}`, res.request, res.limit)),
		}
		app.Isolators = append(app.Isolators, isolator)
	}
	return nil
}

// setApp overrides the app's fields if any of them are specified in the
// container's spec.
func setApp(app *appctypes.App, c *api.Container) error {
	// Override the exec.
	// TOOD(yifan): Revisit this for the overriding rule.
	if len(c.Command) > 0 || len(c.Args) > 0 {
		app.Exec = append(c.Command, c.Args...)
	}

	// TODO(yifan): Use non-root user in the future, see:
	// https://github.com/coreos/rkt/issues/820
	app.User, app.Group = "0", "0"

	// Override the working directory.
	if len(c.WorkingDir) > 0 {
		app.WorkingDirectory = c.WorkingDir
	}

	// Override the environment.
	// TODO(yifan): Use RunContainerOptions.
	if len(c.Env) > 0 {
		app.Environment = []appctypes.EnvironmentVariable{}
	}
	for _, env := range c.Env {
		app.Environment = append(app.Environment, appctypes.EnvironmentVariable{
			Name:  env.Name,
			Value: env.Value,
		})
	}

	// Override the mount points.
	if len(c.VolumeMounts) > 0 {
		app.MountPoints = []appctypes.MountPoint{}
	}
	for _, m := range c.VolumeMounts {
		mountPointName, err := appctypes.NewACName(m.Name)
		if err != nil {
			return err
		}
		app.MountPoints = append(app.MountPoints, appctypes.MountPoint{
			Name:     *mountPointName,
			Path:     m.MountPath,
			ReadOnly: m.ReadOnly,
		})
	}

	// Override the ports.
	if len(c.Ports) > 0 {
		app.Ports = []appctypes.Port{}
	}
	for _, p := range c.Ports {
		portName, err := appctypes.NewACName(p.Name)
		if err != nil {
			return err
		}
		app.Ports = append(app.Ports, appctypes.Port{
			Name:     *portName,
			Protocol: string(p.Protocol),
			Port:     uint(p.ContainerPort),
		})
	}

	// Override isolators.
	return setIsolators(app, c)
}

// makePodManifest transforms a kubelet pod spec to the rkt pod manifest.
// TODO(yifan): Use the RunContainerOptions generated by GenerateRunContainerOptions().
func (r *Runtime) makePodManifest(pod *api.Pod, volumeMap map[string]volume.Volume) (*appcschema.PodManifest, error) {
	manifest := appcschema.BlankPodManifest()

	// Get the image manifests, assume they are already in the cas,
	// and extract the app field from the image and to be the 'base app'.
	//
	// We do this is because we will fully replace the image manifest's app
	// with the pod manifest's app in rkt runtime. See below:
	//
	// https://github.com/coreos/rkt/issues/723.
	//
	s, err := store.NewStore(rktDataDir)
	if err != nil {
		return nil, fmt.Errorf("cannot open store: %v", err)
	}
	for _, c := range pod.Spec.Containers {
		// Assume we are running docker images for now, see #7203.
		imageID, err := r.getImageID(c.Image)
		if err != nil {
			return nil, fmt.Errorf("cannot get image ID for %q: %v", c.Image, err)
		}
		hash, err := appctypes.NewHash(imageID)
		if err != nil {
			return nil, err
		}

		im, err := s.GetImageManifest(hash.String())
		if err != nil {
			return nil, fmt.Errorf("cannot get image manifest: %v", err)
		}

		// Override the image manifest's app and store it in the pod manifest.
		app := im.App
		if err := setApp(app, &c); err != nil {
			return nil, err
		}
		manifest.Apps = append(manifest.Apps, appcschema.RuntimeApp{
			Name:  im.Name,
			Image: appcschema.RuntimeImage{ID: *hash},
			App:   app,
		})
	}

	// Set global volumes.
	for name, volume := range volumeMap {
		volName, err := appctypes.NewACName(name)
		if err != nil {
			return nil, fmt.Errorf("cannot use the volume's name %q as ACName: %v", name, err)
		}
		manifest.Volumes = append(manifest.Volumes, appctypes.Volume{
			Name:   *volName,
			Kind:   "host",
			Source: volume.GetPath(),
		})
	}

	// Set global ports.
	for _, c := range pod.Spec.Containers {
		for _, port := range c.Ports {
			portName, err := appctypes.NewACName(port.Name)
			if err != nil {
				return nil, fmt.Errorf("cannot use the port's name %q as ACName: %v", port.Name, err)
			}
			manifest.Ports = append(manifest.Ports, appctypes.ExposedPort{
				Name:     *portName,
				HostPort: uint(port.HostPort),
			})
		}
	}
	// TODO(yifan): Set pod-level isolators once it's supported in kubernetes.
	return manifest, nil
}

// TODO(yifan): Replace with 'rkt images'.
func (r *Runtime) getImageID(imageName string) (string, error) {
	output, err := r.runCommand("fetch", imageName)
	if err != nil {
		return "", err
	}
	if len(output) == 0 {
		return "", fmt.Errorf("no result from rkt fetch")
	}
	last := output[len(output)-1]
	if !strings.HasPrefix(last, "sha512-") {
		return "", fmt.Errorf("unexpected result: %q", last)
	}
	return last, nil
}
