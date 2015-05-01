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
	"hash/adler32"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"strings"
	"syscall"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/capabilities"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/credentialprovider"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/prober"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	appcschema "github.com/appc/spec/schema"
	appctypes "github.com/appc/spec/schema/types"
	"github.com/coreos/go-systemd/dbus"
	"github.com/coreos/go-systemd/unit"
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
	generator        kubecontainer.RunContainerOptionsGenerator
	readinessManager *kubecontainer.ReadinessManager
	prober           prober.Prober
	systemd          *dbus.Conn
	// The absolute path to rkt binary.
	rktBinAbsPath string
	config        *Config
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
	rktBinAbsPath, err := exec.LookPath(rktBinName)
	if err != nil {
		return nil, fmt.Errorf("cannot find rkt binary: %v", err)
	}

	rkt := &Runtime{
		systemd:       systemd,
		rktBinAbsPath: rktBinAbsPath,
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

func newUnitOption(section, name, value string) *unit.UnitOption {
	return &unit.UnitOption{Section: section, Name: name, Value: value}
}

// TODO(yifan): Move this duplicated function to container runtime.
// hashContainer computes the hash of one api.Container.
func hashContainer(container *api.Container) uint64 {
	hash := adler32.New()
	util.DeepHashObject(hash, *container)
	return uint64(hash.Sum32())
}

// TODO(yifan): Remove the receiver once we can solve the appName->imageID problem.
func (r *Runtime) apiPodToRuntimePod(uuid string, pod *api.Pod) *kubecontainer.Pod {
	p := &kubecontainer.Pod{
		ID:        pod.UID,
		Name:      pod.Name,
		Namespace: pod.Namespace,
	}
	for i := range pod.Spec.Containers {
		c := &pod.Spec.Containers[i]
		imageID, err := r.getImageID(c.Image)
		if err != nil {
			glog.Warningf("rkt: Cannot get image id: %v", err)
		}
		p.Containers = append(p.Containers, &kubecontainer.Container{
			ID:      types.UID(buildContainerID(&containerID{uuid, c.Name, imageID})),
			Name:    c.Name,
			Image:   c.Image,
			Hash:    hashContainer(c),
			Created: time.Now().Unix(),
		})
	}
	return p
}

// preparePod will:
//
// 1. Invoke 'rkt prepare' to prepare the pod, and get the rkt pod uuid.
// 2. Creates the unit file and save it under systemdUnitDir.
//
// On success, it will return a string that represents name of the unit file
// and a boolean that indicates if the unit file needs to be reloaded (whether
// the file is already existed).
func (r *Runtime) preparePod(pod *api.Pod, volumeMap map[string]volume.Volume) (string, bool, error) {
	cmds := []string{"prepare", "--quiet", "--pod-manifest"}

	// Generate the pod manifest from the pod spec.
	manifest, err := r.makePodManifest(pod, volumeMap)
	if err != nil {
		return "", false, err
	}
	manifestFile, err := ioutil.TempFile("", "manifest")
	if err != nil {
		return "", false, err
	}
	defer func() {
		manifestFile.Close()
		if err := os.Remove(manifestFile.Name()); err != nil {
			glog.Warningf("rkt: Cannot remove temp manifest file %q: %v", manifestFile.Name(), err)
		}
	}()

	data, err := json.Marshal(manifest)
	if err != nil {
		return "", false, err
	}
	// Since File.Write returns error if the written length is less than len(data),
	// so check error is enough for us.
	if _, err := manifestFile.Write(data); err != nil {
		return "", false, err
	}

	cmds = append(cmds, manifestFile.Name())
	output, err := r.runCommand(cmds...)
	if err != nil {
		return "", false, err
	}
	if len(output) != 1 {
		return "", false, fmt.Errorf("cannot get uuid from 'rkt prepare'")
	}
	uuid := output[0]
	glog.V(4).Infof("'rkt prepare' returns %q.", uuid)

	p := r.apiPodToRuntimePod(uuid, pod)
	b, err := json.Marshal(p)
	if err != nil {
		return "", false, err
	}

	runPrepared := fmt.Sprintf("%s run-prepared --private-net=%v %s", r.rktBinAbsPath, !pod.Spec.HostNetwork, uuid)
	units := []*unit.UnitOption{
		newUnitOption(unitKubernetesSection, unitRktID, uuid),
		newUnitOption(unitKubernetesSection, unitPodName, string(b)),
		newUnitOption("Service", "ExecStart", runPrepared),
	}

	// Save the unit file under systemd's service directory.
	// TODO(yifan) Garbage collect 'dead' service files.
	needReload := false
	unitName := makePodServiceFileName(pod.UID)
	if _, err := os.Stat(path.Join(systemdServiceDir, unitName)); err == nil {
		needReload = true
	}
	unitFile, err := os.Create(path.Join(systemdServiceDir, unitName))
	if err != nil {
		return "", false, err
	}
	defer unitFile.Close()

	_, err = io.Copy(unitFile, unit.Serialize(units))
	if err != nil {
		return "", false, err
	}
	return unitName, needReload, nil
}

// RunPod first creates the unit file for a pod, and then calls
// StartUnit over d-bus.
func (r *Runtime) RunPod(pod *api.Pod, volumeMap map[string]volume.Volume) error {
	glog.V(4).Infof("Rkt starts to run pod: name %q.", pod.Name)

	name, needReload, err := r.preparePod(pod, volumeMap)
	if err != nil {
		return err
	}
	if needReload {
		// TODO(yifan): More graceful stop. Replace with StopUnit and wait for a timeout.
		r.systemd.KillUnit(name, int32(syscall.SIGKILL))
		if err := r.systemd.Reload(); err != nil {
			return err
		}
	}

	// TODO(yifan): This is the old version of go-systemd. Should update when libcontainer updates
	// its version of go-systemd.
	_, err = r.systemd.StartUnit(name, "replace")
	if err != nil {
		return err
	}
	return nil
}

// makeRuntimePod constructs the container runtime pod. It will:
// 1, Construct the pod by the information stored in the unit file.
// 2, Construct the pod status from pod info.
func (r *Runtime) makeRuntimePod(unitName string, podInfos map[string]*podInfo) (*kubecontainer.Pod, error) {
	f, err := os.Open(path.Join(systemdServiceDir, unitName))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var pod kubecontainer.Pod
	opts, err := unit.Deserialize(f)
	if err != nil {
		return nil, err
	}

	var rktID string
	for _, opt := range opts {
		if opt.Section != unitKubernetesSection {
			continue
		}
		switch opt.Name {
		case unitPodName:
			err = json.Unmarshal([]byte(opt.Value), &pod)
			if err != nil {
				return nil, err
			}
		case unitRktID:
			rktID = opt.Value
		default:
			return nil, fmt.Errorf("rkt: Unexpected key: %q", opt.Name)
		}
	}

	if len(rktID) == 0 {
		return nil, fmt.Errorf("rkt: cannot find rkt ID of pod %v, unit file is broken", pod)
	}
	info, found := podInfos[rktID]
	if !found {
		return nil, fmt.Errorf("rkt: cannot find info for pod %q, rkt uuid: %q", pod.Name, rktID)
	}
	pod.Status = info.toPodStatus(&pod)
	return &pod, nil
}

// GetPods runs 'systemctl list-unit' and 'rkt list' to get the list of rkt pods.
// Then it will use the result to contruct a list of container runtime pods.
// If all is false, then only running pods will be returned, otherwise all pods will be
// returned.
func (r *Runtime) GetPods(all bool) ([]*kubecontainer.Pod, error) {
	glog.V(4).Infof("Rkt getting pods")

	units, err := r.systemd.ListUnits()
	if err != nil {
		return nil, err
	}

	// TODO(yifan): Now we are getting the status of the pod as well.
	// Probably we can leave much of the work to GetPodStatus().
	podInfos, err := r.getPodInfos()
	if err != nil {
		return nil, err
	}

	var pods []*kubecontainer.Pod
	for _, u := range units {
		if strings.HasPrefix(u.Name, kubernetesUnitPrefix) {
			if !all && u.SubState != "running" {
				continue
			}
			pod, err := r.makeRuntimePod(u.Name, podInfos)
			if err != nil {
				glog.Warningf("rkt: Cannot construct pod from unit file: %v.", err)
				continue
			}
			pods = append(pods, pod)
		}
	}
	return pods, nil
}

// KillPod invokes 'systemctl kill' to kill the unit that runs the pod.
func (r *Runtime) KillPod(pod kubecontainer.Pod) error {
	glog.V(4).Infof("Rkt is killing pod: name %q.", pod.Name)

	// TODO(yifan): More graceful stop. Replace with StopUnit and wait for a timeout.
	r.systemd.KillUnit(makePodServiceFileName(pod.ID), int32(syscall.SIGKILL))
	return r.systemd.Reload()
}

// GetPodStatus currently invokes GetPods() to return the status.
// TODO(yifan): Split the get status logic from GetPods().
func (r *Runtime) GetPodStatus(pod *api.Pod) (*api.PodStatus, error) {
	pods, err := r.GetPods(true)
	if err != nil {
		return nil, err
	}
	p := kubecontainer.Pods(pods).FindPodByID(pod.UID)
	if len(p.Containers) == 0 {
		return nil, fmt.Errorf("cannot find status for pod: %q", kubecontainer.BuildPodFullName(pod.Name, pod.Namespace))
	}
	return &p.Status, nil
}
