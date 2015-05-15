/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/credentialprovider"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/prober"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/probe"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/securitycontext"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	appcschema "github.com/appc/spec/schema"
	appctypes "github.com/appc/spec/schema/types"
	"github.com/coreos/go-systemd/dbus"
	"github.com/coreos/go-systemd/unit"
	"github.com/docker/docker/pkg/parsers"
	docker "github.com/fsouza/go-dockerclient"
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

	authDir            = "auth.d"
	dockerAuthTemplate = `{"rktKind":"dockerAuth","rktVersion":"v1","registries":[%q],"credentials":{"user":%q,"password":%q}}`

	// TODO(yifan): Merge with ContainerGCPolicy, i.e., derive
	// the grace period from MinAge in ContainerGCPolicy.
	//
	// Duration to wait before discarding inactive pods from garbage
	defaultGracePeriod = "1m"
	// Duration to wait before expiring prepared pods.
	defaultExpirePrepared = "1m"
)

// runtime implements the Containerruntime for rkt. The implementation
// uses systemd, so in order to run this runtime, systemd must be installed
// on the machine.
type runtime struct {
	systemd *dbus.Conn
	// The absolute path to rkt binary.
	rktBinAbsPath string
	config        *Config
	// TODO(yifan): Refactor this to be generic keyring.
	dockerKeyring credentialprovider.DockerKeyring

	containerRefManager *kubecontainer.RefManager
	generator           kubecontainer.RunContainerOptionsGenerator
	recorder            record.EventRecorder
	prober              prober.Prober
	readinessManager    *kubecontainer.ReadinessManager
	volumeGetter        volumeGetter
}

var _ kubecontainer.Runtime = &runtime{}

// TODO(yifan): Remove this when volumeManager is moved to separate package.
type volumeGetter interface {
	GetVolumes(podUID types.UID) (kubecontainer.VolumeMap, bool)
}

// New creates the rkt container runtime which implements the container runtime interface.
// It will test if the rkt binary is in the $PATH, and whether we can get the
// version of it. If so, creates the rkt container runtime, otherwise returns an error.
func New(config *Config,
	generator kubecontainer.RunContainerOptionsGenerator,
	recorder record.EventRecorder,
	containerRefManager *kubecontainer.RefManager,
	readinessManager *kubecontainer.ReadinessManager,
	volumeGetter volumeGetter) (kubecontainer.Runtime, error) {

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

	rkt := &runtime{
		systemd:             systemd,
		rktBinAbsPath:       rktBinAbsPath,
		config:              config,
		dockerKeyring:       credentialprovider.NewDockerKeyring(),
		containerRefManager: containerRefManager,
		generator:           generator,
		recorder:            recorder,
		readinessManager:    readinessManager,
		volumeGetter:        volumeGetter,
	}
	rkt.prober = prober.New(rkt, readinessManager, containerRefManager, recorder)

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

func (r *runtime) buildCommand(args ...string) *exec.Cmd {
	cmd := exec.Command(rktBinName)
	cmd.Args = append(cmd.Args, r.config.buildGlobalOptions()...)
	cmd.Args = append(cmd.Args, args...)
	return cmd
}

// runCommand invokes rkt binary with arguments and returns the result
// from stdout in a list of strings. Each string in the list is a line.
func (r *runtime) runCommand(args ...string) ([]string, error) {
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

// rawValue converts the request, limit to *json.RawMessage
func rawRequestLimit(request, limit string) *json.RawMessage {
	if request == "" {
		return rawValue(fmt.Sprintf(`{"limit":%q}`, limit))
	}
	if limit == "" {
		return rawValue(fmt.Sprintf(`{"request":%q}`, request))
	}
	return rawValue(fmt.Sprintf(`{"request":%q,"limit":%q}`, request, limit))
}

// setIsolators overrides the isolators of the pod manifest if necessary.
// TODO need an apply config in security context for rkt
func setIsolators(app *appctypes.App, c *api.Container) error {
	hasCapRequests := securitycontext.HasCapabilitiesRequest(c)
	if hasCapRequests || len(c.Resources.Limits) > 0 || len(c.Resources.Requests) > 0 {
		app.Isolators = []appctypes.Isolator{}
	}

	// Retained capabilities/privileged.
	privileged := false
	if c.SecurityContext != nil && c.SecurityContext.Privileged != nil {
		privileged = *c.SecurityContext.Privileged
	}

	var addCaps string
	if privileged {
		addCaps = getAllCapabilities()
	} else {
		if hasCapRequests {
			addCaps = getCapabilities(c.SecurityContext.Capabilities.Add)
		}
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
	var dropCaps string
	if hasCapRequests {
		dropCaps = getCapabilities(c.SecurityContext.Capabilities.Drop)
	}
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
			ValueRaw: rawRequestLimit(res.request, res.limit),
		}
		app.Isolators = append(app.Isolators, isolator)
	}
	return nil
}

// setApp overrides the app's fields if any of them are specified in the
// container's spec.
func setApp(app *appctypes.App, c *api.Container, opts *kubecontainer.RunContainerOptions) error {
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
	if len(opts.Envs) > 0 {
		app.Environment = []appctypes.EnvironmentVariable{}
	}
	for _, env := range c.Env {
		app.Environment = append(app.Environment, appctypes.EnvironmentVariable{
			Name:  env.Name,
			Value: env.Value,
		})
	}

	// Override the mount points.
	if len(opts.Mounts) > 0 {
		app.MountPoints = []appctypes.MountPoint{}
	}
	for _, m := range opts.Mounts {
		mountPointName, err := appctypes.NewACName(m.Name)
		if err != nil {
			return err
		}
		app.MountPoints = append(app.MountPoints, appctypes.MountPoint{
			Name:     *mountPointName,
			Path:     m.ContainerPath,
			ReadOnly: m.ReadOnly,
		})
	}

	// Override the ports.
	if len(opts.PortMappings) > 0 {
		app.Ports = []appctypes.Port{}
	}
	for _, p := range opts.PortMappings {
		name, err := appctypes.SanitizeACName(p.Name)
		if err != nil {
			return err
		}
		portName := appctypes.MustACName(name)
		app.Ports = append(app.Ports, appctypes.Port{
			Name:     *portName,
			Protocol: string(p.Protocol),
			Port:     uint(p.ContainerPort),
		})
	}

	// Override isolators.
	return setIsolators(app, c)
}

// getImageManifest invokes 'rkt image cat-manifest' to retrive the image manifest
// for the image.
func (r *runtime) getImageManifest(image string) (*appcschema.ImageManifest, error) {
	var manifest appcschema.ImageManifest

	// TODO(yifan): Assume docker images for now.
	output, err := r.runCommand("image", "cat-manifest", "--quiet", dockerPrefix+image)
	if err != nil {
		return nil, err
	}
	if len(output) != 1 {
		return nil, fmt.Errorf("invalid output: %v", output)
	}
	return &manifest, json.Unmarshal([]byte(output[0]), &manifest)
}

// makePodManifest transforms a kubelet pod spec to the rkt pod manifest.
// TODO(yifan): Use the RunContainerOptions generated by GenerateRunContainerOptions().
func (r *runtime) makePodManifest(pod *api.Pod) (*appcschema.PodManifest, error) {
	var globalPortMappings []kubecontainer.PortMapping
	manifest := appcschema.BlankPodManifest()

	for _, c := range pod.Spec.Containers {
		imgManifest, err := r.getImageManifest(c.Image)
		if err != nil {
			return nil, err
		}

		if imgManifest.App == nil {
			return nil, fmt.Errorf("no app section in image manifest for image: %q", c.Image)
		}

		img, err := r.getImageByName(c.Image)
		if err != nil {
			return nil, err
		}
		hash, err := appctypes.NewHash(img.id)
		if err != nil {
			return nil, err
		}

		opts, err := r.generator.GenerateRunContainerOptions(pod, &c)
		if err != nil {
			return nil, err
		}

		globalPortMappings = append(globalPortMappings, opts.PortMappings...)

		if err := setApp(imgManifest.App, &c, opts); err != nil {
			return nil, err
		}

		manifest.Apps = append(manifest.Apps, appcschema.RuntimeApp{
			// TODO(yifan): We should allow app name to be different with
			// image name. See https://github.com/coreos/rkt/pull/640.
			Name:  imgManifest.Name,
			Image: appcschema.RuntimeImage{ID: *hash},
			App:   imgManifest.App,
		})
	}

	volumeMap, ok := r.volumeGetter.GetVolumes(pod.UID)
	if !ok {
		return nil, fmt.Errorf("cannot get the volumes for pod %q", kubecontainer.GetPodFullName(pod))
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
	for _, port := range globalPortMappings {
		name, err := appctypes.SanitizeACName(port.Name)
		if err != nil {
			return nil, fmt.Errorf("cannot use the port's name %q as ACName: %v", port.Name, err)
		}
		portName := appctypes.MustACName(name)
		manifest.Ports = append(manifest.Ports, appctypes.ExposedPort{
			Name:     *portName,
			HostPort: uint(port.HostPort),
		})
	}
	// TODO(yifan): Set pod-level isolators once it's supported in kubernetes.
	return manifest, nil
}

func newUnitOption(section, name, value string) *unit.UnitOption {
	return &unit.UnitOption{Section: section, Name: name, Value: value}
}

// TODO(yifan): Remove the receiver once we can solve the appName->imageID problem.
func (r *runtime) apiPodToruntimePod(uuid string, pod *api.Pod) *kubecontainer.Pod {
	p := &kubecontainer.Pod{
		ID:        pod.UID,
		Name:      pod.Name,
		Namespace: pod.Namespace,
	}
	for i := range pod.Spec.Containers {
		c := &pod.Spec.Containers[i]
		img, err := r.getImageByName(c.Image)
		if err != nil {
			glog.Warningf("rkt: Cannot get image for %q: %v", c.Image, err)
		}
		p.Containers = append(p.Containers, &kubecontainer.Container{
			ID:      types.UID(buildContainerID(&containerID{uuid, c.Name, img.id})),
			Name:    c.Name,
			Image:   c.Image,
			Hash:    kubecontainer.HashContainer(c),
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
func (r *runtime) preparePod(pod *api.Pod) (string, bool, error) {
	cmds := []string{"prepare", "--quiet", "--pod-manifest"}

	// Generate the pod manifest from the pod spec.
	manifest, err := r.makePodManifest(pod)
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

	p := r.apiPodToruntimePod(uuid, pod)
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
func (r *runtime) RunPod(pod *api.Pod) error {
	glog.V(4).Infof("Rkt starts to run pod: name %q.", pod.Name)

	name, needReload, err := r.preparePod(pod)
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
func (r *runtime) makeRuntimePod(unitName string, podInfos map[string]*podInfo) (*kubecontainer.Pod, error) {
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
func (r *runtime) GetPods(all bool) ([]*kubecontainer.Pod, error) {
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
func (r *runtime) KillPod(pod kubecontainer.Pod) error {
	glog.V(4).Infof("Rkt is killing pod: name %q.", pod.Name)

	// TODO(yifan): More graceful stop. Replace with StopUnit and wait for a timeout.
	r.systemd.KillUnit(makePodServiceFileName(pod.ID), int32(syscall.SIGKILL))
	return r.systemd.Reload()
}

// GetPodStatus currently invokes GetPods() to return the status.
// TODO(yifan): Split the get status logic from GetPods().
func (r *runtime) GetPodStatus(pod *api.Pod) (*api.PodStatus, error) {
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

// Version invokes 'rkt version' to get the version information of the rkt
// runtime on the machine.
// The return values are an int array containers the version number.
//
// Example:
// rkt:0.3.2+git --> []int{0, 3, 2}.
//
func (r *runtime) Version() (kubecontainer.Version, error) {
	output, err := r.runCommand("version")
	if err != nil {
		return nil, err
	}

	// Example output for 'rkt version':
	// rkt version 0.3.2+git
	// appc version 0.3.0+git
	for _, line := range output {
		tuples := strings.Split(strings.TrimSpace(line), " ")
		if len(tuples) != 3 {
			glog.Warningf("rkt: cannot parse the output: %q.", line)
			continue
		}
		if tuples[0] == "rkt" {
			return parseVersion(tuples[2])
		}
	}
	return nil, fmt.Errorf("rkt: cannot determine the version")
}

// writeDockerAuthConfig writes the docker credentials to rkt auth config files.
// This enables rkt to pull docker images from docker registry with credentials.
func (r *runtime) writeDockerAuthConfig(image string, credsSlice []docker.AuthConfiguration) error {
	creds := docker.AuthConfiguration{}
	// TODO handle multiple creds
	if len(credsSlice) >= 1 {
		creds = credsSlice[0]
	}

	registry := "index.docker.io"
	// Image spec: [<registry>/]<repository>/<image>[:<version]
	explicitRegistry := (strings.Count(image, "/") == 2)
	if explicitRegistry {
		registry = strings.Split(image, "/")[0]
	}

	localConfigDir := rktLocalConfigDir
	if r.config.LocalConfigDir != "" {
		localConfigDir = r.config.LocalConfigDir
	}
	authDir := path.Join(localConfigDir, "auth.d")
	if _, err := os.Stat(authDir); os.IsNotExist(err) {
		if err := os.Mkdir(authDir, 0600); err != nil {
			glog.Errorf("rkt: Cannot create auth dir: %v", err)
			return err
		}
	}
	f, err := os.Create(path.Join(localConfigDir, authDir, registry+".json"))
	if err != nil {
		glog.Errorf("rkt: Cannot create docker auth config file: %v", err)
		return err
	}
	defer f.Close()
	config := fmt.Sprintf(dockerAuthTemplate, registry, creds.Username, creds.Password)
	if _, err := f.Write([]byte(config)); err != nil {
		glog.Errorf("rkt: Cannot write docker auth config file: %v", err)
		return err
	}
	return nil
}

// PullImage invokes 'rkt fetch' to download an aci.
// TODO(yifan): Now we only support docker images, this should be changed
// once the format of image is landed, see:
//
// https://github.com/GoogleCloudPlatform/kubernetes/issues/7203
//
func (r *runtime) PullImage(image kubecontainer.ImageSpec, pullSecrets []api.Secret) error {
	img := image.Image
	// TODO(yifan): The credential operation is a copy from dockertools package,
	// Need to resolve the code duplication.
	repoToPull, tag := parsers.ParseRepositoryTag(img)
	// If no tag was specified, use the default "latest".
	if len(tag) == 0 {
		tag = "latest"
	}

	keyring, err := credentialprovider.MakeDockerKeyring(pullSecrets, r.dockerKeyring)
	if err != nil {
		return err
	}

	creds, ok := keyring.Lookup(repoToPull)
	if !ok {
		glog.V(1).Infof("Pulling image %s without credentials", img)
	}

	// Let's update a json.
	// TODO(yifan): Find a way to feed this to rkt.
	if err := r.writeDockerAuthConfig(img, creds); err != nil {
		return err
	}

	output, err := r.runCommand("fetch", dockerPrefix+img)
	if err != nil {
		return fmt.Errorf("rkt: Failed to fetch image: %v:", output)
	}
	return nil
}

// IsImagePresent returns true if the image is available on the machine.
// TODO(yifan): 'rkt image' is now landed on master, use that once we bump up
// the rkt version.
func (r *runtime) IsImagePresent(image kubecontainer.ImageSpec) (bool, error) {
	img := image.Image
	if _, err := r.runCommand("prepare", "--local=true", dockerPrefix+img); err != nil {
		return false, nil
	}
	return true, nil
}

func (r *runtime) ListImages() ([]kubecontainer.Image, error) {
	return []kubecontainer.Image{}, fmt.Errorf("rkt: ListImages unimplemented")
}

func (r *runtime) RemoveImage(image kubecontainer.ImageSpec) error {
	return fmt.Errorf("rkt: RemoveImages unimplemented")
}

// SyncPod syncs the running pod to match the specified desired pod.
func (r *runtime) SyncPod(pod *api.Pod, runningPod kubecontainer.Pod, podStatus api.PodStatus, pullSecrets []api.Secret) error {
	podFullName := kubecontainer.GetPodFullName(pod)
	if len(runningPod.Containers) == 0 {
		glog.V(4).Infof("Pod %q is not running, will start it", podFullName)
		return r.RunPod(pod)
	}

	// Add references to all containers.
	unidentifiedContainers := make(map[types.UID]*kubecontainer.Container)
	for _, c := range runningPod.Containers {
		unidentifiedContainers[c.ID] = c
	}

	restartPod := false
	for _, container := range pod.Spec.Containers {
		expectedHash := kubecontainer.HashContainer(&container)

		c := runningPod.FindContainerByName(container.Name)
		if c == nil {
			if kubecontainer.ShouldContainerBeRestarted(&container, pod, &podStatus, r.readinessManager) {
				glog.V(3).Infof("Container %+v is dead, but RestartPolicy says that we should restart it.", container)
				// TODO(yifan): Containers in one pod are fate-sharing at this moment, see:
				// https://github.com/appc/spec/issues/276.
				restartPod = true
				break
			}
			continue
		}

		// TODO(yifan): Take care of host network change.
		containerChanged := c.Hash != 0 && c.Hash != expectedHash
		if containerChanged {
			glog.Infof("Pod %q container %q hash changed (%d vs %d), it will be killed and re-created.", podFullName, container.Name, c.Hash, expectedHash)
			restartPod = true
			break
		}

		result, err := r.prober.Probe(pod, podStatus, container, string(c.ID), c.Created)
		// TODO(vmarmol): examine this logic.
		if err == nil && result != probe.Success {
			glog.Infof("Pod %q container %q is unhealthy (probe result: %v), it will be killed and re-created.", podFullName, container.Name, result)
			restartPod = true
			break
		}

		if err != nil {
			glog.V(2).Infof("Probe container %q failed: %v", container.Name, err)
		}
		delete(unidentifiedContainers, c.ID)
	}

	// If there is any unidentified containers, restart the pod.
	if len(unidentifiedContainers) > 0 {
		restartPod = true
	}

	if restartPod {
		// TODO(yifan): Handle network plugin.
		if err := r.KillPod(runningPod); err != nil {
			return err
		}
		if err := r.RunPod(pod); err != nil {
			return err
		}
	}
	return nil
}

// GetContainerLogs uses journalctl to get the logs of the container.
// By default, it returns a snapshot of the container log. Set |follow| to true to
// stream the log. Set |follow| to false and specify the number of lines (e.g.
// "100" or "all") to tail the log.
// TODO(yifan): Currently, it fetches all the containers' log within a pod. We will
// be able to fetch individual container's log once https://github.com/coreos/rkt/pull/841
// landed.
func (r *runtime) GetContainerLogs(pod *api.Pod, containerID string, tail string, follow bool, stdout, stderr io.Writer) error {
	unitName := makePodServiceFileName(pod.UID)
	cmd := exec.Command("journalctl", "-u", unitName)
	if follow {
		cmd.Args = append(cmd.Args, "-f")
	}
	if tail == "all" {
		cmd.Args = append(cmd.Args, "-a")
	} else {
		_, err := strconv.Atoi(tail)
		if err == nil {
			cmd.Args = append(cmd.Args, "-n", tail)
		}
	}
	cmd.Stdout, cmd.Stderr = stdout, stderr
	return cmd.Start()
}

// GarbageCollect collects the pods/containers. TODO(yifan): Enforce the gc policy.
func (r *runtime) GarbageCollect() error {
	if err := exec.Command("systemctl", "reset-failed").Run(); err != nil {
		glog.Errorf("rkt: Failed to reset failed systemd services: %v", err)
	}
	if _, err := r.runCommand("gc", "--grace-period="+defaultGracePeriod, "--expire-prepared="+defaultExpirePrepared); err != nil {
		glog.Errorf("rkt: Failed to gc: %v", err)
		return err
	}
	return nil
}

// Note: In rkt, the container ID is in the form of "UUID:appName:ImageID", where
// appName is the container name.
func (r *runtime) RunInContainer(containerID string, cmd []string) ([]byte, error) {
	glog.V(4).Infof("Rkt running in container.")

	id, err := parseContainerID(containerID)
	if err != nil {
		return nil, err
	}
	// TODO(yifan): Use appName instead of imageID.
	// see https://github.com/coreos/rkt/pull/640
	args := append([]string{}, "enter", "--imageid", id.imageID, id.uuid)
	args = append(args, cmd...)

	result, err := r.runCommand(args...)
	return []byte(strings.Join(result, "\n")), err
}

// Note: In rkt, the container ID is in the form of "UUID:appName:ImageID", where
// appName is the container name.
func (r *runtime) ExecInContainer(containerID string, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool) error {
	glog.V(4).Infof("Rkt execing in container.")

	id, err := parseContainerID(containerID)
	if err != nil {
		return err
	}
	// TODO(yifan): Use appName instead of imageID.
	// see https://github.com/coreos/rkt/pull/640
	args := append([]string{}, "enter", "--imageid", id.imageID, id.uuid)
	args = append(args, cmd...)
	command := r.buildCommand(args...)

	if tty {
		p, err := kubecontainer.StartPty(command)
		if err != nil {
			return err
		}
		defer p.Close()

		// make sure to close the stdout stream
		defer stdout.Close()

		if stdin != nil {
			go io.Copy(p, stdin)
		}
		if stdout != nil {
			go io.Copy(stdout, p)
		}
		return command.Wait()
	}
	if stdin != nil {
		// Use an os.Pipe here as it returns true *os.File objects.
		// This way, if you run 'kubectl exec -p <pod> -i bash' (no tty) and type 'exit',
		// the call below to command.Run() can unblock because its Stdin is the read half
		// of the pipe.
		r, w, err := os.Pipe()
		if err != nil {
			return err
		}
		go io.Copy(w, stdin)

		command.Stdin = r
	}
	if stdout != nil {
		command.Stdout = stdout
	}
	if stderr != nil {
		command.Stderr = stderr
	}
	return command.Run()
}

// findRktID returns the rkt uuid for the pod.
// TODO(yifan): This is unefficient which require us to list
// all the unit files.
func (r *runtime) findRktID(pod *kubecontainer.Pod) (string, error) {
	units, err := r.systemd.ListUnits()
	if err != nil {
		return "", err
	}

	unitName := makePodServiceFileName(pod.ID)
	for _, u := range units {
		// u.Name contains file name ext such as .service, .socket, etc.
		if u.Name != unitName {
			continue
		}

		f, err := os.Open(path.Join(systemdServiceDir, u.Name))
		if err != nil {
			return "", err
		}
		defer f.Close()

		opts, err := unit.Deserialize(f)
		if err != nil {
			return "", err
		}

		for _, opt := range opts {
			if opt.Section == unitKubernetesSection && opt.Name == unitRktID {
				return opt.Value, nil
			}
		}
	}
	return "", fmt.Errorf("rkt uuid not found for pod %v", pod)
}

// PortForward executes socat in the pod's network namespace and copies
// data between stream (representing the user's local connection on their
// computer) and the specified port in the container.
//
// TODO:
//  - match cgroups of container
//  - should we support nsenter + socat on the host? (current impl)
//  - should we support nsenter + socat in a container, running with elevated privs and --pid=host?
//
// TODO(yifan): Merge with the same function in dockertools.
func (r *runtime) PortForward(pod *kubecontainer.Pod, port uint16, stream io.ReadWriteCloser) error {
	glog.V(4).Infof("Rkt port forwarding in container.")

	podInfos, err := r.getPodInfos()
	if err != nil {
		return err
	}

	rktID, err := r.findRktID(pod)
	if err != nil {
		return err
	}

	info, ok := podInfos[rktID]
	if !ok {
		return fmt.Errorf("cannot find the pod info for pod %v", pod)
	}
	if info.pid < 0 {
		return fmt.Errorf("cannot get the pid for pod %v", pod)
	}

	_, lookupErr := exec.LookPath("socat")
	if lookupErr != nil {
		return fmt.Errorf("unable to do port forwarding: socat not found.")
	}
	args := []string{"-t", fmt.Sprintf("%d", info.pid), "-n", "socat", "-", fmt.Sprintf("TCP4:localhost:%d", port)}

	_, lookupErr = exec.LookPath("nsenter")
	if lookupErr != nil {
		return fmt.Errorf("unable to do port forwarding: nsenter not found.")
	}
	command := exec.Command("nsenter", args...)
	command.Stdin = stream
	command.Stdout = stream
	return command.Run()
}

// isUUID returns true if the input is a valid rkt UUID,
// e.g. "2372bc17-47cb-43fb-8d78-20b31729feda".
func isUUID(input string) bool {
	if _, err := appctypes.NewUUID(input); err != nil {
		return false
	}
	return true
}

// getPodInfos returns a map of [pod-uuid]:*podInfo
func (r *runtime) getPodInfos() (map[string]*podInfo, error) {
	result := make(map[string]*podInfo)

	output, err := r.runCommand("list", "--no-legend", "--full")
	if err != nil {
		return result, err
	}
	if len(output) == 0 {
		// No pods are running.
		return result, nil
	}

	// Example output of current 'rkt list --full' (version == 0.4.2):
	// UUID                                 ACI     STATE      NETWORKS
	// 2372bc17-47cb-43fb-8d78-20b31729feda	foo     running    default:ip4=172.16.28.3
	//                                      bar
	// 40e2813b-9d5d-4146-a817-0de92646da96 foo     exited
	// 40e2813b-9d5d-4146-a817-0de92646da96 bar     exited
	//
	// With '--no-legend', the first line is eliminated.
	for _, line := range output {
		tuples := splitLineByTab(line)
		if len(tuples) < 1 {
			continue
		}
		if !isUUID(tuples[0]) {
			continue
		}
		id := tuples[0]
		status, err := r.runCommand("status", id)
		if err != nil {
			glog.Errorf("rkt: Cannot get status for pod (uuid=%q): %v", id, err)
			continue
		}
		info, err := parsePodInfo(status)
		if err != nil {
			glog.Errorf("rkt: Cannot parse status for pod (uuid=%q): %v", id, err)
			continue
		}
		result[id] = info
	}
	return result, nil
}

// listImages lists all the available appc images on the machine by invoking 'rkt images'.
func (r *runtime) listImages() ([]image, error) {
	output, err := r.runCommand("images", "--no-legend=true", "--fields=key,appname")
	if err != nil {
		return nil, err
	}
	if len(output) == 0 {
		return nil, nil
	}

	var images []image
	for _, line := range output {
		var img image
		if err := img.parseString(line); err != nil {
			glog.Warningf("rkt: Cannot parse image info from %q: %v", line, err)
			continue
		}
		images = append(images, img)
	}
	return images, nil
}

// getImageByName tries to find the image info with the given image name.
func (r *runtime) getImageByName(imageName string) (image, error) {
	// TODO(yifan): Print hash in rkt image?
	images, err := r.listImages()
	if err != nil {
		return image{}, err
	}

	var name, version string
	nameVersion := strings.Split(imageName, ":")

	// TODO(yifan): Currently the name cannot include "_", it is replaced
	// by "-". See the issue in appc/spec: https://github.com/appc/spec/issues/406.
	name, err = appctypes.SanitizeACName(nameVersion[0])
	if err != nil {
		return image{}, err
	}

	if len(nameVersion) == 2 {
		version = nameVersion[1]
	}

	for _, img := range images {
		if img.name == name {
			if version == "" || img.version == version {
				return img, nil
			}
		}
	}
	return image{}, fmt.Errorf("cannot find the image %q", imageName)
}
