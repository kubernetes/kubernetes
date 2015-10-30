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
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"strconv"
	"strings"
	"time"

	appcschema "github.com/appc/spec/schema"
	appctypes "github.com/appc/spec/schema/types"
	"github.com/coreos/go-systemd/dbus"
	"github.com/coreos/go-systemd/unit"
	"github.com/docker/docker/pkg/parsers"
	docker "github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/credentialprovider"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/prober"
	kubeletUtil "k8s.io/kubernetes/pkg/kubelet/util"
	"k8s.io/kubernetes/pkg/probe"
	"k8s.io/kubernetes/pkg/securitycontext"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
)

const (
	acVersion             = "0.6.1"
	rktMinimumVersion     = "0.8.1"
	systemdMinimumVersion = "219"

	RktType = "rkt"

	systemdServiceDir = "/run/systemd/system"
	rktDataDir        = "/var/lib/rkt"
	rktLocalConfigDir = "/etc/rkt"

	kubernetesUnitPrefix  = "k8s"
	unitKubernetesSection = "X-Kubernetes"
	unitPodName           = "POD"
	unitRktID             = "RktID"
	unitRestartCount      = "RestartCount"

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

	defaultImageTag = "latest"
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
	imagePuller         kubecontainer.ImagePuller
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
	volumeGetter volumeGetter,
	serializeImagePulls bool,
) (kubecontainer.Runtime, error) {

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

	rktBinAbsPath := config.Path
	if rktBinAbsPath == "" {
		// No default rkt path was set, so try to find one in $PATH.
		var err error
		rktBinAbsPath, err = exec.LookPath("rkt")
		if err != nil {
			return nil, fmt.Errorf("cannot find rkt binary: %v", err)
		}
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
	if serializeImagePulls {
		rkt.imagePuller = kubecontainer.NewSerializedImagePuller(recorder, rkt)
	} else {
		rkt.imagePuller = kubecontainer.NewImagePuller(recorder, rkt)
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

func (r *runtime) buildCommand(args ...string) *exec.Cmd {
	cmd := exec.Command(r.rktBinAbsPath)
	cmd.Args = append(cmd.Args, r.config.buildGlobalOptions()...)
	cmd.Args = append(cmd.Args, args...)
	return cmd
}

// runCommand invokes rkt binary with arguments and returns the result
// from stdout in a list of strings. Each string in the list is a line.
func (r *runtime) runCommand(args ...string) ([]string, error) {
	glog.V(4).Info("rkt: Run command:", args)

	var stdout, stderr bytes.Buffer
	cmd := r.buildCommand(args...)
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("failed to run %v: %v\nstdout: %v\nstderr: %v", args, err, stdout.String(), stderr.String())
	}
	return strings.Split(strings.TrimSpace(stdout.String()), "\n"), nil
}

// makePodServiceFileName constructs the unit file name for a pod using its UID.
func makePodServiceFileName(uid types.UID) string {
	// TODO(yifan): Add name for readability? We need to consider the
	// limit of the length.
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
	var acName appctypes.ACIdentifier
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

// findEnvInList returns the index of environment variable in the environment whose Name equals env.Name.
func findEnvInList(envs appctypes.Environment, env kubecontainer.EnvVar) int {
	for i, e := range envs {
		if e.Name == env.Name {
			return i
		}
	}
	return -1
}

// setApp overrides the app's fields if any of them are specified in the
// container's spec.
func setApp(app *appctypes.App, c *api.Container, opts *kubecontainer.RunContainerOptions) error {
	// Override the exec.

	if len(c.Command) > 0 {
		app.Exec = c.Command
	}
	if len(c.Args) > 0 {
		app.Exec = append(app.Exec, c.Args...)
	}

	// TODO(yifan): Use non-root user in the future, see:
	// https://github.com/coreos/rkt/issues/820
	app.User, app.Group = "0", "0"

	// Override the working directory.
	if len(c.WorkingDir) > 0 {
		app.WorkingDirectory = c.WorkingDir
	}

	// Merge the environment. Override the image with the ones defined in the spec if necessary.
	for _, env := range opts.Envs {
		if ix := findEnvInList(app.Environment, env); ix >= 0 {
			app.Environment[ix].Value = env.Value
			continue
		}
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

// parseImageName parses a docker image string into two parts: repo and tag.
// If tag is empty, return the defaultImageTag.
func parseImageName(image string) (string, string) {
	repoToPull, tag := parsers.ParseRepositoryTag(image)
	// If no tag was specified, use the default "latest".
	if len(tag) == 0 {
		tag = defaultImageTag
	}
	return repoToPull, tag
}

// getImageManifest invokes 'rkt image cat-manifest' to retrive the image manifest
// for the image.
func (r *runtime) getImageManifest(image string) (*appcschema.ImageManifest, error) {
	var manifest appcschema.ImageManifest

	repoToPull, tag := parseImageName(image)
	imgName, err := appctypes.SanitizeACIdentifier(repoToPull)
	if err != nil {
		return nil, err
	}
	output, err := r.runCommand("image", "cat-manifest", fmt.Sprintf("%s:%s", imgName, tag))
	if err != nil {
		return nil, err
	}
	if len(output) != 1 {
		return nil, fmt.Errorf("invalid output: %v", output)
	}
	return &manifest, json.Unmarshal([]byte(output[0]), &manifest)
}

// makePodManifest transforms a kubelet pod spec to the rkt pod manifest.
func (r *runtime) makePodManifest(pod *api.Pod, pullSecrets []api.Secret) (*appcschema.PodManifest, error) {
	var globalPortMappings []kubecontainer.PortMapping
	manifest := appcschema.BlankPodManifest()

	for _, c := range pod.Spec.Containers {
		if err := r.imagePuller.PullImage(pod, &c, pullSecrets); err != nil {
			return nil, err
		}
		imgManifest, err := r.getImageManifest(c.Image)
		if err != nil {
			return nil, err
		}

		if imgManifest.App == nil {
			imgManifest.App = new(appctypes.App)
		}

		img, err := r.getImageByName(c.Image)
		if err != nil {
			return nil, err
		}
		hash, err := appctypes.NewHash(img.ID)
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

		name, err := appctypes.SanitizeACName(c.Name)
		if err != nil {
			return nil, err
		}
		appName := appctypes.MustACName(name)

		manifest.Apps = append(manifest.Apps, appcschema.RuntimeApp{
			Name:  *appName,
			Image: appcschema.RuntimeImage{ID: *hash},
			App:   imgManifest.App,
		})
	}

	volumeMap, ok := r.volumeGetter.GetVolumes(pod.UID)
	if !ok {
		return nil, fmt.Errorf("cannot get the volumes for pod %q", kubeletUtil.FormatPodName(pod))
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

// apiPodToruntimePod converts an api.Pod to kubelet/container.Pod.
// we save the this for later reconstruction of the kubelet/container.Pod
// such as in GetPods().
func apiPodToRuntimePod(uuid string, pod *api.Pod) *kubecontainer.Pod {
	p := &kubecontainer.Pod{
		ID:        pod.UID,
		Name:      pod.Name,
		Namespace: pod.Namespace,
	}
	for i := range pod.Spec.Containers {
		c := &pod.Spec.Containers[i]
		p.Containers = append(p.Containers, &kubecontainer.Container{
			ID:      types.UID(buildContainerID(&containerID{uuid, c.Name})),
			Name:    c.Name,
			Image:   c.Image,
			Hash:    kubecontainer.HashContainer(c),
			Created: time.Now().Unix(),
		})
	}
	return p
}

// serviceFilePath returns the absolute path of the service file.
func serviceFilePath(serviceName string) string {
	return path.Join(systemdServiceDir, serviceName)
}

// preparePod will:
//
// 1. Invoke 'rkt prepare' to prepare the pod, and get the rkt pod uuid.
// 2. Create the unit file and save it under systemdUnitDir.
//
// On success, it will return a string that represents name of the unit file
// and the runtime pod.
func (r *runtime) preparePod(pod *api.Pod, pullSecrets []api.Secret) (string, *kubecontainer.Pod, error) {
	// Generate the pod manifest from the pod spec.
	manifest, err := r.makePodManifest(pod, pullSecrets)
	if err != nil {
		return "", nil, err
	}
	manifestFile, err := ioutil.TempFile("", fmt.Sprintf("manifest-%s-", pod.Name))
	if err != nil {
		return "", nil, err
	}
	defer func() {
		manifestFile.Close()
		if err := os.Remove(manifestFile.Name()); err != nil {
			glog.Warningf("rkt: Cannot remove temp manifest file %q: %v", manifestFile.Name(), err)
		}
	}()

	data, err := json.Marshal(manifest)
	if err != nil {
		return "", nil, err
	}
	// Since File.Write returns error if the written length is less than len(data),
	// so check error is enough for us.
	if _, err := manifestFile.Write(data); err != nil {
		return "", nil, err
	}

	// Run 'rkt prepare' to get the rkt UUID.
	cmds := []string{"prepare", "--quiet", "--pod-manifest", manifestFile.Name()}
	if r.config.Stage1Image != "" {
		cmds = append(cmds, "--stage1-image", r.config.Stage1Image)
	}
	output, err := r.runCommand(cmds...)
	if err != nil {
		return "", nil, err
	}
	if len(output) != 1 {
		return "", nil, fmt.Errorf("invalid output from 'rkt prepare': %v", output)
	}
	uuid := output[0]
	glog.V(4).Infof("'rkt prepare' returns %q", uuid)

	// Create systemd service file for the rkt pod.
	runtimePod := apiPodToRuntimePod(uuid, pod)
	b, err := json.Marshal(runtimePod)
	if err != nil {
		return "", nil, err
	}

	var runPrepared string
	if pod.Spec.HostNetwork {
		runPrepared = fmt.Sprintf("%s run-prepared --mds-register=false %s", r.rktBinAbsPath, uuid)
	} else {
		runPrepared = fmt.Sprintf("%s run-prepared --mds-register=false --private-net %s", r.rktBinAbsPath, uuid)
	}

	// TODO handle pod.Spec.HostPID
	// TODO handle pod.Spec.HostIPC

	units := []*unit.UnitOption{
		newUnitOption(unitKubernetesSection, unitRktID, uuid),
		newUnitOption(unitKubernetesSection, unitPodName, string(b)),
		// This makes the service show up for 'systemctl list-units' even if it exits successfully.
		newUnitOption("Service", "RemainAfterExit", "true"),
		newUnitOption("Service", "ExecStart", runPrepared),
		// This enables graceful stop.
		newUnitOption("Service", "KillMode", "mixed"),
	}

	// Check if there's old rkt pod corresponding to the same pod, if so, update the restart count.
	var restartCount int
	var needReload bool
	serviceName := makePodServiceFileName(pod.UID)
	if _, err := os.Stat(serviceFilePath(serviceName)); err == nil {
		// Service file already exists, that means the pod is being restarted.
		needReload = true
		_, info, err := r.readServiceFile(serviceName)
		if err != nil {
			glog.Warningf("rkt: Cannot get old pod's info from service file %q: (%v), will ignore it", serviceName, err)
			restartCount = 0
		} else {
			restartCount = info.restartCount + 1
		}
	}
	units = append(units, newUnitOption(unitKubernetesSection, unitRestartCount, strconv.Itoa(restartCount)))

	glog.V(4).Infof("rkt: Creating service file %q for pod %q", serviceName, kubeletUtil.FormatPodName(pod))
	serviceFile, err := os.Create(serviceFilePath(serviceName))
	if err != nil {
		return "", nil, err
	}
	defer serviceFile.Close()

	_, err = io.Copy(serviceFile, unit.Serialize(units))
	if err != nil {
		return "", nil, err
	}

	if needReload {
		if err := r.systemd.Reload(); err != nil {
			return "", nil, err
		}
	}
	return serviceName, runtimePod, nil
}

// generateEvents is a helper function that generates some container
// life cycle events for containers in a pod.
func (r *runtime) generateEvents(runtimePod *kubecontainer.Pod, reason string, failure error) {
	// Set up container references.
	for _, c := range runtimePod.Containers {
		containerID := string(c.ID)
		id, err := parseContainerID(containerID)
		if err != nil {
			glog.Warningf("Invalid container ID %q", containerID)
			continue
		}

		ref, ok := r.containerRefManager.GetRef(containerID)
		if !ok {
			glog.Warningf("No ref for container %q", containerID)
			continue
		}

		// Note that 'rkt id' is the pod id.
		uuid := util.ShortenString(id.uuid, 8)
		switch reason {
		case "Created":
			r.recorder.Eventf(ref, "Created", "Created with rkt id %v", uuid)
		case "Started":
			r.recorder.Eventf(ref, "Started", "Started with rkt id %v", uuid)
		case "Failed":
			r.recorder.Eventf(ref, "Failed", "Failed to start with rkt id %v with error %v", uuid, failure)
		case "Killing":
			r.recorder.Eventf(ref, "Killing", "Killing with rkt id %v", uuid)
		default:
			glog.Errorf("rkt: Unexpected event %q", reason)
		}
	}
	return
}

// RunPod first creates the unit file for a pod, and then
// starts the unit over d-bus.
func (r *runtime) RunPod(pod *api.Pod, pullSecrets []api.Secret) error {
	glog.V(4).Infof("Rkt starts to run pod: name %q.", kubeletUtil.FormatPodName(pod))

	name, runtimePod, prepareErr := r.preparePod(pod, pullSecrets)

	// Set container references and generate events.
	// If preparedPod fails, then send out 'failed' events for each container.
	// Otherwise, store the container references so we can use them later to send events.
	for i, c := range pod.Spec.Containers {
		ref, err := kubecontainer.GenerateContainerRef(pod, &c)
		if err != nil {
			glog.Errorf("Couldn't make a ref to pod %q, container %v: '%v'", kubeletUtil.FormatPodName(pod), c.Name, err)
			continue
		}
		if prepareErr != nil {
			r.recorder.Eventf(ref, "Failed", "Failed to create rkt container with error: %v", prepareErr)
			continue
		}
		containerID := string(runtimePod.Containers[i].ID)
		r.containerRefManager.SetRef(containerID, ref)
	}

	if prepareErr != nil {
		return prepareErr
	}

	r.generateEvents(runtimePod, "Created", nil)

	// TODO(yifan): This is the old version of go-systemd. Should update when libcontainer updates
	// its version of go-systemd.
	// RestartUnit has the same effect as StartUnit if the unit is not running, besides it can restart
	// a unit if the unit file is changed and reloaded.
	if _, err := r.systemd.RestartUnit(name, "replace"); err != nil {
		r.generateEvents(runtimePod, "Failed", err)
		return err
	}

	r.generateEvents(runtimePod, "Started", nil)

	return nil
}

// readServiceFile reads the service file and constructs the runtime pod and the rkt info.
func (r *runtime) readServiceFile(serviceName string) (*kubecontainer.Pod, *rktInfo, error) {
	f, err := os.Open(serviceFilePath(serviceName))
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	var pod kubecontainer.Pod
	opts, err := unit.Deserialize(f)
	if err != nil {
		return nil, nil, err
	}

	info := emptyRktInfo()
	for _, opt := range opts {
		if opt.Section != unitKubernetesSection {
			continue
		}
		switch opt.Name {
		case unitPodName:
			err = json.Unmarshal([]byte(opt.Value), &pod)
			if err != nil {
				return nil, nil, err
			}
		case unitRktID:
			info.uuid = opt.Value
		case unitRestartCount:
			cnt, err := strconv.Atoi(opt.Value)
			if err != nil {
				return nil, nil, err
			}
			info.restartCount = cnt
		default:
			return nil, nil, fmt.Errorf("rkt: unexpected key: %q", opt.Name)
		}
	}

	if info.isEmpty() {
		return nil, nil, fmt.Errorf("rkt: cannot find rkt info of pod %v, unit file is broken", pod)
	}
	return &pod, info, nil
}

// GetPods runs 'systemctl list-unit' and 'rkt list' to get the list of rkt pods.
// Then it will use the result to construct a list of container runtime pods.
// If all is false, then only running pods will be returned, otherwise all pods will be
// returned.
func (r *runtime) GetPods(all bool) ([]*kubecontainer.Pod, error) {
	glog.V(4).Infof("Rkt getting pods")

	units, err := r.systemd.ListUnits()
	if err != nil {
		return nil, err
	}

	var pods []*kubecontainer.Pod
	for _, u := range units {
		if strings.HasPrefix(u.Name, kubernetesUnitPrefix) {
			if !all && u.SubState != "running" {
				continue
			}
			pod, _, err := r.readServiceFile(u.Name)
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
// TODO(yifan): Handle network plugin.
func (r *runtime) KillPod(pod *api.Pod, runningPod kubecontainer.Pod) error {
	glog.V(4).Infof("Rkt is killing pod: name %q.", runningPod.Name)

	serviceName := makePodServiceFileName(runningPod.ID)
	r.generateEvents(&runningPod, "Killing", nil)
	for _, c := range runningPod.Containers {
		id := string(c.ID)
		r.containerRefManager.ClearRef(id)
	}

	// Since all service file have 'KillMode=mixed', the processes in
	// the unit's cgroup will receive a SIGKILL if the normal stop timeouts.
	if _, err := r.systemd.StopUnit(serviceName, "replace"); err != nil {
		return err
	}
	// Remove the systemd service file as well.
	return os.Remove(serviceFilePath(serviceName))
}

// getPodStatus reads the service file and invokes 'rkt status $UUID' to get the
// pod's status.
func (r *runtime) getPodStatus(serviceName string) (*api.PodStatus, error) {
	// TODO(yifan): Get rkt uuid from the service file name.
	pod, rktInfo, err := r.readServiceFile(serviceName)
	if err != nil {
		return nil, err
	}
	podInfo, err := r.getPodInfo(rktInfo.uuid)
	if err != nil {
		return nil, err
	}
	status := makePodStatus(pod, podInfo, rktInfo)
	return &status, nil
}

// GetPodStatus returns the status of the given pod.
func (r *runtime) GetPodStatus(pod *api.Pod) (*api.PodStatus, error) {
	serviceName := makePodServiceFileName(pod.UID)
	return r.getPodStatus(serviceName)
}

func (r *runtime) Type() string {
	return RktType
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

// TODO(yifan): This is very racy, unefficient, and unsafe, we need to provide
// different namespaces. See: https://github.com/coreos/rkt/issues/836.
func (r *runtime) writeDockerAuthConfig(image string, credsSlice []docker.AuthConfiguration) error {
	if len(credsSlice) == 0 {
		return nil
	}

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

	config := fmt.Sprintf(dockerAuthTemplate, registry, creds.Username, creds.Password)
	if err := ioutil.WriteFile(path.Join(authDir, registry+".json"), []byte(config), 0600); err != nil {
		glog.Errorf("rkt: Cannot write docker auth config file: %v", err)
		return err
	}
	return nil
}

// PullImage invokes 'rkt fetch' to download an aci.
// TODO(yifan): Now we only support docker images, this should be changed
// once the format of image is landed, see:
//
// http://issue.k8s.io/7203
//
func (r *runtime) PullImage(image kubecontainer.ImageSpec, pullSecrets []api.Secret) error {
	img := image.Image
	// TODO(yifan): The credential operation is a copy from dockertools package,
	// Need to resolve the code duplication.
	repoToPull, _ := parseImageName(img)
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

	if _, err := r.runCommand("fetch", dockerPrefix+img); err != nil {
		glog.Errorf("Failed to fetch: %v", err)
		return err
	}
	return nil
}

// TODO(yifan): Searching the image via 'rkt images' might not be the most efficient way.
func (r *runtime) IsImagePresent(image kubecontainer.ImageSpec) (bool, error) {
	repoToPull, tag := parseImageName(image.Image)
	// Example output of 'rkt image list --fields=name':
	//
	// NAME
	// nginx:latest
	// coreos.com/rkt/stage1:0.8.1
	//
	// With '--no-legend=true' the fist line (NAME) will be omitted.
	output, err := r.runCommand("image", "list", "--no-legend=true", "--fields=name")
	if err != nil {
		return false, err
	}
	for _, line := range output {
		parts := strings.Split(strings.TrimSpace(line), ":")

		var imgName, imgTag string
		switch len(parts) {
		case 1:
			imgName, imgTag = parts[0], defaultImageTag
		case 2:
			imgName, imgTag = parts[0], parts[1]
		default:
			continue
		}

		if imgName == repoToPull && imgTag == tag {
			return true, nil
		}
	}
	return false, nil
}

// SyncPod syncs the running pod to match the specified desired pod.
func (r *runtime) SyncPod(pod *api.Pod, runningPod kubecontainer.Pod, podStatus api.PodStatus, pullSecrets []api.Secret, backOff *util.Backoff) error {
	podFullName := kubeletUtil.FormatPodName(pod)
	if len(runningPod.Containers) == 0 {
		glog.V(4).Infof("Pod %q is not running, will start it", podFullName)
		return r.RunPod(pod, pullSecrets)
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

		// TODO: check for non-root image directives.  See ../docker/manager.go#SyncPod

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
		if err := r.KillPod(pod, runningPod); err != nil {
			return err
		}
		if err := r.RunPod(pod, pullSecrets); err != nil {
			return err
		}
	}
	return nil
}

// GetContainerLogs uses journalctl to get the logs of the container.
// By default, it returns a snapshot of the container log. Set |follow| to true to
// stream the log. Set |follow| to false and specify the number of lines (e.g.
// "100" or "all") to tail the log.
//
// In rkt runtime's implementation, per container log is get via 'journalctl -M [rkt-$UUID] -u [APP_NAME]'.
// See https://github.com/coreos/rkt/blob/master/Documentation/commands.md#logging for more details.
//
// TODO(yifan): If the rkt is using lkvm as the stage1 image, then this function will fail.
func (r *runtime) GetContainerLogs(pod *api.Pod, containerID string, logOptions *api.PodLogOptions, stdout, stderr io.Writer) error {
	id, err := parseContainerID(containerID)
	if err != nil {
		return err
	}

	cmd := exec.Command("journalctl", "-M", fmt.Sprintf("rkt-%s", id.uuid), "-u", id.appName)
	if logOptions.Follow {
		cmd.Args = append(cmd.Args, "-f")
	}
	if logOptions.TailLines == nil {
		cmd.Args = append(cmd.Args, "-a")
	} else {
		cmd.Args = append(cmd.Args, "-n", strconv.FormatInt(*logOptions.TailLines, 10))
	}
	cmd.Stdout, cmd.Stderr = stdout, stderr
	return cmd.Run()
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

// Note: In rkt, the container ID is in the form of "UUID:appName", where
// appName is the container name.
// TODO(yifan): If the rkt is using lkvm as the stage1 image, then this function will fail.
func (r *runtime) RunInContainer(containerID string, cmd []string) ([]byte, error) {
	glog.V(4).Infof("Rkt running in container.")

	id, err := parseContainerID(containerID)
	if err != nil {
		return nil, err
	}
	args := append([]string{}, "enter", fmt.Sprintf("--app=%s", id.appName), id.uuid)
	args = append(args, cmd...)

	result, err := r.runCommand(args...)
	return []byte(strings.Join(result, "\n")), err
}

func (r *runtime) AttachContainer(containerID string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool) error {
	return fmt.Errorf("unimplemented")
}

// Note: In rkt, the container ID is in the form of "UUID:appName", where UUID is
// the rkt UUID, and appName is the container name.
// TODO(yifan): If the rkt is using lkvm as the stage1 image, then this function will fail.
func (r *runtime) ExecInContainer(containerID string, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool) error {
	glog.V(4).Infof("Rkt execing in container.")

	id, err := parseContainerID(containerID)
	if err != nil {
		return err
	}
	args := append([]string{}, "enter", fmt.Sprintf("--app=%s", id.appName), id.uuid)
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
		// This way, if you run 'kubectl exec <pod> -i bash' (no tty) and type 'exit',
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
func (r *runtime) findRktID(pod *kubecontainer.Pod) (string, error) {
	serviceName := makePodServiceFileName(pod.ID)

	f, err := os.Open(serviceFilePath(serviceName))
	if err != nil {
		if os.IsNotExist(err) {
			return "", fmt.Errorf("no service file %v for runtime pod %q, ID %q", serviceName, pod.Name, pod.ID)
		}
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
// TODO(yifan): If the rkt is using lkvm as the stage1 image, then this function will fail.
func (r *runtime) PortForward(pod *kubecontainer.Pod, port uint16, stream io.ReadWriteCloser) error {
	glog.V(4).Infof("Rkt port forwarding in container.")

	rktID, err := r.findRktID(pod)
	if err != nil {
		return err
	}

	info, err := r.getPodInfo(rktID)
	if err != nil {
		return err
	}

	socatPath, lookupErr := exec.LookPath("socat")
	if lookupErr != nil {
		return fmt.Errorf("unable to do port forwarding: socat not found.")
	}

	args := []string{"-t", fmt.Sprintf("%d", info.pid), "-n", socatPath, "-", fmt.Sprintf("TCP4:localhost:%d", port)}

	nsenterPath, lookupErr := exec.LookPath("nsenter")
	if lookupErr != nil {
		return fmt.Errorf("unable to do port forwarding: nsenter not found.")
	}

	command := exec.Command(nsenterPath, args...)
	command.Stdout = stream

	// If we use Stdin, command.Run() won't return until the goroutine that's copying
	// from stream finishes. Unfortunately, if you have a client like telnet connected
	// via port forwarding, as long as the user's telnet client is connected to the user's
	// local listener that port forwarding sets up, the telnet session never exits. This
	// means that even if socat has finished running, command.Run() won't ever return
	// (because the client still has the connection and stream open).
	//
	// The work around is to use StdinPipe(), as Wait() (called by Run()) closes the pipe
	// when the command (socat) exits.
	inPipe, err := command.StdinPipe()
	if err != nil {
		return fmt.Errorf("unable to do port forwarding: error creating stdin pipe: %v", err)
	}
	go func() {
		io.Copy(inPipe, stream)
		inPipe.Close()
	}()

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

// getPodInfo returns the pod info of a single pod according
// to the uuid.
func (r *runtime) getPodInfo(uuid string) (*podInfo, error) {
	status, err := r.runCommand("status", uuid)
	if err != nil {
		return nil, err
	}
	info, err := parsePodInfo(status)
	if err != nil {
		return nil, err
	}
	return info, nil
}

// getImageByName tries to find the image info with the given image name.
// TODO(yifan): Replace with 'rkt image cat-manifest'.
// imageName should be in the form of 'example.com/app:latest', which should matches
// the result of 'rkt image list'. If the version is empty, then 'latest' is assumed.
func (r *runtime) getImageByName(imageName string) (*kubecontainer.Image, error) {
	// TODO(yifan): Print hash in 'rkt image cat-manifest'?
	images, err := r.ListImages()
	if err != nil {
		return nil, err
	}

	nameVersion := strings.Split(imageName, ":")
	switch len(nameVersion) {
	case 1:
		imageName += ":" + defaultImageTag
	case 2:
		break
	default:
		return nil, fmt.Errorf("invalid image name: %q, requires 'name[:version]'")
	}

	for _, img := range images {
		for _, t := range img.Tags {
			if t == imageName {
				return &img, nil
			}
		}
	}
	return nil, fmt.Errorf("cannot find the image %q", imageName)
}

// ListImages lists all the available appc images on the machine by invoking 'rkt image list'.
func (r *runtime) ListImages() ([]kubecontainer.Image, error) {
	// Example output of 'rkt image list --fields=key,name':
	//
	// KEY									        NAME
	// sha512-374770396f23dd153937cd66694fe705cf375bcec7da00cf87e1d9f72c192da7	nginx:latest
	// sha512-bead9e0df8b1b4904d0c57ade2230e6d236e8473f62614a8bc6dcf11fc924123	coreos.com/rkt/stage1:0.8.1
	//
	// With '--no-legend=true' the fist line (KEY NAME) will be omitted.
	output, err := r.runCommand("image", "list", "--no-legend=true", "--fields=key,name")
	if err != nil {
		return nil, err
	}
	if len(output) == 0 {
		return nil, nil
	}

	var images []kubecontainer.Image
	for _, line := range output {
		img, err := parseImageInfo(line)
		if err != nil {
			glog.Warningf("rkt: Cannot parse image info from %q: %v", line, err)
			continue
		}
		images = append(images, *img)
	}
	return images, nil
}

// parseImageInfo creates the kubecontainer.Image struct by parsing the string in the result of 'rkt image list',
// the input looks like:
//
// sha512-91e98d7f1679a097c878203c9659f2a26ae394656b3147963324c61fa3832f15	coreos.com/etcd:v2.0.9
//
func parseImageInfo(input string) (*kubecontainer.Image, error) {
	idName := strings.Split(strings.TrimSpace(input), "\t")
	if len(idName) != 2 {
		return nil, fmt.Errorf("invalid image information from 'rkt image list': %q", input)
	}
	return &kubecontainer.Image{
		ID:   idName[0],
		Tags: []string{idName[1]},
	}, nil
}

// RemoveImage removes an on-disk image using 'rkt image rm'.
// TODO(yifan): Use image ID to reference image.
func (r *runtime) RemoveImage(image kubecontainer.ImageSpec) error {
	img, err := r.getImageByName(image.Image)
	if err != nil {
		return err
	}

	if _, err := r.runCommand("image", "rm", img.ID); err != nil {
		return err
	}
	return nil
}
