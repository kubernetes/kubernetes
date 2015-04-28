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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	appcschema "github.com/appc/spec/schema"
	appctypes "github.com/appc/spec/schema/types"
	"github.com/coreos/go-systemd/dbus"
	"github.com/coreos/go-systemd/unit"
	"github.com/coreos/rkt/store"
	"github.com/docker/docker/pkg/parsers"
	"github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
	"github.com/kr/pty"
)

const (
	acversion            = "0.5.1"
	kubernetesUnitPrefix = "k8s"
	systemdServiceDir    = "/run/systemd/system"

	rktDataDir             = "/var/lib/rkt"
	rktLocalConfigDir      = "/etc/rkt"
	rktMetadataServiceFile = "rkt-metadata.service"
	rktMetadataSocketFile  = "rkt-metadata.socket"

	unitKubernetesSection = "X-Kubernetes"
	unitPodName           = "POD"
	unitRktID             = "RktID"

	dockerPrefix = "docker://"
)

const (
	rktBinName = "rkt"

	Embryo         = "embryo"
	Preparing      = "preparing"
	AbortedPrepare = "aborted prepare"
	Prepared       = "prepared"
	Running        = "running"
	Deleting       = "deleting" // This covers pod.isExitedDeleting and pod.isDeleting.
	Exited         = "exited"   // This covers pod.isExited and pod.isExitedGarbage.
	Garbage        = "garbage"
)

const (
	dockerAuthTemplate = `{"rktKind":"dockerAuth","rktVersion":"v1","registries":[%q],"credentials":{"user":%q,"password":%q}}`
)

var (
	systemd *dbus.Conn
	absPath string
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

// Config stores the global configuration for the rkt runtime.
// Run 'rkt' for more details.
type Config struct {
	// The debug flag for rkt.
	Debug bool
	// The rkt data directory.
	Dir string
	// This flag controls whether we skip image or key verification.
	InsecureSkipVerify bool
	// The local config directory.
	LocalConfigDir string
}

// buildGlobalOptions returns an array of global command line options.
func (c *Config) buildGlobalOptions() []string {
	var result []string
	if c == nil {
		return result
	}

	result = append(result, fmt.Sprintf("--debug=%v", c.Debug))
	result = append(result, fmt.Sprintf("--insecure-skip-verify=%v", c.InsecureSkipVerify))
	if c.LocalConfigDir != "" {
		result = append(result, fmt.Sprintf("--local-config=%s", c.LocalConfigDir))
	}
	if c.Dir != "" {
		result = append(result, fmt.Sprintf("--dir=%s", c.Dir))
	}
	return result
}

func startMetadataService(systemd *dbus.Conn) error {
	units, err := systemd.ListUnits()
	if err != nil {
		return err
	}
	for _, u := range units {
		if u.Name == rktMetadataServiceFile {
			// Metadata is already running.
			return nil
		}
	}

	// Create the service and socket file under systemd directory.
	var src, dst string
	dst = path.Join(systemdServiceDir, rktMetadataServiceFile)
	if _, err := os.Stat(dst); err == nil {
		if err := os.Remove(dst); err != nil {
			return err
		}
	}
	dst = path.Join(systemdServiceDir, rktMetadataSocketFile)
	if _, err := os.Stat(dst); err == nil {
		if err := os.Remove(dst); err != nil {
			return err
		}
	}

	wd, err := os.Getwd()
	if err != nil {
		return err
	}

	dst = path.Join(systemdServiceDir, rktMetadataServiceFile)
	src = path.Join(wd, rktMetadataServiceFile)
	if err := os.Symlink(src, dst); err != nil {
		return err
	}
	dst = path.Join(systemdServiceDir, rktMetadataSocketFile)
	src = path.Join(wd, rktMetadataSocketFile)
	if err := os.Symlink(src, dst); err != nil {
		return err
	}
	_, err = systemd.StartUnit(rktMetadataServiceFile, "replace")
	return err
}

// init will start the metadata-service as a transient systemd service.
func init() {
	var err error
	systemd, err = dbus.New()
	if err != nil {
		glog.Errorf("rkt: Cannot connect to dbus: %v", err)
	}

	// Test if rkt binary is in $PATH.
	absPath, err = exec.LookPath(rktBinName)
	if err != nil {
		glog.Errorf("rkt: Cannot find rkt binary: %v", err)
	}

	if err := startMetadataService(systemd); err != nil {
		glog.Errorf("rkt: Cannot start metadata service: %v", err)
	}
}

// New creates the rkt container runtime which implements the container runtime interface.
// It will test if the rkt binary is in the $PATH, and whether we can get the
// version of it. If so, creates the rkt container runtime, otherwise returns an error.
func New(config *Config) (*Runtime, error) {
	rkt := &Runtime{
		systemd:       systemd,
		absPath:       absPath,
		config:        config,
		dockerKeyring: credentialprovider.NewDockerKeyring(),
	}

	// Simply verify the binary by trying 'rkt version'.
	result, err := rkt.Version()
	if err != nil {
		return nil, err
	}
	if _, found := result[rktBinName]; !found {
		return nil, fmt.Errorf("rkt: cannot get the version of rkt")
	}
	glog.V(4).Infof("Rkt version: %v.", result)
	return rkt, nil
}

// Version invokes 'rkt version' to get the version information of the rkt
// runtime on the machine.
// The return values are a map of component:version.
//
// Example:
// rkt:0.3.2+git
// appc:0.3.0+git
//p
func (r *Runtime) Version() (map[string]string, error) {
	output, err := r.RunCommand("version")
	if err != nil {
		return nil, err
	}

	// Example output for 'rkt version':
	// rkt version 0.3.2+git
	// appc version 0.3.0+git
	result := make(map[string]string)
	for _, line := range output {
		tuples := strings.Split(strings.TrimSpace(line), " ")
		if len(tuples) != 3 {
			glog.Warningf("Cannot parse the output: %q.", line)
			continue
		}
		result[tuples[0]] = tuples[2]
	}
	return result, nil
}

// GetPods runs 'systemctl list-unit' and 'rkt list' to get the list of all the appcs.
// Then it will use the result to contruct list of pods.
func (r *Runtime) GetPods(all bool) ([]*kubecontainer.Pod, error) {
	glog.V(4).Infof("Rkt getting pods.")

	units, err := r.systemd.ListUnits()
	if err != nil {
		return nil, err
	}

	podInfos, err := r.getPodInfos()
	if err != nil {
		return nil, err
	}

	var pods []*kubecontainer.Pod
	for _, u := range units {
		if strings.HasPrefix(u.Name, kubernetesUnitPrefix) {
			pod, err := r.makePod(u.Name, podInfos)
			if err != nil {
				glog.Warningf("Cannot construct pod from unit file: %v.", err)
				continue
			}
			pods = append(pods, pod)
		}
	}
	return pods, nil
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

// KillPod invokes 'systemctl kill' to kill the unit that runs the pod.
func (r *Runtime) KillPod(pod *api.Pod) error {
	glog.V(4).Infof("Rkt is killing pod: name %q.", pod.Name)

	serviceName := makePodServiceFileName(pod.Name, pod.Namespace, pod.UID)

	// TODO(yifan): More graceful stop. Replace with StopUnit and wait for a timeout.
	r.systemd.KillUnit(serviceName, int32(syscall.SIGKILL))
	return r.systemd.Reload()
}

func (r *Runtime) RestartPod(pod *api.Pod, volumeMap map[string]volume.Volume) error {
	if err := r.KillPod(pod); err != nil {
		return err
	}
	return r.RunPod(pod, volumeMap)
}

// GetPodStatus currently invokes GetPods() to return the status. TODO(yifan): This is a hack,
// should try to split the status and the pod list.
func (r *Runtime) GetPodStatus(pod *api.Pod) (*api.PodStatus, error) {
	pods, err := r.GetPods(true)
	if err != nil {
		return nil, err
	}
	p := kubecontainer.Pods(pods).FindPodByID(pod.UID)
	// TODO(yifan): Refactor this, use nil.
	if len(p.Containers) == 0 {
		return nil, fmt.Errorf("cannot find status for pod: %q", kubecontainer.BuildPodFullName(pod.Name, pod.Namespace))
	}
	return &p.Status, nil
}

func (r *Runtime) buildCommand(args ...string) *exec.Cmd {
	cmd := exec.Command(rktBinName)
	cmd.Args = append(cmd.Args, r.config.buildGlobalOptions()...)
	cmd.Args = append(cmd.Args, args...)
	return cmd
}

// RunCommand invokes rkt binary with arguments and returns the result
// from stdout in a list of strings.
// TODO(yifan): Do not export this.
func (r *Runtime) RunCommand(args ...string) ([]string, error) {
	glog.V(4).Info("Run rkt command:", args)

	output, err := r.buildCommand(args...).Output()
	if err != nil {
		return nil, err
	}
	return strings.Split(strings.TrimSpace(string(output)), "\n"), nil
}

// splitLine breaks a line by tabs, and trims the leading and tailing spaces.
func splitLine(line string) []string {
	var result []string
	start := 0

	line = strings.TrimSpace(line)
	for i := 0; i < len(line); i++ {
		if line[i] == '\t' {
			result = append(result, line[start:i])
			for line[i] == '\t' {
				i++
			}
			start = i
		}
	}
	result = append(result, line[start:])
	return result
}

// getPodInfos returns a map of [pod-uuid]:*podInfo
func (r *Runtime) getPodInfos() (map[string]*podInfo, error) {
	output, err := r.RunCommand("list", "--no-legend", "--full")
	if err != nil {
		return nil, err
	}

	if len(output) == 0 {
		// No pods is running.
		return nil, nil
	}

	// Example output of current 'rkt list --full' (version == 0.4.2):
	// UUID                                 ACI     STATE      NETWORKS
	// 2372bc17-47cb-43fb-8d78-20b31729feda	foo     running    default:ip4=172.16.28.3
	//                                      bar
	// 40e2813b-9d5d-4146-a817-0de92646da96 foo     exited
	// 40e2813b-9d5d-4146-a817-0de92646da96 bar     exited
	//
	// With '--no-legend', the first line is eliminated.

	result := make(map[string]*podInfo)
	for _, line := range output {
		tuples := splitLine(line)
		if len(tuples) < 3 { // At least it should have 3 entries.
			continue
		}
		id := tuples[0]

		status, err := r.RunCommand("status", id)
		if err != nil {
			glog.Errorf("Cannot get status for pod (uuid=%q): %v", id, err)
			continue
		}
		info := newPodInfo()
		if err := info.parseStatus(status); err != nil {
			glog.Errorf("Cannot parse status for pod (uuid=%q): %v", id, err)
			continue
		}
		result[id] = info
	}
	return result, nil
}

// makePod constructs the pod by reading information from the given unit file
// and from the pod infos.
func (r *Runtime) makePod(unitName string, podInfos map[string]*podInfo) (*kubecontainer.Pod, error) {
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
			// NOTE: In fact we unmarshal from a serialized
			// api.BoundPod type here.
			err = json.Unmarshal([]byte(opt.Value), &pod)
			if err != nil {
				return nil, err
			}
		case unitRktID:
			rktID = opt.Value
		default:
			glog.Warningf("unexpected key: %q", opt.Name)
		}
	}

	if len(rktID) == 0 {
		return nil, fmt.Errorf("rkt: cannot find rkt ID of pod %v, unit file is broken", pod)
	}
	info, found := podInfos[rktID]
	if !found {
		glog.Warningf("Cannot find info for pod %q, rkt uuid: %q", pod.Name, rktID)
		return &pod, nil
	}
	pod.Status = info.toPodStatus(&pod)
	return &pod, nil
}

// makePodServiceFileName constructs the unit file name for the given pod.
func makePodServiceFileName(name, namespace string, uid types.UID) string {
	// TODO(yifan): Revisit this later, decide whether we want to use UID.
	return fmt.Sprintf("%s_%s_%s_%s.service", kubernetesUnitPrefix, name, namespace, uid)
}

func newUnitOption(section, name, value string) *unit.UnitOption {
	return &unit.UnitOption{Section: section, Name: name, Value: value}
}

type resource struct {
	limit   string
	request string
}

// setIsolators overrides the isolators of the pod manifest if necessary.
func setIsolators(app *appctypes.App, c *api.Container) error {
	var isolator appctypes.Isolator
	if len(c.Capabilities.Add) > 0 || len(c.Capabilities.Drop) > 0 || len(c.Resources.Limits) > 0 || len(c.Resources.Requests) > 0 {
		app.Isolators = []appctypes.Isolator{}
	}

	// Retained capabilities/privileged.
	privileged := false
	if capabilities.Get().AllowPrivileged {
		privileged = c.Privileged
	} else if c.Privileged {
		glog.Errorf("Privileged is disallowed globally")
		// TODO(yifan): Return error?
	}
	var caps string
	var value []byte
	if privileged {
		caps = getAllCapabilities()
	} else {
		caps = getCapabilities(c.Capabilities.Add)
	}
	if len(caps) > 0 {
		value = []byte(fmt.Sprintf(`{"name":"os/linux/capabilities-retain-set","value":{"set":[%s]}`, caps))
		if err := isolator.UnmarshalJSON(value); err != nil {
			glog.Errorf("Cannot unmarshal the retained capabilites %q: %v", value, err)
			return err
		}
		app.Isolators = append(app.Isolators, isolator)
	}

	// Removed capabilities.
	caps = getCapabilities(c.Capabilities.Drop)
	if len(caps) > 0 {
		value = []byte(fmt.Sprintf(`{"name":"os/linux/capabilities-remove-set","value":{"set":[%s]}`, caps))
		if err := isolator.UnmarshalJSON(value); err != nil {
			glog.Errorf("Cannot unmarshal the retained capabilites %q: %v", value, err)
			return err
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
	}
	for name, res := range resources {
		switch name {
		case api.ResourceCPU:
			name = "resource/cpu"
		case api.ResourceMemory:
			name = "resource/memory"
		default:
			glog.Warningf("Resource type not supported: %v", name)
		}
		value = []byte(fmt.Sprintf(`"name":%q,"value":{"request":%q,"limit":%q}`, name, res.request, res.limit))
		if err := isolator.UnmarshalJSON(value); err != nil {
			glog.Errorf("Cannot unmarshal the resource %q: %v", value, err)
			return err
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

	// TODO(yifan): Use non-root user in the future?
	// Currently it's a bug as reported https://github.com/coreos/rkt/issues/539.
	// However since we cannot get the user/group information from the container
	// spec, maybe we use the file path to set the user/group?
	app.User, app.Group = "0", "0"

	// Override the working directory.
	if len(c.WorkingDir) > 0 {
		app.WorkingDirectory = c.WorkingDir
	}

	// Override the environment.
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
			glog.Errorf("Cannot use the volume mount's name %q as ACName: %v", m.Name, err)
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
			glog.Errorf("Cannot use the port's name %q as ACName: %v", p.Name, err)
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
	ds, err := store.NewStore(rktDataDir)
	if err != nil {
		glog.Errorf("Cannot open store: %v", err)
		return nil, err
	}
	for _, c := range pod.Spec.Containers {
		// Assume we are running docker images for now, see #7203.
		imageID, err := r.getImageID(c.Image)
		if err != nil {
			return nil, fmt.Errorf("cannot get image ID for %q: %v", c.Image, err)
		}
		hash, err := appctypes.NewHash(imageID)
		if err != nil {
			glog.Errorf("Cannot create new hash from %q", imageID)
			return nil, err
		}

		im, err := ds.GetImageManifest(hash.String())
		if err != nil {
			glog.Errorf("Cannot get image manifest: %v", err)
			return nil, err
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
			glog.Errorf("Cannot use the volume's name %q as ACName: %v", name, err)
			return nil, err
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
				glog.Errorf("Cannot use the volume's name %q as ACName: %v", port.Name, err)
				return nil, err
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

func (r *Runtime) getImageID(imageName string) (string, error) {
	output, err := r.RunCommand("fetch", imageName)
	if err != nil {
		return "", err
	}
	last := output[len(output)-1]
	if !strings.HasPrefix(last, "sha512-") {
		return "", fmt.Errorf("unexpected result: %q", last)
	}
	return last, err
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
			ID:      buildContainerID(&containerID{uuid, c.Name, imageID}),
			Name:    c.Name,
			Image:   c.Image,
			Hash:    HashContainer(c),
			Created: time.Now().Unix(),
		})
	}
	return p
}

type containerID struct {
	uuid    string // uuid of the pod.
	appName string // name of the app in that pod.
	imageID string // id of the image. TODO(yifan): Depreciate this.
}

// buildContainerID constructs the containers's ID using containerID,
// which containers the pod uuid and the container name.
// The result can be used to globally identify a container.
func buildContainerID(c *containerID) types.UID {
	return types.UID(fmt.Sprintf("%s:%s:%s", c.uuid, c.appName, c.imageID))
}

// parseContainerID parses the containerID into pod uuid and the container name. The
// results can be used to get more information of the container.
func parseContainerID(id string) (*containerID, error) {
	tuples := strings.Split(id, ":")
	if len(tuples) != 3 {
		return nil, fmt.Errorf("rkt: cannot parse container ID for: %v", id)
	}
	return &containerID{
		uuid:    tuples[0],
		appName: tuples[1],
		imageID: tuples[2],
	}, nil
}

// preparePod creates the unit file and save it under systemdUnitDir.
// On success, it will return a string that represents name of the unit file
// and a boolean that indicates if the unit file needs reload.
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
			glog.Warningf("Cannot remove temp manifest file %q: %v", manifestFile.Name(), err)
		}
	}()

	data, err := json.Marshal(manifest)
	if err != nil {
		return "", false, err
	}
	// File.Write returns error if the written length is less than len(data).
	if _, err := manifestFile.Write(data); err != nil {
		return "", false, err
	}

	cmds = append(cmds, manifestFile.Name())
	output, err := r.RunCommand(cmds...)
	if err != nil {
		return "", false, err
	}
	if len(output) != 1 {
		return "", false, fmt.Errorf("rkt: cannot get uuid from 'rkt prepare'")
	}
	uuid := output[0]
	glog.V(4).Infof("'rkt prepare' returns %q.", uuid)

	p := r.apiPodToRuntimePod(uuid, pod)
	b, err := json.Marshal(p)
	if err != nil {
		glog.Errorf("rkt: cannot marshal pod `%s_%s`: %v", p.Name, p.Namespace, err)
		return "", false, err
	}

	runPrepared := fmt.Sprintf("%s run-prepared --private-net=%v %s", r.absPath, pod.Spec.HostNetwork, uuid)
	units := []*unit.UnitOption{
		newUnitOption(unitKubernetesSection, unitRktID, uuid),
		newUnitOption(unitKubernetesSection, unitPodName, string(b)),
		newUnitOption("Service", "ExecStart", runPrepared),
	}

	// Save the unit file under systemd's service directory.
	// TODO(yifan) Garbage collect 'dead' serivce files.
	needReload := false
	unitName := makePodServiceFileName(pod.Name, pod.Namespace, pod.UID)
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

func (r *Runtime) writeDockerAuthConfig(image string, creds docker.AuthConfiguration) error {
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
			glog.Errorf("Cannot create auth dir: %v", err)
			return err
		}
	}
	f, err := os.Create(path.Join(localConfigDir, "auth.d", registry+".json"))
	if err != nil {
		glog.Errorf("Cannot create docker auth config file: %v", err)
		return err
	}
	defer f.Close()
	config := fmt.Sprintf(dockerAuthTemplate, registry, creds.Username, creds.Password)
	if _, err := f.Write([]byte(config)); err != nil {
		glog.Errorf("Cannot write docker auth config file: %v", err)
		return err
	}
	return nil
}

// PullImage invokes 'rkt fetch' to download an aci.
func (r *Runtime) PullImage(img string) error {
	if strings.HasPrefix(img, dockerPrefix) {
		repoToPull, tag := parsers.ParseRepositoryTag(img)
		// If no tag was specified, use the default "latest".
		if len(tag) == 0 {
			tag = "latest"
		}

		creds, ok := r.dockerKeyring.Lookup(repoToPull)
		if !ok {
			glog.V(1).Infof("Pulling image %s without credentials", img)
		}

		// Let's update a json.
		// TODO(yifan): Find a way to feed this to rkt.
		if err := r.writeDockerAuthConfig(img, creds); err != nil {
			return err
		}
	}

	output, err := r.RunCommand("fetch", img)
	if err != nil {
		return fmt.Errorf("failed to fetch image: %v:", output)
	}
	return nil
}

// IsImagePresent returns true if the image is available on the machine.
// TODO(yifan): This is hack, which uses 'rkt prepare --local' to test whether
// the image is present.
func (r *Runtime) IsImagePresent(img string) (bool, error) {
	if _, err := r.RunCommand("prepare", "--local=true", img); err != nil {
		return false, nil
	}
	return true, nil
}

// TODO(yifan): Move this duplicated function to container runtime.
// HashContainer computes the hash of one api.Container.
func HashContainer(container *api.Container) uint64 {
	hash := adler32.New()
	util.DeepHashObject(hash, *container)
	return uint64(hash.Sum32())
}

// Note: In rkt, the container ID is in the form of "UUID:appName:ImageID", where
// appName is the container name.
func (r *Runtime) RunInContainer(containerID string, cmd []string) ([]byte, error) {
	glog.V(4).Infof("Rkt running in container.")

	id, err := parseContainerID(containerID)
	if err != nil {
		return nil, err
	}
	// TODO(yifan): Use appName instead of imageID.
	// see https://github.com/coreos/rkt/pull/640
	args := append([]string{}, "enter", "--imageid", id.imageID, id.uuid)
	args = append(args, cmd...)

	result, err := r.RunCommand(args...)
	return []byte(strings.Join(result, "\n")), err
}

// Pty unsupported yet.
func startPty(c *exec.Cmd) (*os.File, error) {
	return pty.Start(c)
}

// Note: In rkt, the container ID is in the form of "UUID:appName:ImageID", where
// appName is the container name.
func (r *Runtime) ExecInContainer(containerID string, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool) error {
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
		p, err := startPty(command)
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

func (r *Runtime) findRktID(pod kubecontainer.Pod) (string, error) {
	units, err := r.systemd.ListUnits()
	if err != nil {
		return "", err
	}

	for _, u := range units {
		if strings.HasPrefix(u.Name, makePodServiceFileName(pod.Name, pod.Namespace, pod.ID)) {
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
	}
	return "", fmt.Errorf("rkt uuid not found for pod %v", pod)
}

func (r *Runtime) PortForward(pod kubecontainer.Pod, port uint16, stream io.ReadWriteCloser) error {
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

	// TODO what if the host doesn't have it???
	_, lookupErr := exec.LookPath("socat")
	if lookupErr != nil {
		return fmt.Errorf("unable to do port forwarding: socat not found.")
	}
	args := []string{"-t", fmt.Sprintf("%d", info.pid), "-n", "socat", "-", fmt.Sprintf("TCP4:localhost:%d", port)}
	// TODO use exec.LookPath
	command := exec.Command("nsenter", args...)
	command.Stdin = stream
	command.Stdout = stream
	return command.Run()
}
