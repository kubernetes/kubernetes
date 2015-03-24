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

package rocket

import (
	"encoding/json"
	"fmt"
	"hash/adler32"
	"io"
	"os"
	"os/exec"
	"path"
	"strings"
	"syscall"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	"github.com/coreos/go-systemd/dbus"
	"github.com/coreos/go-systemd/unit"
	"github.com/golang/glog"
)

const (
	kubernetesUnitPrefix = "k8s"
	systemdServiceDir    = "/run/systemd/system"

	unitKubernetesSection = "X-Kubernetes"
	unitPodName           = "POD"
	unitRocketID          = "RocketID"
)

const (
	rocketBinName = "rkt"

	// TODO(yifan): Figure out a way to sync up upstream's spec.
	podStateEmbryo    = "embryo"
	podStatePreparing = "preparing"
	podStatePrepared  = "prepared"
	podStateRunning   = "running"
	podStateExited    = "exited"
	podStateDeleting  = "deleting"
	podStateGone      = "gone"
)

// Runtime implements the ContainerRuntime for rocket. The implementation
// uses systemd, so in order to run this runtime, systemd must be installed
// on the machine.
type Runtime struct {
	systemd *dbus.Conn
	absPath string
	config  *Config
}

// Config stores the global configuration for the rocket runtime.
// Run 'rkt' for more details.
type Config struct {
	// The debug flag for rocket.
	Debug bool
	// The rocket data directory
	Dir string
	// This flag controls whether we skip image or key verification.
	InsecureSkipVerify bool
}

// buildGlobalOptions returns an array of global command line options.
func (c *Config) buildGlobalOptions() []string {
	var result []string
	if c == nil {
		return result
	}

	result = append(result, fmt.Sprintf("--debug=%v", c.Debug))
	result = append(result, fmt.Sprintf("--insecure-skip-verify=%v", c.InsecureSkipVerify))
	if c.Dir != "" {
		result = append(result, fmt.Sprintf("--dir=%s", c.Dir))
	}
	return result
}

// New creates the rocket container runtime which implements the container runtime interface.
// It will test if the rocket binary is in the $PATH, and whether we can get the
// version of it. If so, creates the rocket container runtime, otherwise returns an error.
func New(config *Config) (*Runtime, error) {
	systemd, err := dbus.New()
	if err != nil {
		return nil, err
	}
	// Test if rocket binary is in $PATH.
	absPath, err := exec.LookPath(rocketBinName)
	if err != nil {
		return nil, err
	}
	rkt := &Runtime{
		systemd: systemd,
		absPath: absPath,
		config:  config,
	}

	// Simply verify the binary by trying 'rkt version'.
	result, err := rkt.Version()
	if err != nil {
		return nil, err
	}
	if _, found := result[rocketBinName]; !found {
		return nil, fmt.Errorf("rocket: cannot get the version of rocket")
	}
	glog.V(4).Infof("Rocket version: %v.", result)
	return rkt, nil
}

// Version invokes 'rkt version' to get the version information of the rocket
// runtime on the machine.
// The return values are a map of component:version.
//
// Example:
// rkt:0.3.2+git
// appc:0.3.0+git
//
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
func (r *Runtime) GetPods() ([]*kubecontainer.Pod, error) {
	glog.V(4).Infof("Rocket getting pods.")

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
	glog.V(4).Infof("Rocket starts to run pod: name %q.", pod.Name)

	name, needReload, err := r.preparePod(pod, volumeMap)
	if err != nil {
		return err
	}
	if needReload {
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
	glog.V(4).Infof("Rocket is killing pod: name %q.", pod.Name)

	serviceName := makePodServiceFileName(pod.Name, pod.Namespace)

	// TODO(yifan): More graceful stop. Replace with StopUnit and wait for a timeout.
	r.systemd.KillUnit(serviceName, int32(syscall.SIGKILL))
	return nil
}

// RunContainerInPod launches a container in the given pod.
// For now, we need to kill and restart the whole pod. Hopefully we will be
// launching this single container without touching its siblings in the near future.
func (r *Runtime) RunContainerInPod(container api.Container, pod *api.Pod, volumeMap map[string]volume.Volume) error {
	if err := r.KillPod(pod); err != nil {
		return err
	}

	// Update the pod and start it.
	pod.Spec.Containers = append(pod.Spec.Containers, container)
	if err := r.RunPod(pod, volumeMap); err != nil {
		return err
	}
	return nil
}

// KillContainer kills the container in the given pod.
// Like RunContainerInPod, we will have to tear down the whole pod first to kill this
// single container.
func (r *Runtime) KillContainerInPod(container api.Container, pod *api.Pod) error {
	if err := r.KillPod(pod); err != nil {
		return err
	}

	// Update the pod and start it.
	var containers []api.Container
	for _, c := range pod.Spec.Containers {
		if c.Name == container.Name {
			continue
		}
		containers = append(containers, c)
	}
	pod.Spec.Containers = containers
	// TODO(yifan): Bug here, since we cannot get the volume map, after killing the mount
	// path will disappear. This could be fixed if we support killing single container without
	// tearing down the whole pod.
	return r.RunPod(pod, nil)
}

// RunCommand invokes rocket binary with arguments and returns the result
// from stdout in a list of strings.
// TODO(yifan): Do not export this.
func (r *Runtime) RunCommand(args ...string) ([]string, error) {
	glog.V(4).Info("Run rkt command:", args)

	cmd := exec.Command(rocketBinName)
	cmd.Args = append(cmd.Args, r.config.buildGlobalOptions()...)
	cmd.Args = append(cmd.Args, args...)

	output, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	return strings.Split(strings.TrimSpace(string(output)), "\n"), nil
}

type podInfo struct {
	state       string
	networkInfo string
}

// getIP returns the IP of a pod by parsing the network info.
// The network info looks like this:
//
// default:ip4=172.16.28.3, database:ip4=172.16.28.42
//
func (p *podInfo) getIP() string {
	parts := strings.Split(p.networkInfo, ",")

	for _, part := range parts {
		if strings.HasPrefix(part, "default:") {
			return strings.Split(part, "=")[1]
		}
	}
	return ""
}

// getContainerStatus converts the rocket pod state to the api.containerStatus.
// TODO(yifan): Get more detailed info such as Image, ImageID, etc.
func (p *podInfo) getContainerStatus(container *kubecontainer.Container) api.ContainerStatus {
	var status api.ContainerStatus
	status.Name = container.Name
	status.Image = container.Image
	switch p.state {
	case podStateRunning:
		// TODO(yifan): Get StartedAt.
		status.State = api.ContainerState{
			Running: &api.ContainerStateRunning{
				StartedAt: util.Unix(container.Created, 0),
			},
		}
	case podStateEmbryo, podStatePreparing, podStatePrepared:
		status.State = api.ContainerState{Waiting: &api.ContainerStateWaiting{}}
	case podStateExited, podStateDeleting, podStateGone:
		status.State = api.ContainerState{
			Termination: &api.ContainerStateTerminated{
				StartedAt: util.Unix(container.Created, 0),
			},
		}
	default:
		glog.Warningf("Unknown pod state: %q", p.state)
	}
	return status
}

func (p *podInfo) toPodStatus(pod *kubecontainer.Pod) api.PodStatus {
	var status api.PodStatus
	status.PodIP = p.getIP()
	// For now just make every container's state as same as the pod.
	for _, container := range pod.Containers {
		status.ContainerStatuses = append(status.ContainerStatuses, p.getContainerStatus(container))
	}
	return status
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
		info := &podInfo{
			state: tuples[2],
		}
		if len(tuples) == 4 {
			info.networkInfo = tuples[3]
		}
		result[tuples[0]] = info
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

	var rocketID string
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
		case unitRocketID:
			rocketID = opt.Value
		default:
			glog.Warningf("unexpected key: %q", opt.Name)
		}
	}

	if len(rocketID) == 0 {
		return nil, fmt.Errorf("rocket: cannot find rocket ID of pod %v, unit file is broken", pod)
	}
	info, found := podInfos[rocketID]
	if !found {
		glog.Warningf("Cannot find info for pod %q, rocket uuid: %q", pod.Name, rocketID)
		return &pod, nil
	}
	pod.Status = info.toPodStatus(&pod)
	return &pod, nil
}

// makePodServiceFileName constructs the unit file name for the given pod.
// TODO(yifan), when BoundPod and Pod are combined, we can change the input to
// take just a pod.
func makePodServiceFileName(podName, podNamespace string) string {
	// TODO(yifan): Revisit this later, decide whether we want to use UID.
	return fmt.Sprintf("%s_%s_%s.service", kubernetesUnitPrefix, podName, podNamespace)
}

func newUnitOption(section, name, value string) *unit.UnitOption {
	return &unit.UnitOption{Section: section, Name: name, Value: value}
}

func apiPodToRuntimePod(uuid string, pod *api.Pod) *kubecontainer.Pod {
	p := &kubecontainer.Pod{
		ID:        pod.UID,
		Name:      pod.Name,
		Namespace: pod.Namespace,
	}
	for i := range pod.Spec.Containers {
		c := &pod.Spec.Containers[i]
		p.Containers = append(p.Containers, &kubecontainer.Container{
			ID:      buildContainerID(&containerID{uuid, c.Name}),
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
}

// buildContainerID constructs the containers's ID using containerID,
// which containers the pod uuid and the container name.
// The result can be used to globally identify a container.
func buildContainerID(c *containerID) types.UID {
	return types.UID(fmt.Sprintf("%s:%s", c.uuid, c.appName))
}

// parseContainerID parses the containerID into pod uuid and the container name. The
// results can be used to get more information of the container.
func parseContainerID(id types.UID) (*containerID, error) {
	tuples := strings.Split(string(id), ":")
	if len(tuples) != 2 {
		return nil, fmt.Errorf("rocket: cannot parse container ID for: %v", id)
	}
	return &containerID{
		uuid:    tuples[0],
		appName: tuples[1],
	}, nil
}

// preparePod creates the unit file and save it under systemdUnitDir.
// On success, it will return a string that represents name of the unit file
// and a boolean that indicates if the unit file needs reload.
func (r *Runtime) preparePod(pod *api.Pod, volumeMap map[string]volume.Volume) (string, bool, error) {
	cmds := []string{"prepare", "--quiet"}

	// Construct the '--volume' cmd line.
	for name, mount := range volumeMap {
		cmds = append(cmds, "--volume")
		cmds = append(cmds, fmt.Sprintf("%s,kind=host,source=%s", name, mount.GetPath()))
	}

	// Append ACIs.
	for _, c := range pod.Spec.Containers {
		cmds = append(cmds, c.Image)
	}

	output, err := r.RunCommand(cmds...)
	if err != nil {
		return "", false, err
	}
	if len(output) != 1 {
		return "", false, fmt.Errorf("rocket: cannot get uuid from 'rkt prepare'")
	}
	uuid := output[0]
	glog.V(4).Infof("'rkt prepare' returns %q.", uuid)

	p := apiPodToRuntimePod(uuid, pod)
	b, err := json.Marshal(p)
	if err != nil {
		glog.Errorf("rocket: cannot marshal pod `%s_%s`: %v", p.Name, p.Namespace, err)
		return "", false, err
	}

	runPrepared := fmt.Sprintf("%s run-prepared --private-net --spawn-metadata-svc %s", r.absPath, uuid)
	units := []*unit.UnitOption{
		newUnitOption(unitKubernetesSection, unitRocketID, uuid),
		newUnitOption(unitKubernetesSection, unitPodName, string(b)),
		newUnitOption("Service", "ExecStart", runPrepared),
	}

	// Save the unit file under systemd's service directory.
	// TODO(yifan) Garbage collect 'dead' serivce files.
	needReload := false
	unitName := makePodServiceFileName(pod.Name, pod.Namespace)
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

// Note: In rocket, the container ID is in the form of "UUID_ImageID".
func (r *Runtime) RunInContainer(containerID string, cmd []string) ([]byte, error) {
	id, err := parseContainerID(types.UID(containerID))
	if err != nil {
		return nil, err
	}
	// TODO(yifan): Currently, store image ID in appName.
	// This will change in the future, see https://github.com/coreos/rocket/pull/640
	args := append([]string{}, "enter", "--imageid", id.appName, id.uuid)
	args = append(args, cmd...)
	result, err := r.RunCommand(args...)
	return []byte(strings.Join(result, "\n")), err
}

// TODO(yifan): Move this duplicated function to container runtime.
// HashContainer computes the hash of one api.Container.
func HashContainer(container *api.Container) uint64 {
	hash := adler32.New()
	util.DeepHashObject(hash, *container)
	return uint64(hash.Sum32())
}
