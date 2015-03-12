/*
Copyright 2015 CoreOS Inc. All rights reserved.

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
	"io"
	"os"
	"os/exec"
	"path"
	"strings"
	"syscall"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume"
	"github.com/coreos/go-systemd/dbus"
	"github.com/coreos/go-systemd/unit"
	"github.com/golang/glog"
)

const (
	kubernetesUnitPrefix = "K8S"
	systemdServiceDir    = "/run/systemd/system"

	unitKubernetesSection = "X-Kubernetes"
	rocketIDKey           = "RocketID"
	unitPodName           = "POD"
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

// Runtime implements the ContainerRuntime for rocket.
type Runtime struct {
	systemd *dbus.Conn
	absPath string
	config  *Config
}

// Config stores the configuration for the rocket runtime.
type Config struct {
	Debug              bool
	Dir                string
	InsecureSkipVerify bool
}

// parseGlobalOptions returns an array of global command line options.
func (c *Config) parseGlobalOptions() []string {
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
// It will test if the 'rkt' binary is in the $PATH, and whether we can get the
// version of it. If so, creates the rocket container runtime, otherwise returns an error.
func New(config *Config) (*Runtime, error) {
	systemd, err := dbus.New()
	if err != nil {
		return nil, err
	}
	// Test if 'rkt' is in $PATH.
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

// ListPods runs 'systemctl list-unit' and 'rkt list' to get the list of all the appcs.
// Then it will use the result to contruct list of pods.
func (r *Runtime) ListPods() ([]*api.Pod, error) {
	glog.V(4).Infof("Rocket is listing pods.")

	units, err := r.systemd.ListUnits()
	if err != nil {
		return nil, err
	}

	var pods []*api.Pod
	for _, u := range units {
		if strings.HasPrefix(u.Name, kubernetesUnitPrefix) {
			pod, err := r.makePod(u.Name)
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
func (r *Runtime) RunPod(pod *api.BoundPod, volumeMap map[string]volume.Interface) error {
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

	ch := make(chan string)
	_, err = r.systemd.StartUnit(name, "replace", ch)
	if err != nil {
		return err
	}
	if status := <-ch; status != "done" {
		return fmt.Errorf("rocket: unexpected return status %q", status)
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
func (r *Runtime) RunContainerInPod(container api.Container, pod *api.Pod, volumeMap map[string]volume.Interface) error {
	if err := r.KillPod(pod); err != nil {
		return err
	}

	// Update the pod and start it.
	pod.Spec.Containers = append(pod.Spec.Containers, container)
	boundPod := &api.BoundPod{pod.TypeMeta, pod.ObjectMeta, pod.Spec}
	if err := r.RunPod(boundPod, volumeMap); err != nil {
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
	boundPod := &api.BoundPod{pod.TypeMeta, pod.ObjectMeta, pod.Spec}
	// TODO(yifan): Bug here, since we cannot get the volume map, after killing the mount
	// path will disappear. This could be fixed if we support killing single container without
	// tearing down the whole pod.
	return r.RunPod(boundPod, nil)
}

// RunCommand invokes 'rkt' and returns the result from stdout
// in a list of strings.
func (r *Runtime) RunCommand(args ...string) ([]string, error) {
	glog.V(4).Info("Run rkt command:", args)

	cmd := exec.Command(rocketBinName)
	cmd.Args = append(cmd.Args, r.config.parseGlobalOptions()...)
	cmd.Args = append(cmd.Args, args...)

	output, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	return strings.Split(strings.TrimSpace(string(output)), "\n"), nil
}

// getPodsState returns a map of [pod-uuid]:[pod-state].
func (r *Runtime) getPodsState() (map[string]string, error) {
	result := make(map[string]string)
	output, err := r.RunCommand("list", "--no-legend")
	if err != nil {
		return nil, err
	}

	if len(output) == 0 {
		// No pods is running.
		return nil, nil
	}

	// Example output of current 'rkt list' (version <= 0.4.0):
	// UUID                                 ACI     STATE
	// 2372bc17-47cb-43fb-8d78-20b31729feda	foo     running
	//                                      bar
	// 40e2813b-9d5d-4146-a817-0de92646da96 foo     exited
	// 40e2813b-9d5d-4146-a817-0de92646da96 bar     exited
	//
	// With '--no-legend', the first line is eliminated.
	for _, line := range output {
		tuples := strings.Split(strings.TrimSpace(line), "\t")
		if len(tuples) != 3 {
			continue
		}
		result[tuples[0]] = tuples[2]
	}
	return result, nil
}

// getPodStatus fills the status of the given pod. Especially, it will
// fill the status for each container in the pod.
func (r *Runtime) getPodStatus(pod *api.Pod) error {
	// TODO(yifan) Cache this.
	podStates, err := r.getPodsState()
	if err != nil {
		return err
	}

	rktID, found := pod.Annotations[rocketIDKey]
	if !found {
		// TODO(yifan): Maybe we should panic here...
		return fmt.Errorf("rocket: cannot find rocket pod: %v, this is impossible", pod)
	}

	state, found := podStates[rktID]
	if !found {
		return fmt.Errorf("rocket: cannot find the state for pod: %q, rocket ID: %q", pod.Name, rktID)
	}

	// For now just make every container's state as same as the pod.
	pod.Status.Info = make(map[string]api.ContainerStatus)
	for _, container := range pod.Spec.Containers {
		// TODO(yifan): Pull out creationg, starting time.
		switch state {
		case podStateRunning:
			pod.Status.Info[container.Name] = api.ContainerStatus{
				State: api.ContainerState{
					Running: &api.ContainerStateRunning{},
				},
			}
		case podStateEmbryo, podStatePreparing, podStatePrepared:
			pod.Status.Info[container.Name] = api.ContainerStatus{
				State: api.ContainerState{
					Waiting: &api.ContainerStateWaiting{},
				},
			}
		case podStateExited, podStateDeleting, podStateGone:
			pod.Status.Info[container.Name] = api.ContainerStatus{
				State: api.ContainerState{
					Termination: &api.ContainerStateTerminated{},
				},
			}
		default:
			return fmt.Errorf("rocket: unexpected state: %q", state)
		}
	}
	return nil
}

// makePod constructs the pod by reading information from the given unit file
// and from rocket APIs.
func (r *Runtime) makePod(unitName string) (*api.Pod, error) {
	f, err := os.Open(path.Join(systemdServiceDir, unitName))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var pod api.Pod
	opts, err := unit.Deserialize(f)
	if err != nil {
		return nil, err
	}

	for _, opt := range opts {
		if opt.Section != unitKubernetesSection {
			continue
		}
		if opt.Name == unitPodName {
			// NOTE: In fact we unmarshal from a serialized
			// api.BoundPod type here.
			err = json.Unmarshal([]byte(opt.Value), &pod)
			if err != nil {
				return nil, err
			}
		}
	}

	if err = r.getPodStatus(&pod); err != nil {
		glog.Errorf("Error getting pod status of pod %q.", pod.Name)
		return nil, err
	}
	return &pod, nil
}

// makePodServiceFileName constructs the unit file name for the given pod.
// TODO(yifan), when BoundPod and Pod are combined, we can change the input to
// take just a pod.
func makePodServiceFileName(podName, podNamespace string) string {
	// TODO(yifan): Revisit this later, decide whether we want to use UID.
	return fmt.Sprintf("%s_%s_%s.service", kubernetesUnitPrefix, podName, podNamespace)
}

// preparePod creates the unit file and save it under systemdUnitDir.
// On success, it will return a string that represents name of the unit file
// and a boolean that indicates if the unit file needs reload.
func (r *Runtime) preparePod(pod *api.BoundPod, volumeMap map[string]volume.Interface) (string, bool, error) {
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

	// Save the rocket uuid in pod annotations.
	pod.Annotations[rocketIDKey] = uuid
	b, err := json.Marshal(pod)
	if err != nil {
		return "", false, err
	}

	units := []*unit.UnitOption{
		{
			Section: unitKubernetesSection,
			Name:    unitPodName,
			Value:   string(b),
		},
		{
			Section: "Service",
			Name:    "ExecStart",
			Value:   fmt.Sprintf("%s run-prepared %s", r.absPath, uuid),
		},
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
