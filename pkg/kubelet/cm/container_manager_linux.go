// +build linux

/*
Copyright 2015 The Kubernetes Authors.

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

package cm

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"strconv"
	"sync"
	"time"

	"github.com/golang/glog"
	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fs"
	"github.com/opencontainers/runc/libcontainer/configs"
	"k8s.io/apimachinery/pkg/api/resource"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	cmutil "k8s.io/kubernetes/pkg/kubelet/cm/util"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/qos"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/oom"
	"k8s.io/kubernetes/pkg/util/procfs"
	utilsysctl "k8s.io/kubernetes/pkg/util/sysctl"
	utilversion "k8s.io/kubernetes/pkg/util/version"
)

const (
	// The percent of the machine memory capacity. The value is used to calculate
	// docker memory resource container's hardlimit to workaround docker memory
	// leakage issue. Please see kubernetes/issues/9881 for more detail.
	DockerMemoryLimitThresholdPercent = 70
	// The minimum memory limit allocated to docker container: 150Mi
	MinDockerMemoryLimit = 150 * 1024 * 1024

	dockerProcessName     = "docker"
	dockerPidFile         = "/var/run/docker.pid"
	containerdProcessName = "docker-containerd"
	containerdPidFile     = "/run/docker/libcontainerd/docker-containerd.pid"
)

var (
	// The docker version in which containerd was introduced.
	containerdVersion = utilversion.MustParseSemantic("1.11.0")
)

// A non-user container tracked by the Kubelet.
type systemContainer struct {
	// Absolute name of the container.
	name string

	// CPU limit in millicores.
	cpuMillicores int64

	// Function that ensures the state of the container.
	// m is the cgroup manager for the specified container.
	ensureStateFunc func(m *fs.Manager) error

	// Manager for the cgroups of the external container.
	manager *fs.Manager
}

func newSystemCgroups(containerName string) *systemContainer {
	return &systemContainer{
		name:    containerName,
		manager: createManager(containerName),
	}
}

type containerManagerImpl struct {
	sync.RWMutex
	cadvisorInterface cadvisor.Interface
	mountUtil         mount.Interface
	NodeConfig
	status Status
	// External containers being managed.
	systemContainers []*systemContainer
	qosContainers    QOSContainersInfo
	periodicTasks    []func()
	// holds all the mounted cgroup subsystems
	subsystems  *CgroupSubsystems
	nodeInfo    *v1.Node
	getPodsFunc func() []*v1.Pod
}

type features struct {
	cpuHardcapping bool
}

var _ ContainerManager = &containerManagerImpl{}

// checks if the required cgroups subsystems are mounted.
// As of now, only 'cpu' and 'memory' are required.
// cpu quota is a soft requirement.
func validateSystemRequirements(mountUtil mount.Interface) (features, error) {
	const (
		cgroupMountType = "cgroup"
		localErr        = "system validation failed"
	)
	var (
		cpuMountPoint string
		f             features
	)
	mountPoints, err := mountUtil.List()
	if err != nil {
		return f, fmt.Errorf("%s - %v", localErr, err)
	}

	expectedCgroups := sets.NewString("cpu", "cpuacct", "cpuset", "memory")
	for _, mountPoint := range mountPoints {
		if mountPoint.Type == cgroupMountType {
			for _, opt := range mountPoint.Opts {
				if expectedCgroups.Has(opt) {
					expectedCgroups.Delete(opt)
				}
				if opt == "cpu" {
					cpuMountPoint = mountPoint.Path
				}
			}
		}
	}

	if expectedCgroups.Len() > 0 {
		return f, fmt.Errorf("%s - Following Cgroup subsystem not mounted: %v", localErr, expectedCgroups.List())
	}

	// Check if cpu quota is available.
	// CPU cgroup is required and so it expected to be mounted at this point.
	periodExists, err := util.FileExists(path.Join(cpuMountPoint, "cpu.cfs_period_us"))
	if err != nil {
		glog.Errorf("failed to detect if CPU cgroup cpu.cfs_period_us is available - %v", err)
	}
	quotaExists, err := util.FileExists(path.Join(cpuMountPoint, "cpu.cfs_quota_us"))
	if err != nil {
		glog.Errorf("failed to detect if CPU cgroup cpu.cfs_quota_us is available - %v", err)
	}
	if quotaExists && periodExists {
		f.cpuHardcapping = true
	}
	return f, nil
}

// TODO(vmarmol): Add limits to the system containers.
// Takes the absolute name of the specified containers.
// Empty container name disables use of the specified container.
func NewContainerManager(mountUtil mount.Interface, cadvisorInterface cadvisor.Interface, nodeConfig NodeConfig, failSwapOn bool) (ContainerManager, lifecycle.PodAdmitHandler, error) {
	subsystems, err := GetCgroupSubsystems()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get mounted cgroup subsystems: %v", err)
	}

	// Check whether swap is enabled. The Kubelet does not support running with swap enabled.
	cmd := exec.Command("cat", "/proc/swaps")
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, nil, err
	}
	var buf []string
	scanner := bufio.NewScanner(stdout)
	for scanner.Scan() { // Splits on newlines by default
		buf = append(buf, scanner.Text())
	}
	if err := cmd.Wait(); err != nil { // Clean up
		return nil, nil, err
	}

	// TODO(#34726:1.8.0): Remove the opt-in for failing when swap is enabled.
	//     Running with swap enabled should be considered an error, but in order to maintain legacy
	//     behavior we have to require an opt-in to this error for a period of time.

	// If there is more than one line (table headers) in /proc/swaps, swap is enabled and we should error out.
	if len(buf) > 1 {
		if failSwapOn {
			return nil, nil, fmt.Errorf("Running with swap on is not supported, please disable swap! /proc/swaps contained: %v", buf)
		}
		glog.Warningf("Running with swap on is not supported, please disable swap! " +
			"This will be a fatal error by default starting in K8s v1.6! " +
			"In the meantime, you can opt-in to making this a fatal error by enabling --experimental-fail-swap-on.")
	}

	// Check if Cgroup-root actually exists on the node
	if nodeConfig.CgroupsPerQOS {
		// this does default to / when enabled, but this tests against regressions.
		if nodeConfig.CgroupRoot == "" {
			return nil, nil, fmt.Errorf("invalid configuration: cgroups-per-qos was specified and cgroup-root was not specified. To enable the QoS cgroup hierarchy you need to specify a valid cgroup-root")
		}

		// we need to check that the cgroup root actually exists for each subsystem
		// of note, we always use the cgroupfs driver when performing this check since
		// the input is provided in that format.
		// this is important because we do not want any name conversion to occur.
		cgroupManager := NewCgroupManager(subsystems, "cgroupfs")
		if !cgroupManager.Exists(CgroupName(nodeConfig.CgroupRoot)) {
			return nil, nil, fmt.Errorf("invalid configuration: cgroup-root doesn't exist: %v", err)
		}
	}
	containerManager := &containerManagerImpl{
		cadvisorInterface: cadvisorInterface,
		mountUtil:         mountUtil,
		NodeConfig:        nodeConfig,
		subsystems:        subsystems,
	}
	return containerManager, containerManager, nil
}

// NewPodContainerManager is a factory method returns a PodContainerManager object
// If qosCgroups are enabled then it returns the general pod container manager
// otherwise it returns a no-op manager which essentially does nothing
func (cm *containerManagerImpl) NewPodContainerManager() PodContainerManager {
	if cm.NodeConfig.CgroupsPerQOS {
		return &podContainerManagerImpl{
			qosContainersInfo: cm.qosContainers,
			nodeInfo:          cm.nodeInfo,
			subsystems:        cm.subsystems,
			cgroupManager:     NewCgroupManager(cm.subsystems, cm.NodeConfig.CgroupDriver),
		}
	}
	return &podContainerManagerNoop{
		cgroupRoot: CgroupName(cm.NodeConfig.CgroupRoot),
	}
}

// Create a cgroup container manager.
func createManager(containerName string) *fs.Manager {
	allowAllDevices := true
	return &fs.Manager{
		Cgroups: &configs.Cgroup{
			Parent: "/",
			Name:   containerName,
			Resources: &configs.Resources{
				AllowAllDevices: &allowAllDevices,
			},
		},
	}
}

type KernelTunableBehavior string

const (
	KernelTunableWarn   KernelTunableBehavior = "warn"
	KernelTunableError  KernelTunableBehavior = "error"
	KernelTunableModify KernelTunableBehavior = "modify"
)

// initQOS creates the top level qos cgroup containers
// We create top level QoS containers for only Burstable and Best Effort
// and not Guaranteed QoS class. All guaranteed pods are nested under the
// RootContainer by default. initQOS is called only once during kubelet bootstrapping.
func initQOS(cgroupDriver, rootContainer string, subsystems *CgroupSubsystems) (QOSContainersInfo, error) {
	cm := NewCgroupManager(subsystems, cgroupDriver)
	// Top level for Qos containers are created only for Burstable
	// and Best Effort classes
	qosClasses := [2]v1.PodQOSClass{v1.PodQOSBurstable, v1.PodQOSBestEffort}

	// Create containers for both qos classes
	for _, qosClass := range qosClasses {
		// get the container's absolute name
		absoluteContainerName := CgroupName(path.Join(rootContainer, string(qosClass)))
		// containerConfig object stores the cgroup specifications
		containerConfig := &CgroupConfig{
			Name:               absoluteContainerName,
			ResourceParameters: &ResourceConfig{},
		}
		// check if it exists
		if !cm.Exists(absoluteContainerName) {
			if err := cm.Create(containerConfig); err != nil {
				return QOSContainersInfo{}, fmt.Errorf("failed to create top level %v QOS cgroup : %v", qosClass, err)
			}
		}
	}
	// Store the top level qos container names
	qosContainersInfo := QOSContainersInfo{
		Guaranteed: rootContainer,
		Burstable:  path.Join(rootContainer, string(v1.PodQOSBurstable)),
		BestEffort: path.Join(rootContainer, string(v1.PodQOSBestEffort)),
	}
	return qosContainersInfo, nil
}

// setupKernelTunables validates kernel tunable flags are set as expected
// depending upon the specified option, it will either warn, error, or modify the kernel tunable flags
func setupKernelTunables(option KernelTunableBehavior) error {
	desiredState := map[string]int{
		utilsysctl.VmOvercommitMemory: utilsysctl.VmOvercommitMemoryAlways,
		utilsysctl.VmPanicOnOOM:       utilsysctl.VmPanicOnOOMInvokeOOMKiller,
		utilsysctl.KernelPanic:        utilsysctl.KernelPanicRebootTimeout,
		utilsysctl.KernelPanicOnOops:  utilsysctl.KernelPanicOnOopsAlways,
	}

	sysctl := utilsysctl.New()

	errList := []error{}
	for flag, expectedValue := range desiredState {
		val, err := sysctl.GetSysctl(flag)
		if err != nil {
			errList = append(errList, err)
			continue
		}
		if val == expectedValue {
			continue
		}

		switch option {
		case KernelTunableError:
			errList = append(errList, fmt.Errorf("Invalid kernel flag: %v, expected value: %v, actual value: %v", flag, expectedValue, val))
		case KernelTunableWarn:
			glog.V(2).Infof("Invalid kernel flag: %v, expected value: %v, actual value: %v", flag, expectedValue, val)
		case KernelTunableModify:
			glog.V(2).Infof("Updating kernel flag: %v, expected value: %v, actual value: %v", flag, expectedValue, val)
			err = sysctl.SetSysctl(flag, expectedValue)
			if err != nil {
				errList = append(errList, err)
			}
		}
	}
	return utilerrors.NewAggregate(errList)
}

func setQOSCgroupMemoryLimit(cgroupManager CgroupManager, qosClass v1.PodQOSClass, limit int64) error {
	glog.V(2).Infof("[Container Manager] Setting %s QoS memory limit to %d bytes", qosClass, limit)
	// Attempt to set specified limit
	cgroupConfig := CgroupConfig{
		Name: cgroupManager.CgroupName(string(qosClass)),
		ResourceParameters: &ResourceConfig{
			Memory: &limit,
		},
	}
	err := cgroupManager.Update(&cgroupConfig)
	if err == nil {
		return nil
	}

	// Unreclaimable memory usage already exceeds the desired limit.
	// Attempt to set the limit near the current usage to put pressure
	// on the cgroup and prevent further growth.
	glog.Warningf("[Container Manager] %s QoS memory usage is already above desired limit", qosClass)
	stats, err := cgroupManager.Get(cgroupConfig.Name)
	if err != nil {
		return err
	}
	usageLimit := stats.MemoryStats.Usage + memoryUsageMargin
	cgroupConfig.ResourceParameters.Memory = &usageLimit
	glog.V(2).Infof("[Container Manager] Setting %s QoS memory limit to %d bytes", qosClass, usageLimit)
	return cgroupManager.Update(&cgroupConfig)
}

// syncQOSMemoryLimts sums the memory limits of all pods in a QOS class,
// calculate QOS class memory limits, and set those limits in the QOS cgroups
func setQOSMemoryReserveLimit(cm *containerManagerImpl, reservePercent int64) error {
	qosMemoryLimits := map[v1.PodQOSClass]int64{
		v1.PodQOSGuaranteed: 0,
		v1.PodQOSBurstable:  0,
	}

	// Sum the pod limits for pods in each QOS class
	pods := cm.getPodsFunc()
	for _, pod := range pods {
		podMemoryLimit := int64(0)
		qosClass := qos.GetPodQOS(pod)
		if qosClass == v1.PodQOSBestEffort {
			// limits are not set for Best Effort pods
			continue
		}
		for _, container := range pod.Spec.Containers {
			podMemoryLimit += container.Resources.Limits.Memory().Value()
		}
		qosMemoryLimits[qosClass] += podMemoryLimit
	}

	allocatable := cm.nodeInfo.Status.Allocatable.Memory().Value()
	if allocatable == 0 {
		return fmt.Errorf("memory capacity reported as 0, might be in standalone mode")
	}

	for qos, limits := range qosMemoryLimits {
		glog.V(2).Infof("[Container Manager] %s pod limits total %d bytes (reserve %d%%)", qos, limits, reservePercent)
	}

	// Calculate QOS memory limits
	burstableLimit := allocatable - (qosMemoryLimits[v1.PodQOSGuaranteed] * reservePercent / 100)
	if burstableLimit < minimumLimit {
		burstableLimit = minimumLimit
	}
	bestEffortLimit := burstableLimit - (qosMemoryLimits[v1.PodQOSBurstable] * reservePercent / 100)
	if bestEffortLimit < minimumLimit {
		bestEffortLimit = minimumLimit
	}

	// Update QOS class cgroups with calculated memory limits
	subsystems, err := GetCgroupSubsystems()
	if err != nil {
		return err
	}

	cgroupManager := NewCgroupManager(subsystems, cm.CgroupDriver)
	err = setQOSCgroupMemoryLimit(cgroupManager, v1.PodQOSBurstable, burstableLimit)
	if err != nil {
		return err
	}
	err = setQOSCgroupMemoryLimit(cgroupManager, v1.PodQOSBestEffort, bestEffortLimit)
	if err != nil {
		return err
	}

	return nil
}

type qosReserveLimitHandler func(*containerManagerImpl, int64) error

var resourceQOSReserveLimitHandlers = map[string]qosReserveLimitHandler{
	"memory": setQOSMemoryReserveLimit,
}

// If usage is already above the limit, set the limit to the usage plus this margin
// This increases the likelyhood that the limit can be set successfully and gives the
// cgroup a little forgiveness so it doesn't OOM kill in response to a little growth.
const memoryUsageMargin = 10 * 1024 * 1024 // 10MB

// The minumum limit to which a QoS level cgroup can be set
const minimumLimit = 512 * 1024 * 1024 // 512MB

func (cm *containerManagerImpl) setQOSReserveLimits() error {
	for _, reserveLimit := range cm.NodeConfig.QOSReserveLimits {
		handler, ok := resourceQOSReserveLimitHandlers[reserveLimit.resource]
		if !ok {
			// programming error if we hit this
			return fmt.Errorf("no reserve limit hanlder for resource '%s'", reserveLimit.resource)
		}
		if err := handler(cm, reserveLimit.reservePercent); err != nil {
			return err
		}
	}
	return nil
}

// Admit always returns Admit: true but allows the ConatinerManager to hook into the admit path
func (cm *containerManagerImpl) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	glog.V(2).Infof("[Container Manager] Updating QoS reserve limits on pod admission")
	if cm.CgroupsPerQOS {
		if err := cm.setQOSReserveLimits(); err != nil {
			glog.Warningf("[ContainerManager] Failed to set QoS reserve limits at pod admission: %v", err)
			// don't block admission on error here
		}
	}
	return lifecycle.PodAdmitResult{
		Admit: true,
	}
}

func (cm *containerManagerImpl) setupNode() error {
	f, err := validateSystemRequirements(cm.mountUtil)
	if err != nil {
		return err
	}
	if !f.cpuHardcapping {
		cm.status.SoftRequirements = fmt.Errorf("CPU hardcapping unsupported")
	}
	b := KernelTunableModify
	if cm.GetNodeConfig().ProtectKernelDefaults {
		b = KernelTunableError
	}
	if err := setupKernelTunables(b); err != nil {
		return err
	}

	// Setup top level qos containers only if CgroupsPerQOS flag is specified as true
	if cm.NodeConfig.CgroupsPerQOS {
		qosContainersInfo, err := initQOS(cm.NodeConfig.CgroupDriver, cm.NodeConfig.CgroupRoot, cm.subsystems)
		if err != nil {
			return fmt.Errorf("failed to initialise top level QOS containers: %v", err)
		}
		cm.qosContainers = qosContainersInfo
	}

	systemContainers := []*systemContainer{}
	if cm.ContainerRuntime == "docker" {
		dockerVersion := getDockerVersion(cm.cadvisorInterface)
		if cm.EnableCRI {
			// If kubelet uses CRI, dockershim will manage the cgroups and oom
			// score for the docker processes.
			// In the future, NodeSpec should mandate the cgroup that the
			// runtime processes need to be in. For now, we still check the
			// cgroup for docker periodically, so that kubelet can recognize
			// the cgroup for docker and serve stats for the runtime.
			// TODO(#27097): Fix this after NodeSpec is clearly defined.
			cm.periodicTasks = append(cm.periodicTasks, func() {
				glog.V(4).Infof("[ContainerManager]: Adding periodic tasks for docker CRI integration")
				cont, err := getContainerNameForProcess(dockerProcessName, dockerPidFile)
				if err != nil {
					glog.Error(err)
					return
				}
				glog.V(2).Infof("[ContainerManager]: Discovered runtime cgroups name: %s", cont)
				cm.Lock()
				defer cm.Unlock()
				cm.RuntimeCgroupsName = cont
			})
		} else if cm.RuntimeCgroupsName != "" {
			cont := newSystemCgroups(cm.RuntimeCgroupsName)
			var capacity = v1.ResourceList{}
			if info, err := cm.cadvisorInterface.MachineInfo(); err == nil {
				capacity = cadvisor.CapacityFromMachineInfo(info)
			}
			memoryLimit := (int64(capacity.Memory().Value() * DockerMemoryLimitThresholdPercent / 100))
			if memoryLimit < MinDockerMemoryLimit {
				glog.Warningf("Memory limit %d for container %s is too small, reset it to %d", memoryLimit, cm.RuntimeCgroupsName, MinDockerMemoryLimit)
				memoryLimit = MinDockerMemoryLimit
			}

			glog.V(2).Infof("Configure resource-only container %s with memory limit: %d", cm.RuntimeCgroupsName, memoryLimit)
			allowAllDevices := true
			dockerContainer := &fs.Manager{
				Cgroups: &configs.Cgroup{
					Parent: "/",
					Name:   cm.RuntimeCgroupsName,
					Resources: &configs.Resources{
						Memory:          memoryLimit,
						MemorySwap:      -1,
						AllowAllDevices: &allowAllDevices,
					},
				},
			}
			cont.ensureStateFunc = func(manager *fs.Manager) error {
				return EnsureDockerInContainer(dockerVersion, qos.DockerOOMScoreAdj, dockerContainer)
			}
			systemContainers = append(systemContainers, cont)
		} else {
			cm.periodicTasks = append(cm.periodicTasks, func() {
				glog.V(10).Infof("Adding docker daemon periodic tasks")
				if err := EnsureDockerInContainer(dockerVersion, qos.DockerOOMScoreAdj, nil); err != nil {
					glog.Error(err)
					return
				}
				cont, err := getContainerNameForProcess(dockerProcessName, dockerPidFile)
				if err != nil {
					glog.Error(err)
					return
				}
				glog.V(2).Infof("Discovered runtime cgroups name: %s", cont)
				cm.Lock()
				defer cm.Unlock()
				cm.RuntimeCgroupsName = cont
			})
		}
	}

	if cm.SystemCgroupsName != "" {
		if cm.SystemCgroupsName == "/" {
			return fmt.Errorf("system container cannot be root (\"/\")")
		}
		cont := newSystemCgroups(cm.SystemCgroupsName)
		cont.ensureStateFunc = func(manager *fs.Manager) error {
			return ensureSystemCgroups("/", manager)
		}
		systemContainers = append(systemContainers, cont)
	}

	if cm.KubeletCgroupsName != "" {
		cont := newSystemCgroups(cm.KubeletCgroupsName)
		allowAllDevices := true
		manager := fs.Manager{
			Cgroups: &configs.Cgroup{
				Parent: "/",
				Name:   cm.KubeletCgroupsName,
				Resources: &configs.Resources{
					AllowAllDevices: &allowAllDevices,
				},
			},
		}
		cont.ensureStateFunc = func(_ *fs.Manager) error {
			return ensureProcessInContainerWithOOMScore(os.Getpid(), qos.KubeletOOMScoreAdj, &manager)
		}
		systemContainers = append(systemContainers, cont)
	} else {
		cm.periodicTasks = append(cm.periodicTasks, func() {
			if err := ensureProcessInContainerWithOOMScore(os.Getpid(), qos.KubeletOOMScoreAdj, nil); err != nil {
				glog.Error(err)
				return
			}
			cont, err := getContainer(os.Getpid())
			if err != nil {
				glog.Errorf("failed to find cgroups of kubelet - %v", err)
				return
			}
			cm.Lock()
			defer cm.Unlock()

			cm.KubeletCgroupsName = cont
		})
	}

	cm.systemContainers = systemContainers
	return nil
}

func getContainerNameForProcess(name, pidFile string) (string, error) {
	pids, err := getPidsForProcess(name, pidFile)
	if err != nil {
		return "", fmt.Errorf("failed to detect process id for %q - %v", name, err)
	}
	if len(pids) == 0 {
		return "", nil
	}
	cont, err := getContainer(pids[0])
	if err != nil {
		return "", err
	}
	return cont, nil
}

func (cm *containerManagerImpl) GetNodeConfig() NodeConfig {
	cm.RLock()
	defer cm.RUnlock()
	return cm.NodeConfig
}

func (cm *containerManagerImpl) GetMountedSubsystems() *CgroupSubsystems {
	return cm.subsystems
}

func (cm *containerManagerImpl) GetQOSContainersInfo() QOSContainersInfo {
	return cm.qosContainers
}

func (cm *containerManagerImpl) Status() Status {
	cm.RLock()
	defer cm.RUnlock()
	return cm.status
}

func (cm *containerManagerImpl) Start(node *v1.Node, getPods func() []*v1.Pod) error {
	// cache the node Info including resource capacity and
	// allocatable of the node
	cm.nodeInfo = node
	cm.getPodsFunc = getPods
	// Setup the node
	if err := cm.setupNode(); err != nil {
		return err
	}
	// Don't run a background thread if there are no ensureStateFuncs.
	hasEnsureStateFuncs := false
	for _, cont := range cm.systemContainers {
		if cont.ensureStateFunc != nil {
			hasEnsureStateFuncs = true
			break
		}
	}
	if hasEnsureStateFuncs {
		// Run ensure state functions every minute.
		go wait.Until(func() {
			for _, cont := range cm.systemContainers {
				if cont.ensureStateFunc != nil {
					if err := cont.ensureStateFunc(cont.manager); err != nil {
						glog.Warningf("[ContainerManager] Failed to ensure state of %q: %v", cont.name, err)
					}
				}
			}
		}, time.Minute, wait.NeverStop)

	}

	if cm.CgroupsPerQOS && len(cm.QOSReserveLimits) != 0 {
		// sync qos level limits every 10 seconds.
		go wait.Until(func() {
			err := cm.setQOSReserveLimits()
			if err != nil {
				glog.Warningf("[ContainerManager] Failed to set QoS reserve limits: %v", err)
			}
		}, 10*time.Second, wait.NeverStop)
	}

	if len(cm.periodicTasks) > 0 {
		go wait.Until(func() {
			for _, task := range cm.periodicTasks {
				if task != nil {
					task()
				}
			}
		}, 5*time.Minute, wait.NeverStop)
	}

	return nil
}

func (cm *containerManagerImpl) SystemCgroupsLimit() v1.ResourceList {
	cpuLimit := int64(0)

	// Sum up resources of all external containers.
	for _, cont := range cm.systemContainers {
		cpuLimit += cont.cpuMillicores
	}

	return v1.ResourceList{
		v1.ResourceCPU: *resource.NewMilliQuantity(
			cpuLimit,
			resource.DecimalSI),
	}
}

func isProcessRunningInHost(pid int) (bool, error) {
	// Get init pid namespace.
	initPidNs, err := os.Readlink("/proc/1/ns/pid")
	if err != nil {
		return false, fmt.Errorf("failed to find pid namespace of init process")
	}
	glog.V(10).Infof("init pid ns is %q", initPidNs)
	processPidNs, err := os.Readlink(fmt.Sprintf("/proc/%d/ns/pid", pid))
	if err != nil {
		return false, fmt.Errorf("failed to find pid namespace of process %q", pid)
	}
	glog.V(10).Infof("Pid %d pid ns is %q", pid, processPidNs)
	return initPidNs == processPidNs, nil
}

func getPidFromPidFile(pidFile string) (int, error) {
	file, err := os.Open(pidFile)
	if err != nil {
		return 0, fmt.Errorf("error opening pid file %s: %v", pidFile, err)
	}
	defer file.Close()

	data, err := ioutil.ReadAll(file)
	if err != nil {
		return 0, fmt.Errorf("error reading pid file %s: %v", pidFile, err)
	}

	pid, err := strconv.Atoi(string(data))
	if err != nil {
		return 0, fmt.Errorf("error parsing %s as a number: %v", string(data), err)
	}

	return pid, nil
}

func getPidsForProcess(name, pidFile string) ([]int, error) {
	if len(pidFile) > 0 {
		if pid, err := getPidFromPidFile(pidFile); err == nil {
			return []int{pid}, nil
		} else {
			// log the error and fall back to pidof
			runtime.HandleError(err)
		}
	}
	return procfs.PidOf(name)
}

// Ensures that the Docker daemon is in the desired container.
// Temporarily export the function to be used by dockershim.
// TODO(yujuhong): Move this function to dockershim once kubelet migrates to
// dockershim as the default.
func EnsureDockerInContainer(dockerVersion *utilversion.Version, oomScoreAdj int, manager *fs.Manager) error {
	type process struct{ name, file string }
	dockerProcs := []process{{dockerProcessName, dockerPidFile}}
	if dockerVersion.AtLeast(containerdVersion) {
		dockerProcs = append(dockerProcs, process{containerdProcessName, containerdPidFile})
	}
	var errs []error
	for _, proc := range dockerProcs {
		pids, err := getPidsForProcess(proc.name, proc.file)
		if err != nil {
			errs = append(errs, fmt.Errorf("failed to get pids for %q: %v", proc.name, err))
			continue
		}

		// Move if the pid is not already in the desired container.
		for _, pid := range pids {
			if err := ensureProcessInContainerWithOOMScore(pid, oomScoreAdj, manager); err != nil {
				errs = append(errs, fmt.Errorf("errors moving %q pid: %v", proc.name, err))
			}
		}
	}
	return utilerrors.NewAggregate(errs)
}

func ensureProcessInContainerWithOOMScore(pid int, oomScoreAdj int, manager *fs.Manager) error {
	if runningInHost, err := isProcessRunningInHost(pid); err != nil {
		// Err on the side of caution. Avoid moving the docker daemon unless we are able to identify its context.
		return err
	} else if !runningInHost {
		// Process is running inside a container. Don't touch that.
		glog.V(2).Infof("pid %d is not running in the host namespaces", pid)
		return nil
	}

	var errs []error
	if manager != nil {
		cont, err := getContainer(pid)
		if err != nil {
			errs = append(errs, fmt.Errorf("failed to find container of PID %d: %v", pid, err))
		}

		if cont != manager.Cgroups.Name {
			err = manager.Apply(pid)
			if err != nil {
				errs = append(errs, fmt.Errorf("failed to move PID %d (in %q) to %q: %v", pid, cont, manager.Cgroups.Name, err))
			}
		}
	}

	// Also apply oom-score-adj to processes
	oomAdjuster := oom.NewOOMAdjuster()
	glog.V(5).Infof("attempting to apply oom_score_adj of %d to pid %d", oomScoreAdj, pid)
	if err := oomAdjuster.ApplyOOMScoreAdj(pid, oomScoreAdj); err != nil {
		glog.V(3).Infof("Failed to apply oom_score_adj %d for pid %d: %v", oomScoreAdj, pid, err)
		errs = append(errs, fmt.Errorf("failed to apply oom score %d to PID %d: %v", oomScoreAdj, pid, err))
	}
	return utilerrors.NewAggregate(errs)
}

// getContainer returns the cgroup associated with the specified pid.
// It enforces a unified hierarchy for memory and cpu cgroups.
// On systemd environments, it uses the name=systemd cgroup for the specified pid.
func getContainer(pid int) (string, error) {
	cgs, err := cgroups.ParseCgroupFile(fmt.Sprintf("/proc/%d/cgroup", pid))
	if err != nil {
		return "", err
	}

	cpu, found := cgs["cpu"]
	if !found {
		return "", cgroups.NewNotFoundError("cpu")
	}
	memory, found := cgs["memory"]
	if !found {
		return "", cgroups.NewNotFoundError("memory")
	}

	// since we use this container for accounting, we need to ensure its a unified hierarchy.
	if cpu != memory {
		return "", fmt.Errorf("cpu and memory cgroup hierarchy not unified.  cpu: %s, memory: %s", cpu, memory)
	}

	// on systemd, every pid is in a unified cgroup hierarchy (name=systemd as seen in systemd-cgls)
	// cpu and memory accounting is off by default, users may choose to enable it per unit or globally.
	// users could enable CPU and memory accounting globally via /etc/systemd/system.conf (DefaultCPUAccounting=true DefaultMemoryAccounting=true).
	// users could also enable CPU and memory accounting per unit via CPUAccounting=true and MemoryAccounting=true
	// we only warn if accounting is not enabled for CPU or memory so as to not break local development flows where kubelet is launched in a terminal.
	// for example, the cgroup for the user session will be something like /user.slice/user-X.slice/session-X.scope, but the cpu and memory
	// cgroup will be the closest ancestor where accounting is performed (most likely /) on systems that launch docker containers.
	// as a result, on those systems, you will not get cpu or memory accounting statistics for kubelet.
	// in addition, you would not get memory or cpu accounting for the runtime unless accounting was enabled on its unit (or globally).
	if systemd, found := cgs["name=systemd"]; found {
		if systemd != cpu {
			glog.Warningf("CPUAccounting not enabled for pid: %d", pid)
		}
		if systemd != memory {
			glog.Warningf("MemoryAccounting not enabled for pid: %d", pid)
		}
		return systemd, nil
	}

	return cpu, nil
}

// Ensures the system container is created and all non-kernel threads and process 1
// without a container are moved to it.
//
// The reason of leaving kernel threads at root cgroup is that we don't want to tie the
// execution of these threads with to-be defined /system quota and create priority inversions.
//
func ensureSystemCgroups(rootCgroupPath string, manager *fs.Manager) error {
	// Move non-kernel PIDs to the system container.
	attemptsRemaining := 10
	var errs []error
	for attemptsRemaining >= 0 {
		// Only keep errors on latest attempt.
		errs = []error{}
		attemptsRemaining--

		allPids, err := cmutil.GetPids(rootCgroupPath)
		if err != nil {
			errs = append(errs, fmt.Errorf("failed to list PIDs for root: %v", err))
			continue
		}

		// Remove kernel pids and other protected PIDs (pid 1, PIDs already in system & kubelet containers)
		pids := make([]int, 0, len(allPids))
		for _, pid := range allPids {
			if pid == 1 || isKernelPid(pid) {
				continue
			}

			pids = append(pids, pid)
		}
		glog.Infof("Found %d PIDs in root, %d of them are not to be moved", len(allPids), len(allPids)-len(pids))

		// Check if we have moved all the non-kernel PIDs.
		if len(pids) == 0 {
			break
		}

		glog.Infof("Moving non-kernel processes: %v", pids)
		for _, pid := range pids {
			err := manager.Apply(pid)
			if err != nil {
				errs = append(errs, fmt.Errorf("failed to move PID %d into the system container %q: %v", pid, manager.Cgroups.Name, err))
			}
		}

	}
	if attemptsRemaining < 0 {
		errs = append(errs, fmt.Errorf("ran out of attempts to create system containers %q", manager.Cgroups.Name))
	}

	return utilerrors.NewAggregate(errs)
}

// Determines whether the specified PID is a kernel PID.
func isKernelPid(pid int) bool {
	// Kernel threads have no associated executable.
	_, err := os.Readlink(fmt.Sprintf("/proc/%d/exe", pid))
	return err != nil
}

// Helper for getting the docker version.
func getDockerVersion(cadvisor cadvisor.Interface) *utilversion.Version {
	versions, err := cadvisor.VersionInfo()
	if err != nil {
		glog.Errorf("Error requesting cAdvisor VersionInfo: %v", err)
		return utilversion.MustParseSemantic("0.0.0")
	}
	dockerVersion, err := utilversion.ParseSemantic(versions.DockerVersion)
	if err != nil {
		glog.Errorf("Error parsing docker version %q: %v", versions.DockerVersion, err)
		return utilversion.MustParseSemantic("0.0.0")
	}
	return dockerVersion
}
