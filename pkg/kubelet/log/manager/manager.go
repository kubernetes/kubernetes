package manager

import (
	"context"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/golang/glog"
	"google.golang.org/grpc"
	"k8s.io/api/core/v1"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/record"
	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/logplugin/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/config"
	"k8s.io/kubernetes/pkg/kubelet/configmap"
	"k8s.io/kubernetes/pkg/kubelet/log/manager/util"
	"k8s.io/kubernetes/pkg/kubelet/log/policy"
	"k8s.io/kubernetes/pkg/kubelet/pod"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/kubelet/volumemanager"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

const (
	resyncPeriod            = 1 * time.Minute
	podLogPolicyCategoryStd = "std"
	containerLogDirPerm     = 0766
)

type sourcesReadyStub struct{}

func (s *sourcesReadyStub) AddSource(source string) {}

func (s *sourcesReadyStub) AllReady() bool { return true }

type ManagerImpl struct {
	kubeClient   clientset.Interface
	recorder     record.EventRecorder
	sourcesReady config.SourcesReady
	// logPluginName -> logPluginEndpoint
	mutex      sync.Mutex
	logPlugins map[string]pluginEndpoint
	// gRPC server
	socketDir  string
	socketName string
	server     *grpc.Server
	// managers
	policyStatusManager policy.LogStatusManager
	pluginStatusManager *pluginStatusManager
	configMapWatcher    *util.ConfigMapWatcher
	// kubelet managers
	podManager       pod.Manager
	configMapManager configmap.Manager
	volumeManager    volumemanager.VolumeManager
}

func NewLogPluginManagerImpl(
	kubeClient clientset.Interface,
	recorder record.EventRecorder,
	podManager pod.Manager,
	configMapManager configmap.Manager,
	volumeManager volumemanager.VolumeManager,
	policyStatusManager policy.LogStatusManager,
) (Manager, error) {
	socketDir, socketName := filepath.Split(pluginapi.KubeletSocket)
	m := &ManagerImpl{
		kubeClient:          kubeClient,
		recorder:            recorder,
		sourcesReady:        &sourcesReadyStub{},
		logPlugins:          make(map[string]pluginEndpoint),
		socketDir:           socketDir,
		socketName:          socketName,
		policyStatusManager: policyStatusManager,
		pluginStatusManager: newPluginStatusManager(),
		podManager:          podManager,
		configMapManager:    configMapManager,
		volumeManager:       volumeManager,
	}
	m.configMapWatcher = util.NewConfigMapWatcher(configMapManager, m.onConfigMapUpdate)
	return m, nil
}

func (m *ManagerImpl) cleanUpDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer d.Close()
	names, err := d.Readdirnames(-1)
	if err != nil {
		return err
	}
	for _, name := range names {
		filePath := filepath.Join(dir, name)
		stat, err := os.Stat(filePath)
		if err != nil {
			glog.Errorf("failed to stat file %s: %v", filePath, err)
			continue
		}
		if stat.IsDir() {
			continue
		}
		err = os.RemoveAll(filePath)
		if err != nil {
			return err
		}
	}
	return nil
}

func (m *ManagerImpl) Start(sourcesReady config.SourcesReady) error {
	glog.V(2).Infof("starting log plugin manager")
	m.sourcesReady = sourcesReady

	socketPath := filepath.Join(m.socketDir, m.socketName)
	os.MkdirAll(m.socketDir, 0755)

	// Removes all stale sockets in m.socketDir. Log plugins can monitor
	// this and use it as a signal to re-register with the new Kubelet.
	if err := m.cleanUpDir(m.socketDir); err != nil {
		glog.Errorf("fail to clean up stale contents under %s: %+v", m.socketDir, err)
		return err
	}

	s, err := net.Listen("unix", socketPath)
	if err != nil {
		glog.Errorf("server listen error, %v, socket path: %s", err, socketPath)
		return err
	}

	m.server = grpc.NewServer([]grpc.ServerOption{}...)

	pluginapi.RegisterRegistrationServer(m.server, m)
	go m.server.Serve(s)
	glog.V(2).Infof("serving log plugin registration server on %q", socketPath)

	go wait.Until(m.sync, resyncPeriod, wait.NeverStop)

	return nil
}

func filterPods(allPods []*v1.Pod) []*v1.Pod {
	pods := make([]*v1.Pod, 0)
	for _, p := range allPods {
		if policy.IsPodLogPolicyExists(p) {
			pods = append(pods, p)
		}
	}
	return pods
}

// ensure consistent between pod log policy and log plugin config
// 1. get all configs from log plugins
// 2. save config states to stateManager
// 3. traversal all pods with log policy, diff configs between log policy configs and log plugin configs
// 4. update configs from log policy to log plugin
func (m *ManagerImpl) sync() {
	// if !m.sourcesReady.AllReady() {
	// 	return
	// }

	glog.V(7).Infof("log manager sync start")
	// get all pod log configs from log plugins and update inner state
	for logPluginName, endpoint := range m.logPlugins {
		err := m.refreshPluginState(endpoint)
		if err != nil {
			glog.Errorf("pull pod log configs error, %v, endpoint name: %s", err, logPluginName)
			return
		}
	}

	// get all pods with log policy
	pods := filterPods(m.podManager.GetPods())

	// sync pod state to log plugin state
	for _, pod := range pods {
		//if volumeutil.IsPodTerminated(pod, pod.Status) {
		//	glog.V(4).Infof("pod is terminated, skip sync, pod: %q", format.Pod(pod))
		//	continue
		//}
		err := m.refreshPodState(pod)
		if err != nil {
			glog.Errorf("refresh pod log policy error, %v, pod: %q", err, format.Pod(pod))
			continue
		}
		err = m.pushConfigs(pod)
		if err != nil {
			glog.Errorf("push pod log configs error, %v, pod: %q", err, format.Pod(pod))
			continue
		}
	}

	for podUID := range m.pluginStatusManager.getAllPodUIDs() {
		podUID := k8stypes.UID(podUID)
		_, e1 := m.podManager.GetPodByUID(podUID)
		_, e2 := m.policyStatusManager.GetLogPolicy(podUID)
		if e1 || e2 {
			continue
		}
		glog.Infof("removing pod which not found in k8s, pod uid: %s", podUID)
		pluginName, exists := m.pluginStatusManager.getLogPluginName(podUID)
		if !exists {
			glog.Errorf("log plugin endpoint not found, pod uid: %s", podUID)
			continue
		}
		endpoint, err := m.getLogPluginEndpoint(pluginName)
		if err != nil {
			glog.Errorf("get log plugin endpoint error, %v, log plugin name: %s, pod uid: %s", err, pluginName, podUID)
			continue
		}
		err = m.deletePluginConfigs(podUID, endpoint)
		if err != nil {
			glog.Errorf("delete pod log configs error, %v, log plugin name: %s, pod uid: %s", err, pluginName, podUID)
			continue
		}
		m.removePodState(podUID)
	}

	m.configMapWatcher.Sync(m.policyStatusManager.GetAllConfigMapKeys())
	glog.V(7).Infof("log manager sync finished")
}

func (m *ManagerImpl) refreshPluginState(endpoint pluginEndpoint) error {
	rsp, err := endpoint.listConfig()
	if err != nil {
		glog.Errorf("list configs from log plugin error, %v", err)
		return err
	}

	glog.V(7).Infof("update all log configs: %+v", rsp.Configs)
	m.pluginStatusManager.updateAllLogConfigs(rsp.Configs, endpoint.name())
	return nil
}

func (m *ManagerImpl) createPodLogDirSymLink(logVolumes policy.LogVolumesMap) error {
	// create symlink for log volumes
	// eg. /var/log/pods/<pod-uid>/<container-name>/<category>
	// we should make dir /var/log/pods/<pod-uid>/<container-name> first and then create symlink <category>
	for _, logVolume := range logVolumes {
		containerLogDirPath := filepath.Dir(logVolume.LogDirPath)
		if _, err := os.Stat(containerLogDirPath); os.IsNotExist(err) {
			glog.V(4).Infof("creating log dir %q", containerLogDirPath)
			mkdirErr := os.MkdirAll(containerLogDirPath, containerLogDirPerm)
			if mkdirErr != nil {
				glog.Errorf("mkdir %q error, %v", containerLogDirPath, err)
				return mkdirErr
			}
		}
		if _, err := os.Lstat(logVolume.LogDirPath); os.IsNotExist(err) {
			glog.V(4).Infof("creating log dir symbolic link from %q to %q", logVolume.LogDirPath, logVolume.HostPath)
			slErr := os.Symlink(logVolume.HostPath, logVolume.LogDirPath)
			if slErr != nil {
				glog.Errorf("create symbolic link from %q to %q error, %v", logVolume.LogDirPath, logVolume.HostPath, err)
				return slErr
			}
		}
	}
	return nil
}

// Register registers a log plugin.
func (m *ManagerImpl) Register(ctx context.Context, r *pluginapi.RegisterRequest) (*pluginapi.Empty, error) {
	glog.Infof("got registration request from log plugin: %s, endpoint: %s", r.Name, r.Endpoint)
	if r.Version != pluginapi.Version {
		err := fmt.Errorf("invalid version: %s, expected: %s", r.Version, pluginapi.Version)
		glog.Infof("bad registration request from log plugin, %v", err)
		return &pluginapi.Empty{}, err
	}

	go m.addEndpoint(r)

	return &pluginapi.Empty{}, nil
}

func (m *ManagerImpl) addEndpoint(r *pluginapi.RegisterRequest) {
	glog.Infof("endpoint %q is registering", r.Name)
	socketPath := filepath.Join(pluginapi.LogPluginPath, r.Endpoint)
	e, err := newEndpointImpl(socketPath, r.Name)
	if err != nil {
		glog.Errorf("create endpoint error, %v, log plugin name: %s, socket path: %s", err, r.Name, socketPath)
		return
	}

	m.mutex.Lock()
	oldLogPlugin, exists := m.logPlugins[r.Name]
	if exists && oldLogPlugin != nil {
		oldLogPlugin.stop()
	}
	m.logPlugins[r.Name] = e
	m.mutex.Unlock()
	glog.Infof("endpoint %q is registered, socket path: %s", r.Name, socketPath)
}

func (m *ManagerImpl) buildPodLogVolumes(pod *v1.Pod, podLogPolicy *policy.PodLogPolicy) (policy.LogVolumesMap, error) {
	logVolumes := make(policy.LogVolumesMap)
	glog.V(7).Infof("getting mounted volume for pod: %q", format.Pod(pod))
	podVolumes := m.volumeManager.GetMountedVolumesForPod(volumeutil.GetUniquePodName(pod))
	glog.V(7).Infof("got mounted pod volumes: %+v, pod: %q", podVolumes, format.Pod(pod))
	for containerName, containerLogPolicies := range podLogPolicy.ContainerLogPolicies {
		for _, containerLogPolicy := range containerLogPolicies {
			if containerLogPolicy.Category == podLogPolicyCategoryStd {
				continue
			}
			volumeInfo, exists := podVolumes[containerLogPolicy.VolumeName]
			if !exists {
				err := fmt.Errorf("%q is not found in podVolumes, pod: %q", containerLogPolicy.VolumeName, format.Pod(pod))
				glog.Error(err)
				return nil, err
			}
			logVolume := &policy.LogVolume{
				VolumeName: containerLogPolicy.VolumeName,
				Path:       containerLogPolicy.Path,
				HostPath:   volumeInfo.Mounter.GetPath(),
				LogDirPath: buildLogPolicyDirectory(pod.UID, containerName, containerLogPolicy.Category),
			}
			logVolumes[containerLogPolicy.VolumeName] = logVolume
		}
	}
	return logVolumes, nil
}

func (m *ManagerImpl) buildPodLogConfigMapKeys(pod *v1.Pod, podLogPolicy *policy.PodLogPolicy) (sets.String, error) {
	// configMap key set
	configMapKeys := sets.NewString()
	for _, containerLogPolicies := range podLogPolicy.ContainerLogPolicies {
		for _, containerLogPolicy := range containerLogPolicies {
			// get log config from configmap
			configMap, err := m.configMapManager.GetConfigMap(pod.Namespace, containerLogPolicy.PluginConfigMap)
			if err != nil {
				glog.Errorf("get configmap error, %v, namespace: %s, name: %s, pod: %q", err, pod.Namespace, containerLogPolicy.PluginConfigMap, format.Pod(pod))
				return nil, err
			}
			configMapKeys.Insert(buildConfigMapKey(configMap.Namespace, configMap.Name))
		}
	}
	return configMapKeys, nil
}

func (m *ManagerImpl) buildPodLogConfigs(pod *v1.Pod, podLogPolicy *policy.PodLogPolicy, podLogVolumes policy.LogVolumesMap) (logConfigsMap, error) {
	// configName -> PluginLogConfig
	logConfigs := make(logConfigsMap)
	for containerName, containerLogPolicies := range podLogPolicy.ContainerLogPolicies {
		for _, containerLogPolicy := range containerLogPolicies {
			// get log config from configmap
			configMap, err := m.configMapManager.GetConfigMap(pod.Namespace, containerLogPolicy.PluginConfigMap)
			if err != nil {
				glog.Errorf("get configmap error, %v, namespace: %s, name: %s, pod: %q", err, pod.Namespace, containerLogPolicy.PluginConfigMap, format.Pod(pod))
				return nil, err
			}

			var path string
			if containerLogPolicy.Category == podLogPolicyCategoryStd {
				path = buildPodLogsDirectory(pod.UID)
			} else {
				logVolume, exists := podLogVolumes[containerLogPolicy.VolumeName]
				if !exists {
					glog.Errorf("volume is not found in log policy, volume name: %s, pod: %q, log policy: %v, log volumes: %v", containerLogPolicy.VolumeName, format.Pod(pod), podLogPolicy, podLogVolumes)
					continue
				}
				path = logVolume.LogDirPath
			}

			// build log config
			for filename, content := range configMap.Data {
				configName := buildLogConfigName(pod.UID, containerName, containerLogPolicy.Category, filename)
				logConfigs[configName] = &pluginapi.Config{
					Metadata: &pluginapi.ConfigMeta{
						Name:          configName,
						PodNamespace:  pod.Namespace,
						PodName:       pod.Name,
						PodUID:        string(pod.UID),
						ContainerName: containerName,
					},
					Spec: &pluginapi.ConfigSpec{
						Content:  content,
						Path:     path,
						Category: containerLogPolicy.Category,
					},
				}
			}
		}
	}
	return logConfigs, nil
}

func (m *ManagerImpl) pushPluginConfigs(podUID k8stypes.UID, endpoint pluginEndpoint, logConfigs logConfigsMap) error {
	// diff between logConfigs and podLogPolicyManager.logConfigs
	// generate deleted config name set
	configNames := sets.NewString()
	for configName := range logConfigs {
		configNames.Insert(configName)
	}
	deleted := m.pluginStatusManager.getLogConfigNames(podUID).Difference(configNames)

	// delete config from log plugin
	for configName := range deleted {
		// invoke log plugin api to delete config
		glog.V(7).Infof("calling log plugin to delete config for pod, pod uid: %s", podUID)
		rsp, err := endpoint.delConfig(configName)
		if err != nil {
			glog.Errorf("delete config to log plugin error, %v, config name: %s, pod uid: %s", err, configName, podUID)
			return err
		}
		glog.V(7).Infof("pod log plugin config deleted, pod uid: %s, changed: %t", podUID, rsp.Changed)
	}

	// add config to log plugin
	for _, config := range logConfigs {
		// invoke log plugin api to add config
		glog.Infof("calling log plugin to add config for pod, pod uid: %s", podUID)
		rsp, err := endpoint.addConfig(config)
		if err != nil {
			glog.Errorf("add config to log plugin error, %v, config: %v, pod uid: %s", err, config, podUID)
			return err
		}
		glog.V(7).Infof("pod log plugin config added, pod uid: %s, changed: %t, hash: %s", podUID, rsp.Changed, rsp.Hash)
	}

	return nil
}

func (m *ManagerImpl) deletePluginConfigs(podUID k8stypes.UID, endpoint pluginEndpoint) error {
	for configName := range m.pluginStatusManager.getLogConfigNames(podUID) {
		// invoke log plugin api to delete config
		glog.Infof("calling log plugin to delete config for pod, config name: %s, pod uid: %s", configName, podUID)
		rsp, err := endpoint.delConfig(configName)
		if err != nil {
			glog.Errorf("delete config to log plugin error, %v, config name: %s, pod uid: %s", err, configName, podUID)
			return err
		}
		glog.V(7).Infof("pod log plugin config deleted, config name: %s, pod uid: %s, changed: %t", configName, podUID, rsp.Changed)
	}
	return nil
}

func (m *ManagerImpl) refreshPodState(pod *v1.Pod) error {
	glog.V(7).Infof("refresh pod state, pod: %q", format.Pod(pod))
	podLogPolicy, err := policy.GetPodLogPolicy(pod)
	if err != nil {
		glog.Errorf("get pod log policy error, %v, pod: %q", err, format.Pod(pod))
		return err
	}

	// create pod log volumes map
	podLogVolumes, err := m.buildPodLogVolumes(pod, podLogPolicy)
	if err != nil {
		glog.Errorf("build pod log volumes error, %v, pod: %q", err, format.Pod(pod))
		return err
	}

	podConfigMapKeys, err := m.buildPodLogConfigMapKeys(pod, podLogPolicy)
	if err != nil {
		glog.Errorf("build pod log configmap keys error, %v, pod: %q", err, format.Pod(pod))
		return err
	}

	// if finished status not found, update from log plugin
	//_, exists := m.policyStatusManager.GetCollectFinishedStatus(pod.UID)
	//if !exists {
	//	isFinished, _ := m.isCollectFinished(pod, podLogPolicy)
	//	m.policyStatusManager.UpdateCollectFinishedStatus(pod.UID, isFinished)
	//}
	isFinished, _ := m.isCollectFinished(pod, podLogPolicy)
	m.policyStatusManager.UpdateCollectFinishedStatus(pod.UID, isFinished)

	m.policyStatusManager.UpdateConfigMapKeys(pod.UID, podConfigMapKeys)
	// add log volumes to podLogPolicyManager
	m.policyStatusManager.UpdateLogVolumes(pod.UID, podLogVolumes)
	// add log policies to podLogPolicyManager
	m.policyStatusManager.UpdateLogPolicy(pod.UID, podLogPolicy)
	return nil
}

func (m *ManagerImpl) removePodState(podUID k8stypes.UID) {
	m.policyStatusManager.RemoveCollectFinishedStatus(podUID)
	m.policyStatusManager.RemoveConfigMapKeys(podUID)
	// remove log volumes from podLogPolicyManager
	m.policyStatusManager.RemoveLogVolumes(podUID)
	// remove log policy from podLogPolicyManager
	m.policyStatusManager.RemoveLogPolicy(podUID)
}

func (m *ManagerImpl) pushConfigs(pod *v1.Pod) error {
	glog.V(7).Infof("push pod configs, pod: %q", format.Pod(pod))
	logPolicy, exists := m.policyStatusManager.GetLogPolicy(pod.UID)
	if !exists {
		err := fmt.Errorf("log policy not found in state manager, pod: %q", format.Pod(pod))
		glog.Error(err)
		return err
	}

	logVolumes, exists := m.policyStatusManager.GetLogVolumes(pod.UID)
	if !exists {
		err := fmt.Errorf("log volumes not found in state manager, pod: %q", format.Pod(pod))
		glog.Error(err)
		return err
	}

	podLogConfigs, err := m.buildPodLogConfigs(pod, logPolicy, logVolumes)
	if err != nil {
		glog.Errorf("build pod log configs error, %v, pod: %q", err, format.Pod(pod))
		return err
	}

	endpoint, err := m.getLogPluginEndpoint(logPolicy.LogPlugin)
	if err != nil {
		glog.Errorf("get log plugin endpoint error, %v, log plugin name: %s, pod: %q", err, logPolicy.LogPlugin, format.Pod(pod))
		return err
	}

	err = m.pushPluginConfigs(pod.UID, endpoint, podLogConfigs)
	if err != nil {
		glog.Errorf("update pod log configs error, %v, log plugin name: %s, pod: %q", err, endpoint.name(), format.Pod(pod))
		return err
	}

	return nil
}

func (m *ManagerImpl) CreateLogPolicy(pod *v1.Pod) error {
	// ignore pod without log policy
	if !policy.IsPodLogPolicyExists(pod) {
		return nil
	}

	err := m.refreshPodState(pod)
	if err != nil {
		glog.Errorf("refresh pod log policy error, %v, pod: %q", err, format.Pod(pod))
		return err
	}

	// ignore check because it must be exists
	logVolumes, _ := m.policyStatusManager.GetLogVolumes(pod.UID)
	// create log symbol link for pod
	err = m.createPodLogDirSymLink(logVolumes)
	if err != nil {
		glog.Errorf("create pod log symbol link error, %v, pod: %q", err, format.Pod(pod))
		return err
	}

	err = m.pushConfigs(pod)
	if err != nil {
		glog.Errorf("push pod log configs error, %v, pod: %q", err, format.Pod(pod))
		return err
	}

	glog.Infof("create log policy success, pod: %q", format.Pod(pod))
	m.recorder.Eventf(pod, v1.EventTypeNormal, policy.LogPolicyCreateSuccess, "create log policy success")

	return nil
}

func (m *ManagerImpl) exceedTerminationGracePeriod(pod *v1.Pod) bool {
	// check TerminationGracePeriodSeconds
	if pod.DeletionTimestamp != nil && pod.DeletionGracePeriodSeconds != nil {
		now := time.Now()
		deletionTime := pod.DeletionTimestamp.Time
		gracePeriod := time.Duration(*pod.DeletionGracePeriodSeconds) * time.Second
		if now.After(deletionTime.Add(gracePeriod)) {
			return true
		}
		return false
	}
	return false
}

func (m *ManagerImpl) RemoveLogPolicy(pod *v1.Pod) error {
	if pod == nil {
		// pod is deleted, but containers may be running
		return nil
	}
	// ignore pod without log policy
	if !policy.IsPodLogPolicyExists(pod) {
		return nil
	}

	// get log policy from podLogPolicyManager
	podLogPolicy, exists := m.policyStatusManager.GetLogPolicy(pod.UID)
	if !exists {
		glog.Warningf("pod log policy not found, pod: %q", format.Pod(pod))
		return nil
	}

	isFinished, message := m.isCollectFinished(pod, podLogPolicy)
	m.policyStatusManager.UpdateCollectFinishedStatus(pod.UID, isFinished)
	if !isFinished {
		glog.Errorf(message)
		return fmt.Errorf(message)
	}

	endpoint, err := m.getLogPluginEndpoint(podLogPolicy.LogPlugin)
	if err != nil {
		glog.Errorf("get log plugin endpoint error, %v, log plugin name: %s, pod: %q", err, podLogPolicy.LogPlugin, format.Pod(pod))
		return err
	}

	err = m.deletePluginConfigs(pod.UID, endpoint)
	if err != nil {
		glog.Errorf("update pod log configs error, %v, pod: %q, log plugin name: %s", err, format.Pod(pod), endpoint.name())
		return err
	}

	m.removePodState(pod.UID)

	glog.Infof("remove log policy success, pod: %q", format.Pod(pod))
	m.recorder.Eventf(pod, v1.EventTypeNormal, policy.LogPolicyRemoveSuccess, "remove log policy success")

	return nil
}

func (m *ManagerImpl) isCollectFinished(pod *v1.Pod, podLogPolicy *policy.PodLogPolicy) (bool, string) {
	collectFinished := m.getPluginCollectState(pod.UID, podLogPolicy)
	if !collectFinished {
		if podLogPolicy.SafeDeletionEnabled {
			message := fmt.Sprintf("log config state is running and safe_deletion_enable is true, cannot remove log policy, pod: %q", format.Pod(pod))
			return false, message
		}
		if !m.exceedTerminationGracePeriod(pod) {
			message := fmt.Sprintf("log config state is running, cannot remove log policy after grace period seconds, pod: %q", format.Pod(pod))
			return false, message
		}
	}
	return true, ""
}

func (m *ManagerImpl) getPluginCollectState(podUID k8stypes.UID, podLogPolicy *policy.PodLogPolicy) bool {
	configNames := m.pluginStatusManager.getLogConfigNames(podUID)
	if len(configNames) == 0 {
		glog.V(7).Infof("no config found by pod uid: %s", podUID)
		return true
	}
	endpoint, err := m.getLogPluginEndpoint(podLogPolicy.LogPlugin)
	if err != nil {
		glog.Errorf("get log plugin endpoint error, %v, log plugin name: %s, pod uid: %s", err, podLogPolicy.LogPlugin, podUID)
		return false
	}
	for configName := range configNames {
		glog.V(7).Infof("calling log plugin to get state for pod, pod uid: %s, config name: %s", podUID, configName)
		rsp, err := endpoint.getState(configName)
		if err != nil {
			glog.Errorf("get state error, %v, config name: %s, pod uid: %s", err, configName, podUID)
			return false
		}
		glog.V(7).Infof("get state %q from plugin, config name: %s, pod uid: %s", rsp.State, configName, podUID)
		if rsp.State == pluginapi.State_Running {
			return false
		}
	}
	glog.V(7).Infof("all plugin configs of pod are collected, pod uid: %s, config names: %v", podUID, configNames)
	return true
}

func (m *ManagerImpl) getLogPluginEndpoint(logPluginName string) (pluginEndpoint, error) {
	ep, exists := m.logPlugins[logPluginName]
	if !exists {
		return nil, fmt.Errorf("invalid endpoint %s", logPluginName)
	}
	return ep, nil
}

func (m *ManagerImpl) onConfigMapUpdate(configMap *v1.ConfigMap) {
	configMapKey := buildConfigMapKey(configMap.Namespace, configMap.Name)
	glog.Infof("configMap %q updated", configMapKey)

	// TODO: use work queue
	podUIDs := m.policyStatusManager.GetPodUIDsByConfigMapKey(configMapKey)
	for podUID := range podUIDs {
		pod, exists := m.podManager.GetPodByUID(k8stypes.UID(podUID))
		if !exists {
			glog.Warningf("pod not found in podManager, pod uid: %s", podUID)
			continue
		}

		podLogPolicy, exists := m.policyStatusManager.GetLogPolicy(pod.UID)
		if !exists {
			glog.Warningf("pod log policy not found, pod: %q", format.Pod(pod))
			continue
		}

		podConfigMapKeys, err := m.buildPodLogConfigMapKeys(pod, podLogPolicy)
		if err != nil {
			glog.Errorf("build pod log configmap key error, %v, pod: %q", err, format.Pod(pod))
			m.recorder.Eventf(pod, v1.EventTypeWarning, policy.LogPolicyConfigUpdateFailedReason, "build pod log configmap keys error, %v", err)
			continue
		}
		m.policyStatusManager.UpdateConfigMapKeys(pod.UID, podConfigMapKeys)

		err = m.pushConfigs(pod)
		if err != nil {
			glog.Errorf("push configs error, %v, pod: %q", err, format.Pod(pod))
			m.recorder.Eventf(pod, v1.EventTypeWarning, policy.LogPolicyConfigUpdateFailedReason, "push configs to log plugin error, %v", err)
			continue
		}
	}
}

func (m *ManagerImpl) IsCollectFinished(podUID k8stypes.UID) bool {
	return m.policyStatusManager.IsCollectFinished(podUID)
}
