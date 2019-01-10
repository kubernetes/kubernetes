package policy

import (
	"k8s.io/kubernetes/pkg/kubelet/log/api"
	"sync"

	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
)

type LogStatusProvider interface {
	IsCollectFinished(podUID k8stypes.UID) bool
}

type LogStatusManager interface {
	LogStatusProvider
	UpdateCollectFinishedStatus(podUID k8stypes.UID, isFinished bool)
	RemoveCollectFinishedStatus(podUID k8stypes.UID)
	GetCollectFinishedStatus(podUID k8stypes.UID) (bool, bool)

	UpdateConfigMapKeys(podUID k8stypes.UID, configMapKeys sets.String)
	RemoveConfigMapKeys(podUID k8stypes.UID)
	GetAllConfigMapKeys() sets.String
	GetPodUIDsByConfigMapKey(configMapKey string) sets.String

	UpdateLogPolicy(podUID k8stypes.UID, podLogPolicy *api.PodLogPolicy)
	RemoveLogPolicy(podUID k8stypes.UID)
	GetLogPolicy(podUID k8stypes.UID) (*api.PodLogPolicy, bool)

	UpdateLogVolumes(podUID k8stypes.UID, logVolumes LogVolumesMap)
	RemoveLogVolumes(podUID k8stypes.UID)
	GetLogVolumes(podUID k8stypes.UID) (LogVolumesMap, bool)
}

type LogVolume struct {
	VolumeName string
	// path in container
	// eg. /var/log/<category>
	Path string
	// real mount path in host
	// eg. /var/lib/kubelet/pods/<pod-uid>/volumes/kubernetes.io~<volume-type>/<volume-name>
	HostPath string
	// pod logs symlink path
	// eg. /var/log/pods/<pod-uid>/<container-name>/<category>
	LogDirPath string
}

// volumeName -> logVolume
type LogVolumesMap map[string]*LogVolume

type policyStatusManager struct {
	mutex sync.RWMutex
	// pod uid -> podLogPolicy
	// desired state
	podLogPolicies map[k8stypes.UID]*api.PodLogPolicy
	// pod uid -> podLogVolume
	// desired state
	podLogVolumes map[k8stypes.UID]LogVolumesMap
	// pod uid -> configmap key set
	// update from pod policy
	// desired state
	podConfigMaps map[k8stypes.UID]sets.String
	// configmap key -> pod uid set
	// configmap key := <namespace>/<name>
	// update from pod policy
	// desired state
	configMapPodUIDs map[string]sets.String
	// pod uid -> is finished
	podLogCollectFinishedStatus map[k8stypes.UID]bool
}

func NewPolicyStatusManager() *policyStatusManager {
	return &policyStatusManager{
		podLogPolicies:              make(map[k8stypes.UID]*api.PodLogPolicy),
		podLogVolumes:               make(map[k8stypes.UID]LogVolumesMap),
		podConfigMaps:               make(map[k8stypes.UID]sets.String),
		configMapPodUIDs:            make(map[string]sets.String),
		podLogCollectFinishedStatus: make(map[k8stypes.UID]bool),
	}
}

func (m *policyStatusManager) UpdateConfigMapKeys(podUID k8stypes.UID, configMapKeys sets.String) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	m.podConfigMaps[podUID] = configMapKeys
	for key := range configMapKeys {
		podUIDs, exists := m.configMapPodUIDs[key]
		if !exists {
			podUIDs = sets.NewString()
			m.configMapPodUIDs[key] = podUIDs
		}
		podUIDs.Insert(string(podUID))
	}
}

func (m *policyStatusManager) RemoveConfigMapKeys(podUID k8stypes.UID) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	configMapKeys, uidExists := m.podConfigMaps[podUID]
	if uidExists {
		for key := range configMapKeys {
			podUIDs, keyExists := m.configMapPodUIDs[key]
			if keyExists {
				podUIDs.Delete(string(podUID))
				if podUIDs.Len() == 0 {
					delete(m.configMapPodUIDs, key)
				}
			}
		}
	}
	delete(m.podConfigMaps, podUID)
}

func (m *policyStatusManager) GetAllConfigMapKeys() sets.String {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	configMapKeys := sets.NewString()
	for key := range m.configMapPodUIDs {
		configMapKeys.Insert(key)
	}
	return configMapKeys
}

func (m *policyStatusManager) GetPodUIDsByConfigMapKey(configMapKey string) sets.String {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	podUIDs, exists := m.configMapPodUIDs[configMapKey]
	if !exists {
		podUIDs = sets.NewString()
	}
	return podUIDs
}

func (m *policyStatusManager) UpdateLogPolicy(podUID k8stypes.UID, podLogPolicy *api.PodLogPolicy) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	m.podLogPolicies[podUID] = podLogPolicy
}

func (m *policyStatusManager) RemoveLogPolicy(podUID k8stypes.UID) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	delete(m.podLogPolicies, podUID)
}

func (m *policyStatusManager) GetLogPolicy(podUID k8stypes.UID) (*api.PodLogPolicy, bool) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	podLogPolicy, exists := m.podLogPolicies[podUID]
	return podLogPolicy, exists
}

func (m *policyStatusManager) UpdateLogVolumes(podUID k8stypes.UID, logVolumes LogVolumesMap) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	m.podLogVolumes[podUID] = logVolumes
}

func (m *policyStatusManager) RemoveLogVolumes(podUID k8stypes.UID) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	delete(m.podLogVolumes, podUID)
}

func (m *policyStatusManager) GetLogVolumes(podUID k8stypes.UID) (LogVolumesMap, bool) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	podLogVolume, exists := m.podLogVolumes[podUID]
	return podLogVolume, exists
}

func (m *policyStatusManager) IsCollectFinished(podUID k8stypes.UID) bool {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	finished, exists := m.podLogCollectFinishedStatus[podUID]
	if !exists {
		return true
	}
	return finished
}

func (m *policyStatusManager) UpdateCollectFinishedStatus(podUID k8stypes.UID, isFinished bool) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	m.podLogCollectFinishedStatus[podUID] = isFinished
}

func (m *policyStatusManager) RemoveCollectFinishedStatus(podUID k8stypes.UID) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	delete(m.podLogCollectFinishedStatus, podUID)
}

func (m *policyStatusManager) GetCollectFinishedStatus(podUID k8stypes.UID) (bool, bool) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	isFinished, exists := m.podLogCollectFinishedStatus[podUID]
	return isFinished, exists
}
