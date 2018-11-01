package manager

import (
	"sync"

	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/logplugin/v1alpha1"
)

type pluginStatusManager struct {
	mutex sync.RWMutex
	// pod uid -> config name set
	// update from log plugins
	// current state
	podLogConfigNames    map[k8stypes.UID]sets.String
	podLogPluginNames    map[k8stypes.UID]string
	podLogFinishedStatus map[k8stypes.UID]bool
}

// configName -> config
type logConfigsMap map[string]*pluginapi.Config

func newPluginStatusManager() *pluginStatusManager {
	return &pluginStatusManager{
		podLogConfigNames:    make(map[k8stypes.UID]sets.String),
		podLogPluginNames:    make(map[k8stypes.UID]string),
		podLogFinishedStatus: make(map[k8stypes.UID]bool),
	}
}

func (m *pluginStatusManager) getLogConfigNames(podUID k8stypes.UID) sets.String {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	configNames, exists := m.podLogConfigNames[podUID]
	if !exists {
		return sets.NewString()
	}
	return configNames
}

func (m *pluginStatusManager) updateAllLogConfigs(configs []*pluginapi.Config, pluginName string) {
	if configs == nil {
		return
	}
	m.mutex.Lock()
	defer m.mutex.Unlock()
	podLogConfigNames := make(map[k8stypes.UID]sets.String)
	podLogPluginNames := make(map[k8stypes.UID]string)
	for _, config := range configs {
		// update podLogConfigNames map
		configNames, exists := podLogConfigNames[k8stypes.UID(config.Metadata.PodUID)]
		if !exists {
			configNames = sets.NewString()
			podLogConfigNames[k8stypes.UID(config.Metadata.PodUID)] = configNames
		}
		configNames.Insert(config.Metadata.Name)
		// update podLogPluginName map
		_, exists = podLogPluginNames[k8stypes.UID(config.Metadata.PodUID)]
		if !exists {
			podLogPluginNames[k8stypes.UID(config.Metadata.PodUID)] = pluginName
		}
	}
	m.podLogConfigNames = podLogConfigNames
	m.podLogPluginNames = podLogPluginNames
}

func (m *pluginStatusManager) getAllPodUIDs() sets.String {
	podUIDs := sets.NewString()
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	for uid := range m.podLogConfigNames {
		podUIDs.Insert(string(uid))
	}
	return podUIDs
}

func (m *pluginStatusManager) getLogPluginName(podUID k8stypes.UID) (string, bool) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	pluginName, exists := m.podLogPluginNames[podUID]
	return pluginName, exists
}
