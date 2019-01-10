package manager

import (
	"fmt"
	"path/filepath"

	k8stypes "k8s.io/apimachinery/pkg/types"
)

const (
	// The root directory for pod logs
	podLogsRootDirectory = "/var/log/pods"
)

func buildConfigMapKey(namespace, name string) string {
	return fmt.Sprintf("%s/%s", namespace, name)
}

// buildPodLogsDirectory builds absolute log directory path for a pod.
func buildPodLogsDirectory(podUID k8stypes.UID) string {
	return filepath.Join(podLogsRootDirectory, string(podUID))
}

func buildLogPolicyDirectory(podUID k8stypes.UID, containerName string, category string) string {
	return filepath.Join(buildPodLogsDirectory(podUID), containerName, category)
}

func buildLogConfigName(podUID k8stypes.UID, containerName string, category string, filename string) string {
	// <pod-uid>/<container_name>/<category>/<filename>
	return fmt.Sprintf("%s/%s/%s/%s", podUID, containerName, category, filename)
}
