package util

import (
	"fmt"
	"github.com/golang/glog"
	"hash/adler32"
	"k8s.io/kubernetes/pkg/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	gpuTypes "k8s.io/kubernetes/pkg/kubelet/gpu/types"
	"k8s.io/kubernetes/pkg/types"
	hashutil "k8s.io/kubernetes/pkg/util/hash"
	"strconv"
	"strings"
)

// HashContainer returns the hash of the container. It is used to get
// gpu usage status
func HashContainerFromLabel(podUID string, containerHashID string) gpuTypes.PodCotainerHashID {
	hash := adler32.New()
	hashString := fmt.Sprintf("%s_%s", podUID, containerHashID)
	hashutil.DeepHashObject(hash, hashString)
	return gpuTypes.PodCotainerHashID(hash.Sum32())
}

// HashContainer returns the hash of the container. It is used to get
// gpu usage status
func HashContainerFromData(podUID types.UID, container *api.Container) gpuTypes.PodCotainerHashID {
	hash := adler32.New()
	hashString := fmt.Sprintf("%s_%s", string(podUID), strconv.FormatUint(kubecontainer.HashContainer(container), 16))
	hashutil.DeepHashObject(hash, hashString)
	return gpuTypes.PodCotainerHashID(hash.Sum32())
}

func GetGPUIndexFromLabel(label string) []int {
	intSlice := []int{}
	stringSlice := strings.Split(label, ",")
	for _, stringValue := range stringSlice {
		intValue, err := strconv.Atoi(stringValue)
		if err != nil {
			glog.Errorf("Failed to convert string(%s) to int. Reason: %s", stringValue, err)
			return []int{}
		}
		intSlice = append(intSlice, intValue)
	}

	return intSlice
}

func GetLabelFromGPUIndex(gpuIndexes []int) string {
	stringSlice := []string{}
	for _, value := range gpuIndexes {
		stringSlice = append(stringSlice, strconv.Itoa(value))
	}

	return strings.Join(stringSlice, ",")
}
