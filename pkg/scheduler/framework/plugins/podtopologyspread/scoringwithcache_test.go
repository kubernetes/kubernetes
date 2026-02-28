package podtopologyspread

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/cacheplugin"
)

func Test_addPod(t *testing.T) {
	initCache := cachedPodsMap{}

	initCache.AddPod("pod1", "ns1", "node1", "value1", 3)

	assert.Equal(t, len(initCache[cacheplugin.NamespaceedNameNode{Namespace: "ns1", ReservedNode: "node1", Name: "pod1"}]), 4)
}
