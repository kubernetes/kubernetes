package generic

import (
	"github.com/kcp-dev/logicalcluster/v3"
	"k8s.io/client-go/informers"
	coreinformers "k8s.io/client-go/informers/core/v1"
)

func (c *Plugin[H]) SetNamespaceInformer(i coreinformers.NamespaceInformer) {
	c.namespaceInformer = i
}

func (c *Plugin[H]) SetInformerFactory(f informers.SharedInformerFactory) {
	c.informerFactory = f
}

func (c *Plugin[H]) SetSourceFactory(s sourceFactory[H]) {
	c.sourceFactory = s
}

func (c *Plugin[H]) SetClusterName(clusterName logicalcluster.Name) {
	c.clusterName = clusterName
}
