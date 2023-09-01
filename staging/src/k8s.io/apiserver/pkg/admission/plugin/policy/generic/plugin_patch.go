package generic

import (
	coreinformers "k8s.io/client-go/informers/core/v1"
	"k8s.io/client-go/tools/cache"
)

func (c *Plugin[H]) SetNamespaceInformer(i coreinformers.NamespaceInformer) {
	c.namespaceInformer = i
}

func (c *Plugin[H]) SetPolicyInformer(i cache.SharedIndexInformer) {
	c.policyInformer = i
}

func (c *Plugin[H]) SetBindingInformer(i cache.SharedIndexInformer) {
	c.bindingInformer = i
}
