package generic

import (
	coreinformers "k8s.io/client-go/informers/core/v1"
)

func (a *Webhook) SetNamespaceInformer(namespaceInformer coreinformers.NamespaceInformer) {
	a.namespaceMatcher.NamespaceLister = namespaceInformer.Lister()
}

func (a *Webhook) SetHookSource(hookSource Source) {
	a.hookSource = hookSource
}

func (a *Webhook) SetReadyFuncFromKCP(namespaceInformer coreinformers.NamespaceInformer) {
	a.SetReadyFunc(func() bool {
		return namespaceInformer.Informer().HasSynced() && a.hookSource.HasSynced()
	})
}
