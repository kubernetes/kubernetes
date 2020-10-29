package daemon

import (
	"context"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	appsinformers "k8s.io/client-go/informers/apps/v1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/klog/v2"

	projectv1 "github.com/openshift/api/project/v1"
)

func NewNodeSelectorAwareDaemonSetsController(ctx context.Context, openshiftDefaultNodeSelectorString, kubeDefaultNodeSelectorString string, namepaceInformer coreinformers.NamespaceInformer, daemonSetInformer appsinformers.DaemonSetInformer, historyInformer appsinformers.ControllerRevisionInformer, podInformer coreinformers.PodInformer, nodeInformer coreinformers.NodeInformer, kubeClient clientset.Interface, failedPodsBackoff *flowcontrol.Backoff) (*DaemonSetsController, error) {
	controller, err := NewDaemonSetsController(ctx, daemonSetInformer, historyInformer, podInformer, nodeInformer, kubeClient, failedPodsBackoff)

	if err != nil {
		return controller, err
	}
	controller.namespaceLister = namepaceInformer.Lister()
	controller.namespaceStoreSynced = namepaceInformer.Informer().HasSynced
	controller.openshiftDefaultNodeSelectorString = openshiftDefaultNodeSelectorString
	if len(controller.openshiftDefaultNodeSelectorString) > 0 {
		controller.openshiftDefaultNodeSelector, err = labels.Parse(controller.openshiftDefaultNodeSelectorString)
		if err != nil {
			return nil, err
		}
	}
	controller.kubeDefaultNodeSelectorString = kubeDefaultNodeSelectorString
	if len(controller.kubeDefaultNodeSelectorString) > 0 {
		controller.kubeDefaultNodeSelector, err = labels.Parse(controller.kubeDefaultNodeSelectorString)
		if err != nil {
			return nil, err
		}
	}

	return controller, nil
}

func (dsc *DaemonSetsController) nodeShouldRunDaemonPod(logger klog.Logger, node *v1.Node, ds *appsv1.DaemonSet) (bool, bool) {
	shouldRun, shouldContinueRunning := NodeShouldRunDaemonPod(logger, node, ds)
	if shouldRun && shouldContinueRunning {
		if matches, matchErr := dsc.namespaceNodeSelectorMatches(node, ds); !matches || matchErr != nil {
			return false, false
		}
	}

	return shouldRun, shouldContinueRunning
}

func (dsc *DaemonSetsController) namespaceNodeSelectorMatches(node *v1.Node, ds *appsv1.DaemonSet) (bool, error) {
	if dsc.namespaceLister == nil {
		return true, nil
	}

	// this is racy (different listers) and we get to choose which way to fail.  This should requeue.
	ns, err := dsc.namespaceLister.Get(ds.Namespace)
	if apierrors.IsNotFound(err) {
		return false, err
	}
	// if we had any error, default to the safe option of creating a pod for the node.
	if err != nil {
		utilruntime.HandleError(err)
		return true, nil
	}

	return dsc.nodeSelectorMatches(node, ns), nil
}

func (dsc *DaemonSetsController) nodeSelectorMatches(node *v1.Node, ns *v1.Namespace) bool {
	kubeNodeSelector, ok := ns.Annotations["scheduler.alpha.kubernetes.io/node-selector"]
	if !ok {
		originNodeSelector, ok := ns.Annotations[projectv1.ProjectNodeSelector]
		switch {
		case ok:
			selector, err := labels.Parse(originNodeSelector)
			if err == nil {
				if !selector.Matches(labels.Set(node.Labels)) {
					return false
				}
			}
		case !ok && len(dsc.openshiftDefaultNodeSelectorString) > 0:
			if !dsc.openshiftDefaultNodeSelector.Matches(labels.Set(node.Labels)) {
				return false
			}
		}
	}

	switch {
	case ok:
		selector, err := labels.Parse(kubeNodeSelector)
		if err == nil {
			if !selector.Matches(labels.Set(node.Labels)) {
				return false
			}
		}
	case !ok && len(dsc.kubeDefaultNodeSelectorString) > 0:
		if !dsc.kubeDefaultNodeSelector.Matches(labels.Set(node.Labels)) {
			return false
		}
	}

	return true
}
