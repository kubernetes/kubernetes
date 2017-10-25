package node

import (
	"encoding/json"
	"fmt"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"

	nodeconfigv1alpha1 "k8s.io/kubernetes/pkg/controller/node/nodeconfig/v1alpha1"

	"github.com/golang/glog"
)

func CreateAddNodeConfigSourcePoolHandler(f func(pool *nodeconfigv1alpha1.NodeConfigSourcePool) error) func(obj interface{}) {
	return func(obj interface{}) {
		pool := obj.(*nodeconfigv1alpha1.NodeConfigSourcePool).DeepCopy()
		if err := f(pool); err != nil {
			utilruntime.HandleError(fmt.Errorf("Error while processing NodeConfigSourcePool Add: %v", err))
		}
	}
}

func CreateUpdateNodeConfigSourcePoolHandler(f func(oldPool, newPool *nodeconfigv1alpha1.NodeConfigSourcePool) error) func(oldObj, newObj interface{}) {
	return func(oldObj, newObj interface{}) {
		oldPool := oldObj.(*nodeconfigv1alpha1.NodeConfigSourcePool).DeepCopy()
		newPool := newObj.(*nodeconfigv1alpha1.NodeConfigSourcePool).DeepCopy()
		if err := f(oldPool, newPool); err != nil {
			utilruntime.HandleError(fmt.Errorf("Error while processing NodeConfigSourcePool Update: %v", err))
		}
	}
}

func (nc *Controller) needsNodeConfigSourcePoolSync() {
	select {
	case nc.pendingNodeConfigSourcePoolSync <- true:
	default:
	}
}

// checks if work needs to be done to sync configurations, does work if necessary
func (nc *Controller) syncNodeConfigSourcePools() {
	// glog.Errorf("will sync NodeConfigSourcePools if necessary")
	// select {
	// case <-nc.pendingNodeConfigSourcePoolSync:
	// default:
	// 	// no work to be done, return
	// 	return
	// }

	glog.Infof("will definitely sync NodeConfigSourcePools")

	// if the sync fails, we want to retry
	var syncerr error
	defer func() {
		if syncerr != nil {
			// TODO(mtaufen): Use utilruntime.HandleError for errors
			glog.Errorf(syncerr.Error())
			nc.needsNodeConfigSourcePoolSync()
		}
	}()

	// get the nodeConfigSourcePools from the local cache
	pools, err := nc.nodeConfigSourcePoolLister.List(labels.Everything())
	if err != nil {
		syncerr = err
		return
	}

	glog.Infof("NodeConfigSourcePools: %#v", pools)

	// pools is a list of pointers, so this doesn't copy the objects
	for _, pool := range pools {
		// validate percentNew, skip this pool if invalid
		// TODO(mtaufen): add this validation to the custom resource as well
		if pool.Spec.PercentNew < 0 || pool.Spec.PercentNew > 100 {
			utilruntime.HandleError(fmt.Errorf("invalid PercentNew for NodeConfigSourcePool %s/%s, must be between 0 and 100", pool.Namespace, pool.Name))
			continue
		}

		var oldSource, newSource *v1.NodeConfigSource
		// note this strategy will force default config on old-configured nodes for an empty or 1 element history
		n := len(pool.Spec.History)
		if n == 1 {
			newSource = pool.Spec.History[0]
		} else if n > 1 {
			newSource = pool.Spec.History[n-1]
			oldSource = pool.Spec.History[n-2]
		}

		newSourceNodes := []*v1.Node{}
		oldSourceNodes := []*v1.Node{}

		glog.Infof("length of pool history is %d", len(pool.Spec.History))

		// select the nodes in the pool from the local cache
		if pool.Spec.LabelSelector == nil {
			glog.Infof("pool %s has nil label selector", pool.Name)
		} else {
			glog.Infof("pool %s has label selector %#v", pool.Name, *pool.Spec.LabelSelector)
		}
		nodeSelector, err := metav1.LabelSelectorAsSelector(pool.Spec.LabelSelector)
		if err != nil {
			syncerr = err
			return
		}

		// TODO(mtaufen): Do I need to make copies of the nodes before modifying them? I probably shouldn't be modifying stuff directly in the store
		glog.Infof("listing nodes in pool %s with selector %#v", pool.Name, nodeSelector)
		nodes, err := nc.nodeLister.List(nodeSelector)
		if err != nil {
			syncerr = err
			return
		} else if len(nodes) == 0 {
			glog.Infof("no nodes in NodeConfigSourcePool %s", pool.Name)
			continue
		}

		// nodes is a list of pointers, so this doesn't copy the objects
		for _, node := range nodes {
			if ConfigSourceEq(newSource, node.Spec.ConfigSource) {
				newSourceNodes = append(newSourceNodes, node)
				continue
			}
			// if not on the new source, assume on an old source
			// all will shortly be forced to conform
			oldSourceNodes = append(oldSourceNodes, node)
		}

		glog.Infof("old source: %#v", oldSource)
		glog.Infof("new source: %#v", newSource)
		glog.Infof("desired percent: %d", pool.Spec.PercentNew)
		glog.Infof("current old source nodes: %#v", oldSourceNodes)
		glog.Infof("current new source nodes: %#v", newSourceNodes)

		// TODO(mtaufen): double check math
		// TODO(mtaufen): unit test the hell out of this

		// If you have a cluster with a small number of nodes, you'll obviously need to be careful to set a % that has a real effect
		// e.g. with 3 nodes, 34% will result in 1 node on the new config source

		totalNodes := len(oldSourceNodes) + len(newSourceNodes)

		targetNewSourceNodes := (pool.Spec.PercentNew * totalNodes) / 100

		if len(newSourceNodes) < targetNewSourceNodes {
			// move nodes from old to new
			n := targetNewSourceNodes - len(newSourceNodes)
			newSourceNodes = append(newSourceNodes, oldSourceNodes[:n]...)
			oldSourceNodes = oldSourceNodes[n:]
		} else if len(newSourceNodes) > targetNewSourceNodes {
			// move nodes from new to old
			n := len(newSourceNodes) - targetNewSourceNodes
			oldSourceNodes = append(oldSourceNodes, newSourceNodes[:n]...)
			newSourceNodes = newSourceNodes[n:]
		}

		glog.Infof("new old source nodes: %#v", oldSourceNodes)
		glog.Infof("new new source nodes: %#v", newSourceNodes)

		// map of node names to patches
		patches := map[string][]byte{}

		// generate all of the necessary patches
		for _, node := range oldSourceNodes {
			if !ConfigSourceEq(oldSource, node.Spec.ConfigSource) {
				patch := map[string]interface{}{
					// TOOD(mtaufen): copied this from adapter.go, is there a reason all this metadata is necessary?
					"apiVersion": node.APIVersion,
					"kind":       node.Kind,
					"metadata":   map[string]interface{}{"name": node.Name},
					"spec":       map[string]interface{}{"configSource": oldSource},
				}
				bytes, err := json.Marshal(patch)
				if err != nil {
					// if there is an error, just log it and skip to the next node
					utilruntime.HandleError(fmt.Errorf("could not generate ConfigSource patch for node %s in pool %s source %#v, error: %v", node.Name, pool.Name, oldSource, err))
					continue
				}
				patches[node.Name] = bytes
			}
		}
		for _, node := range newSourceNodes {
			if !ConfigSourceEq(newSource, node.Spec.ConfigSource) {
				patch := map[string]interface{}{
					// TOOD(mtaufen): copied this from adapter.go, is there a reason all this metadata is necessary?
					"apiVersion": node.APIVersion,
					"kind":       node.Kind,
					"metadata":   map[string]interface{}{"name": node.Name},
					"spec":       map[string]interface{}{"configSource": newSource},
				}
				bytes, err := json.Marshal(patch)
				if err != nil {
					// if there is an error, just log it and skip to the next node
					utilruntime.HandleError(fmt.Errorf("could not generate ConfigSource patch for node %s in pool %s source %#v, error: %v", node.Name, pool.Name, newSource, err))
					continue
				}
				patches[node.Name] = bytes
			}
		}

		// patch the nodes
		for name, patch := range patches {
			glog.Infof("will apply patch to %s: %s", name, string(patch))
			if _, err := nc.kubeClient.CoreV1().Nodes().Patch(name, types.StrategicMergePatchType, patch); err != nil {
				utilruntime.HandleError(err)
			}
		}
	}
}

// TODO(mtaufen): should put these eq helpers in a centralized place, at present just copypasta from the Kubelet

// ConfigSourceEq returns true if the two config sources are semantically equivalent in the context of dynamic config
func ConfigSourceEq(a, b *v1.NodeConfigSource) bool {
	if a == b {
		return true
	} else if a == nil || b == nil {
		// not equal, and one is nil
		return false
	}
	// check equality of config source subifelds
	if a.ConfigMapRef != b.ConfigMapRef {
		return ObjectRefEq(a.ConfigMapRef, b.ConfigMapRef)
	}
	// all internal subfields of the config soruce are equal
	return true
}

// ObjectRefEq returns true if the two object references are semantically equivalent in the context of dynamic config
func ObjectRefEq(a, b *v1.ObjectReference) bool {
	if a == b {
		return true
	} else if a == nil || b == nil {
		// not equal, and one is nil
		return false
	}
	return a.UID == b.UID && a.Namespace == b.Namespace && a.Name == b.Name
}
