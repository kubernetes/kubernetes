package interpodaffinity

import (
	"context"
	"os"
	"reflect"
	"strconv"
	"sync"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	informerv1 "k8s.io/client-go/informers/core/v1"
	listersv1 "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/cacheplugin"
)

type cachedPodsMap map[cacheplugin.NamespaceedNameNode]scoreMap

func (c cachedPodsMap) append(other map[cacheplugin.NamespaceedNameNode]scoreMap) {
	for k, v := range other {
		c[k] = v
	}
}

type ExistingPodAffinityTermDetailedState struct {
	lock sync.RWMutex

	preCalRes scoreMap
	// we store ns/name for all cachedPods
	// key is ns/name, value is the topoValue
	cachedPods cachedPodsMap
	// immutable
	namespace       string
	labels          map[string]string
	namespaceLabels map[string]string
}

type ExisingPodCacheProxy struct {
	impl *cacheplugin.CacheImpl[namespacedLabels, *ExistingPodAffinityTermDetailedState]
}

func (n *ExistingPodAffinityTermDetailedState) RemovePod(key cacheplugin.NamespaceedNameNode) scoreMap {
	tv, found := n.cachedPods[key]
	if !found {
		return tv
	}
	delete(n.cachedPods, key)
	n.preCalRes.append(reverse(tv))
	return tv
}

type namespacedLabels struct {
	labels    map[string]string
	namespace string
}

var cacheSize = 10
var workNum = 4

func init() {
	s := os.Getenv("InterPodAffinityCacheSize")
	n := os.Getenv("InterPodAffinityCacheWorkerNum")
	if s != "" {
		cacheSize, _ = strconv.Atoi(s)
	}
	if n != "" {
		workNum, _ = strconv.Atoi(n)
	}
}

func NewExistingPodCacheProxy(ctx context.Context, args config.InterPodAffinityArgs, poi informerv1.PodInformer, pl listersv1.PodLister, nsl listersv1.NamespaceLister, nsi informerv1.NamespaceInformer, noi informerv1.NodeInformer, snapshot fwk.SharedLister) *ExisingPodCacheProxy {
	c := cacheplugin.NewCache[namespacedLabels, *ExistingPodAffinityTermDetailedState](ctx,
		"InterPodAffinityExistingPod", cacheSize, workNum, func(pat namespacedLabels) string {
			return pat.namespace + "/" + labels.Set(pat.labels).String()
		}, func(key cacheplugin.NamespaceedNameNode, tds *ExistingPodAffinityTermDetailedState, logger logr.Logger) {
			tds.lock.Lock()
			defer tds.lock.Unlock()

			reservedNode := key.ReservedNode
			key.ReservedNode = ""
			// tds.unreservePod(key)
			p, err := cacheplugin.GetPodFromLister(key, pl)
			if err != nil || reservedNode == "" {
				deleted := tds.RemovePod(key)
				logger.V(5).Error(err, "remove pre calculate score in cache", "key", key, "removed score", deleted)
				return
			}
			if p.Spec.Affinity == nil || p.Spec.Affinity.PodAffinity == nil && p.Spec.Affinity.PodAntiAffinity == nil {
				return
			}

			ni, err := snapshot.NodeInfos().Get(reservedNode)
			if err != nil {
				return
			}
			pi, err := framework.NewPodInfo(p)
			if err != nil {
				return
			}
			topoScore := scoreMap{}
			fakePod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: key.Namespace, Labels: tds.labels}}
			// For every hard pod affinity term of <existingPod>, if <pod> matches the term,
			// increment <p.counts> for every node in the cluster with the same <term.TopologyKey>
			// value as that of <existingPod>'s node by the constant <args.hardPodAffinityWeight>
			if args.HardPodAffinityWeight > 0 && len(ni.Node().Labels) != 0 {
				for _, t := range pi.RequiredAffinityTerms {
					topoScore.processTerm(&t, args.HardPodAffinityWeight, fakePod, tds.namespaceLabels, ni.Node(), 1)
				}
			}
			// For every soft pod affinity term of <existingPod>, if <pod> matches the term,
			// increment <p.counts> for every node in the cluster with the same <term.TopologyKey>
			// value as that of <existingPod>'s node by the term's weight.
			topoScore.processTerms(pi.PreferredAffinityTerms, fakePod, tds.namespaceLabels, ni.Node(), 1)
			// For every soft pod anti-affinity term of <existingPod>, if <pod> matches the term,
			// decrement <pm.counts> for every node in the cluster with the same <term.TopologyKey>
			// value as that of <existingPod>'s node by the term's weight.
			topoScore.processTerms(pi.PreferredAntiAffinityTerms, fakePod, tds.namespaceLabels, ni.Node(), -1)
			deleted := tds.RemovePod(key)
			tds.cachedPods[key] = topoScore
			tds.preCalRes.append(topoScore)
			if len(topoScore) > 0 || len(deleted) > 0 {
				logger.V(3).Info("update pre calculate score in cache", "pod", klog.KObj(p), "score", topoScore, "removed score", deleted)
			}
		})
	poi.Informer().AddEventHandler(cache.FilteringResourceEventHandler{
		FilterFunc: func(obj interface{}) bool {
			po := obj.(*corev1.Pod)
			if po == nil || po.Spec.NodeName == "" {
				return false
			}
			return true
		},
		Handler: cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				po := obj.(*corev1.Pod)
				if po == nil {
					return
				}
				c.ProcessUpdatePod(po, po.Spec.NodeName)
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				oldPo := oldObj.(*corev1.Pod)
				newPo := newObj.(*corev1.Pod)
				if oldPo == nil || newPo == nil {
					return
				}
				if !reflect.DeepEqual(oldPo.Labels, newPo.Labels) {
					c.ProcessUpdatePod(newPo, newPo.Spec.NodeName)
				}
			},
			DeleteFunc: func(obj interface{}) {
				var pod *corev1.Pod
				switch t := obj.(type) {
				case *corev1.Pod:
					pod = t
				case cache.DeletedFinalStateUnknown:
					pod, _ = t.Obj.(*corev1.Pod)
				}
				if pod == nil {
					return
				}
				c.ProcessUpdatePod(pod, pod.Spec.NodeName)
			},
		},
	})
	nsi.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		UpdateFunc: func(oldObj, newObj interface{}) {
			oldNo := oldObj.(*corev1.Namespace)
			newNo := newObj.(*corev1.Namespace)
			if oldNo == nil || newNo == nil {
				return
			}
			if !reflect.DeepEqual(oldNo.Labels, newNo.Labels) {
				c.Forget(func(at namespacedLabels, tds *ExistingPodAffinityTermDetailedState) bool {
					return tds.namespace == newNo.Name
				})
			}
		},
	})
	noi.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		UpdateFunc: func(oldObj, newObj interface{}) {
			oldNo := oldObj.(*corev1.Node)
			newNo := newObj.(*corev1.Node)
			if oldNo == nil || newNo == nil {
				return
			}
			if !reflect.DeepEqual(oldNo.Labels, newNo.Labels) {
				c.Process(func(at namespacedLabels, tds *ExistingPodAffinityTermDetailedState) {
					ni, err := snapshot.NodeInfos().Get(newNo.Name)
					if err != nil {
						return
					}
					for _, p := range ni.GetPods() {
						c.ProcessUpdatePod(p.GetPod(), newNo.Name)
					}
				})
			}
		},
	})
	return &ExisingPodCacheProxy{impl: c}
}

type IncomingPodAffinityTermDetailedState struct {
	lock sync.RWMutex

	// we will find the number of pods by namespace and topoValue
	preCalRes scoreMap
	// we store ns/name for all cachedPods
	// key is ns/name, value is the topoValue
	cachedPods cachedPodsMap
	//
	affinity     []fwk.WeightedAffinityTerm
	antiaffinity []fwk.WeightedAffinityTerm
}

type IncomingPodCacheProxy struct {
	impl *cacheplugin.CacheImpl[string, *IncomingPodAffinityTermDetailedState]
}

func (n scoreMap) Reserve(tpk, tpv string, value int64) {
	if _, ok := n[tpk]; !ok {
		n[tpk] = map[string]int64{}
	}
	n[tpk][tpv] += value
}

func reverse(sm scoreMap) scoreMap {
	new := make(scoreMap, len(sm))
	for k, v := range sm {
		vnew := make(map[string]int64, len(v))
		for k1, v1 := range v {
			vnew[k1] = -v1
		}
		new[k] = vnew
	}
	return new
}

func (n *IncomingPodAffinityTermDetailedState) RemovePod(key cacheplugin.NamespaceedNameNode) scoreMap {
	tv, found := n.cachedPods[key]
	if !found {
		return tv
	}
	delete(n.cachedPods, key)
	n.preCalRes.append(reverse(tv))
	return tv
}

func NewIncomingPodCacheProxy(ctx context.Context, pl listersv1.PodLister, poi informerv1.PodInformer, nsi informerv1.NamespaceInformer, noi informerv1.NodeInformer, snapshot fwk.SharedLister) *IncomingPodCacheProxy {
	c := cacheplugin.NewCache[string, *IncomingPodAffinityTermDetailedState](ctx,
		"InterPodAffinity", cacheSize, workNum, func(pat string) string {
			return pat
		}, func(key cacheplugin.NamespaceedNameNode, tds *IncomingPodAffinityTermDetailedState, logger logr.Logger) {
			reservedNode := key.ReservedNode
			key.ReservedNode = ""
			// p, err := pl.Pods(key.Namespace).Get(key.Name)
			p, err := cacheplugin.GetPodFromLister(key, pl)
			tds.lock.Lock()
			defer tds.lock.Unlock()
			if err != nil || reservedNode == "" {
				deleted := tds.RemovePod(key)
				if len(deleted) > 0 {
					logger.V(5).Info("remove pre calculate score in cache", "pod", klog.KObj(p), "removed score", deleted)
				}
				return
			}
			ni, err := snapshot.NodeInfos().Get(reservedNode)
			if err != nil {
				return
			}
			deleted := tds.RemovePod(key)
			topoScore := scoreMap{}
			topoScore.processTerms(tds.affinity, p, nil, ni.Node(), 1)
			topoScore.processTerms(tds.antiaffinity, p, nil, ni.Node(), -1)
			tds.cachedPods[key] = topoScore
			tds.preCalRes.append(topoScore)
			if len(topoScore) > 0 || len(deleted) > 0 {
				logger.V(3).Info("update pre calculate score in cache", "pod", klog.KObj(p), "score", topoScore, "removed score", deleted)
			}
		})
	poi.Informer().AddEventHandler(cache.FilteringResourceEventHandler{
		FilterFunc: func(obj interface{}) bool {
			po := obj.(*corev1.Pod)
			if po == nil || po.Spec.NodeName == "" {
				return false
			}
			return true
		},
		Handler: cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				po := obj.(*corev1.Pod)
				if po == nil {
					return
				}
				c.ProcessUpdatePod(po, po.Spec.NodeName)
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				oldPo := oldObj.(*corev1.Pod)
				newPo := newObj.(*corev1.Pod)
				if oldPo == nil || newPo == nil {
					return
				}
				if !reflect.DeepEqual(oldPo.Labels, newPo.Labels) {
					c.ProcessUpdatePod(newPo, newPo.Spec.NodeName)
				}
			},
			DeleteFunc: func(obj interface{}) {
				var pod *corev1.Pod
				switch t := obj.(type) {
				case *corev1.Pod:
					pod = t
				case cache.DeletedFinalStateUnknown:
					pod, _ = t.Obj.(*corev1.Pod)
				}
				if pod == nil {
					return
				}
				c.ProcessUpdatePod(pod, pod.Spec.NodeName)
			},
		},
	})

	noi.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		UpdateFunc: func(oldObj, newObj interface{}) {
			oldNo := oldObj.(*corev1.Node)
			newNo := newObj.(*corev1.Node)
			if oldNo == nil || newNo == nil {
				return
			}
			if !reflect.DeepEqual(oldNo.Labels, newNo.Labels) {
				c.Process(func(at string, tds *IncomingPodAffinityTermDetailedState) {
					ni, err := snapshot.NodeInfos().Get(newNo.Name)
					if err != nil {
						return
					}
					for _, p := range ni.GetPods() {
						c.ProcessUpdatePod(p.GetPod(), newNo.Name)
					}
				})
			}
			// TODO: delete node from nodeToTopoValue
		},
	})

	nsi.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		UpdateFunc: func(oldObj, newObj interface{}) {
			oldNo := oldObj.(*corev1.Namespace)
			newNo := newObj.(*corev1.Namespace)
			if oldNo == nil || newNo == nil {
				return
			}
			// we simply forget all caches here.
			// TODO: find better way to only forget some of the cache
			if !reflect.DeepEqual(oldNo.Labels, newNo.Labels) {
				c.Forget(func(at string, tds *IncomingPodAffinityTermDetailedState) bool {
					return true
				})
			}
		},
	})
	return &IncomingPodCacheProxy{impl: c}
}
