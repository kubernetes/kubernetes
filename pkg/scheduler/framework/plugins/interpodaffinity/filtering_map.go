package interpodaffinity

import (
	"context"
	"fmt"
	"reflect"
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
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/cacheplugin"
)

func NewFilteringExistingPodAffinityTermDetailedState(res topologyToMatchedTermCount, mpByPod map[cacheplugin.NamespaceedNameNode]topologyToMatchedTermCount,
	ns string, labels, nsLabels map[string]string) *FilteringExistingPodAffinityTermDetailedState {
	affinityCount := res.clone()
	labelCp := make(map[string]string, len(labels))
	for k, v := range labels {
		labelCp[k] = v
	}
	return &FilteringExistingPodAffinityTermDetailedState{
		preCalRes:       affinityCount,
		cachedPods:      mpByPod,
		namespace:       ns,
		labels:          labelCp,
		namespaceLabels: nsLabels,
	}
}

func cachePodsMapString(mp map[cacheplugin.NamespaceedNameNode]topologyToMatchedTermCount) string {
	ret := map[string]string{}
	for k, v := range mp {
		ret[fmt.Sprintf("%v", k)] = v.String()
	}
	return fmt.Sprintf("%v", ret)
}

type FilteringExistingPodAffinityTermDetailedState struct {
	lock sync.RWMutex

	preCalRes topologyToMatchedTermCount
	// we store ns/name for all cachedPods
	// key is ns/name, value is the topoValue
	cachedPods map[cacheplugin.NamespaceedNameNode]topologyToMatchedTermCount
	// immutable
	namespace       string
	labels          map[string]string
	namespaceLabels map[string]string
}

type FilteringExisingPodCacheProxy struct {
	impl *cacheplugin.CacheImpl[namespacedLabels, *FilteringExistingPodAffinityTermDetailedState]
}

func (n *FilteringExistingPodAffinityTermDetailedState) RemovePod(key cacheplugin.NamespaceedNameNode) topologyToMatchedTermCount {
	tv, found := n.cachedPods[key]
	if !found {
		return tv
	}
	delete(n.cachedPods, key)
	for k := range tv {
		if n.preCalRes[k] > 0 {
			n.preCalRes[k]--
		}
	}
	return tv
}

func NewFilteringExistingPodCacheProxy(ctx context.Context, poi informerv1.PodInformer, pl listersv1.PodLister, nsl listersv1.NamespaceLister, nsi informerv1.NamespaceInformer, noi informerv1.NodeInformer, snapshot fwk.SharedLister) *FilteringExisingPodCacheProxy {
	c := cacheplugin.NewCache[namespacedLabels, *FilteringExistingPodAffinityTermDetailedState](ctx,
		"FilteringInterPodAffinityExistingPod", cacheSize, workNum, func(pat namespacedLabels) string {
			return pat.namespace + "/" + labels.Set(pat.labels).String()
		}, func(key cacheplugin.NamespaceedNameNode, tds *FilteringExistingPodAffinityTermDetailedState, logger logr.Logger) {
			tds.lock.Lock()
			defer tds.lock.Unlock()
			reservedNode := key.ReservedNode
			key.ReservedNode = ""
			// p, err := pl.Pods(key.Namespace).Get(key.Name)
			p, err := cacheplugin.GetPodFromLister(key, pl)
			if err != nil || reservedNode == "" {
				deleted := tds.RemovePod(key)
				if len(deleted) > 0 {
					logger.V(5).Info("remove pre calculate score in cache", "pod", key.Namespace+"/"+key.Name,
						"removed score", deleted.String(), "after", tds.preCalRes.String(), "pods", cachePodsMapString(tds.cachedPods))
				} else {
					logger.V(5).Info("remove pre calculate score in cache", "pod", key.Namespace+"/"+key.Name,
						"removed score", deleted.String(), "after", tds.preCalRes.String(), "pods", cachePodsMapString(tds.cachedPods))
				}
				return
			}
			if p.Spec.Affinity == nil || p.Spec.Affinity.PodAffinity == nil && p.Spec.Affinity.PodAntiAffinity == nil {
				return
			}

			nodeName := reservedNode
			ni, err := snapshot.NodeInfos().Get(nodeName)
			if err != nil {
				return
			}
			pi, err := framework.NewPodInfo(p)
			if err != nil {
				return
			}
			fakePod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: key.Namespace, Labels: tds.labels}}
			topoMap := make(topologyToMatchedTermCount)
			topoMap.updateWithAntiAffinityTerms(pi.RequiredAntiAffinityTerms, fakePod, tds.labels, ni.Node(), 1)
			deleted := tds.RemovePod(key)
			tds.cachedPods[cacheplugin.NamespaceedNameNode{Namespace: key.Namespace, Name: key.Name, IsReservePod: key.IsReservePod, ReservationName: key.ReservationName}] = topoMap
			tds.preCalRes.merge(topoMap)
			if len(topoMap) > 0 || len(deleted) > 0 {
				logger.V(3).Info("update pre calculate score in cache", "pod", klog.KObj(p), "score", topoMap.String(), "removed score", deleted.String())
			}
		})
	poi.Informer().AddEventHandler(cache.FilteringResourceEventHandler{
		FilterFunc: func(obj interface{}) bool {
			switch t := obj.(type) {
			case cache.DeletedFinalStateUnknown:
				pod, ok := t.Obj.(*corev1.Pod)
				if !ok || pod == nil {
					return true
				}
				return pod.Spec.NodeName != ""
			case *corev1.Pod:
				return t.Spec.NodeName != ""
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
				c.Forget(func(at namespacedLabels, tds *FilteringExistingPodAffinityTermDetailedState) bool {
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
				c.Process(func(at namespacedLabels, tds *FilteringExistingPodAffinityTermDetailedState) {
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
	return &FilteringExisingPodCacheProxy{impl: c}
}

func NewFilteringIncomingPodAffinityTermDetailedState(afC, antiC topologyToMatchedTermCount,
	tpByPod map[cacheplugin.NamespaceedNameNode][]topologyToMatchedTermCount,
	af, anti []fwk.AffinityTerm) *FilteringIncomingPodAffinityTermDetailedState {
	affinityCount := afC.clone()
	antiAffinityCount := antiC.clone()
	return &FilteringIncomingPodAffinityTermDetailedState{
		affinityCounts:     affinityCount,
		antiAffinityCounts: antiAffinityCount,
		cachedPods:         tpByPod,
		affinity:           af,
		antiaffinity:       anti,
	}
}

type FilteringIncomingPodAffinityTermDetailedState struct {
	lock sync.RWMutex

	// we will find the number of pods by namespace and topoValue
	affinityCounts     topologyToMatchedTermCount
	antiAffinityCounts topologyToMatchedTermCount
	// we store ns/name for all cachedPods
	// key is ns/name, value is the topoValue
	cachedPods map[cacheplugin.NamespaceedNameNode][]topologyToMatchedTermCount
	//
	affinity     []fwk.AffinityTerm
	antiaffinity []fwk.AffinityTerm
}

type FilteringIncomingPodCacheProxy struct {
	impl *cacheplugin.CacheImpl[string, *FilteringIncomingPodAffinityTermDetailedState]
}

func (n *FilteringIncomingPodAffinityTermDetailedState) RemovePod(key cacheplugin.NamespaceedNameNode) (topologyToMatchedTermCount, topologyToMatchedTermCount) {
	tv, found := n.cachedPods[key]
	if !found {
		return topologyToMatchedTermCount{}, topologyToMatchedTermCount{}
	}
	delete(n.cachedPods, key)
	for k := range tv[0] {
		if n.affinityCounts[k] > 0 {
			n.affinityCounts[k]--
			if n.affinityCounts[k] == 0 {
				delete(n.affinityCounts, k)
			}
		}
	}
	for k := range tv[1] {
		if n.antiAffinityCounts[k] > 0 {
			n.antiAffinityCounts[k]--
			if n.antiAffinityCounts[k] == 0 {
				delete(n.antiAffinityCounts, k)
			}
		}
	}
	return tv[0], tv[1]
}

func NewFilteringIncomingPodCacheProxy(ctx context.Context, pl listersv1.PodLister, poi informerv1.PodInformer, nsi informerv1.NamespaceInformer, noi informerv1.NodeInformer, snapshot fwk.SharedLister) *FilteringIncomingPodCacheProxy {
	c := cacheplugin.NewCache[string, *FilteringIncomingPodAffinityTermDetailedState](ctx,
		"FilteringInterPodAffinity", cacheSize, workNum, func(pat string) string {
			return pat
		}, func(key cacheplugin.NamespaceedNameNode, tds *FilteringIncomingPodAffinityTermDetailedState, logger logr.Logger) {
			reservedNode := key.ReservedNode
			key.ReservedNode = ""
			// p, err := pl.Pods(key.Namespace).Get(key.Name)
			p, err := cacheplugin.GetPodFromLister(key, pl)
			tds.lock.Lock()
			defer tds.lock.Unlock()
			if err != nil || reservedNode == "" {
				deletedaffi, deleteantiaffi := tds.RemovePod(key)
				if len(deletedaffi) > 0 || len(deleteantiaffi) > 0 {
					logger.V(5).Info("remove pre calculate score in cache", "pod", key.Namespace+"/"+key.Name, "removed affi", deletedaffi.String(), "removed antiaffi", deleteantiaffi.String())
				}
				return
			}
			ni, err := snapshot.NodeInfos().Get(reservedNode)
			if err != nil {
				return
			}
			deletedaffi, deleteantiaffi := tds.RemovePod(key)
			affinity := make(topologyToMatchedTermCount)
			antiAffinity := make(topologyToMatchedTermCount)
			affinity.updateWithAffinityTerms(tds.affinity, p, ni.Node(), 1)
			// The incoming pod's terms have the namespaceSelector merged into the namespaces, and so
			// here we don't lookup the existing pod's namespace labels, hence passing nil for nsLabels.
			antiAffinity.updateWithAntiAffinityTerms(tds.antiaffinity, p, nil, ni.Node(), 1)
			tds.cachedPods[cacheplugin.NamespaceedNameNode{Namespace: key.Namespace, Name: key.Name, IsReservePod: key.IsReservePod, ReservationName: key.ReservationName}] = []topologyToMatchedTermCount{
				affinity, antiAffinity,
			}
			tds.affinityCounts.merge(affinity)
			tds.antiAffinityCounts.merge(antiAffinity)
			if len(affinity) > 0 || len(antiAffinity) > 0 || len(deletedaffi) > 0 || len(deleteantiaffi) > 0 {
				logger.V(3).Info("update pre calculate score in cache", "pod", klog.KObj(p),
					"affinity", affinity.String(), "antiAffinity", antiAffinity.String(), "removed affi", deletedaffi.String(), "removed antiaffi", deleteantiaffi.String())
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
				c.Process(func(at string, tds *FilteringIncomingPodAffinityTermDetailedState) {
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
				c.Forget(func(at string, tds *FilteringIncomingPodAffinityTermDetailedState) bool {
					return true
				})
			}
		},
	})
	return &FilteringIncomingPodCacheProxy{impl: c}
}
