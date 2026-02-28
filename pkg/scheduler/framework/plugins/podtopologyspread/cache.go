package podtopologyspread

import (
	"context"
	"os"
	"reflect"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/go-logr/logr"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	informerv1 "k8s.io/client-go/informers/core/v1"
	listersv1 "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/cacheplugin"
	pointer "k8s.io/utils/ptr"
)

type topologyPair struct {
	key   string
	value string
}

// cachedPodsMap is a slice indexed by constraints
// each entry is a the topology value
type cachedPodsMap map[cacheplugin.NamespaceedNameNode][]string

func (c cachedPodsMap) AddPod(name, ns, node, value string, index int) {
	key := cacheplugin.NamespaceedNameNode{Namespace: ns, ReservedNode: node, Name: name}
	if c[key] == nil {
		c[key] = []string{}
	}
	for len(c[key]) <= index {
		c[key] = append(c[key], "")
	}
	c[key][index] = value
}

func (c cachedPodsMap) merge(caches ...cachedPodsMap) {
	for _, in := range caches {
		for k, v := range in {
			c[k] = v
		}
	}
}

type PodTopologySpreadState struct {
	lock sync.RWMutex

	preCalRes            []map[string]*int64
	cachedPods           cachedPodsMap
	namespace            string
	constraints          []topologySpreadConstraint
	requiredNodeAffinity nodeaffinity.RequiredNodeAffinity
	requireAllTopologies bool
}

func (n *PodTopologySpreadState) Append(caches []cachedPodsMap, preCalRes []map[string]*int64) {
	n.lock.Lock()
	defer n.lock.Unlock()

	n.cachedPods.merge(caches...)
	n.preCalRes = make([]map[string]*int64, len(preCalRes))
	for k, v := range preCalRes {
		n.preCalRes[k] = make(map[string]*int64)
		for tv, c := range v {
			n.preCalRes[k][tv] = c
		}
	}
}

func (n *PodTopologySpreadState) RemovePod(key cacheplugin.NamespaceedNameNode) []topologyPair {
	tv, found := n.cachedPods[key]
	if !found {
		return []topologyPair{}
	}
	delete(n.cachedPods, key)
	ret := make([]topologyPair, len(tv))
	l := len(n.constraints)
	if len(n.preCalRes) < l {
		klog.ErrorS(nil, "preCalRes's length is not equal to TopologyValueToPodCounts's length", "tv", tv, "preCalRes", n.preCalRes, "constraints", n.constraints)
		origin := n.preCalRes
		n.preCalRes = make([]map[string]*int64, l)
		copy(n.preCalRes, origin)
	}
	for k := range tv {
		if n.preCalRes[k] == nil {
			continue
		}
		value := tv[k]
		if l > k {
			ret = append(ret, topologyPair{
				key:   n.constraints[k].TopologyKey,
				value: value,
			})
		}
		if n.preCalRes[k][value] == nil {
			continue
		}
		if *n.preCalRes[k][value] > 0 {
			atomic.AddInt64(n.preCalRes[k][value], -1)
		}
	}
	return ret
}

type PodTopologySpreadCacheProxy struct {
	impl *cacheplugin.CacheImpl[string, *PodTopologySpreadState]
}

var cacheSize = 10
var workNum = 4

func init() {
	s := os.Getenv("PodTopologySpreadCacheSize")
	n := os.Getenv("PodTopologySpreadCacheWorkerNum")
	if s != "" {
		cacheSize, _ = strconv.Atoi(s)
	}
	if n != "" {
		workNum, _ = strconv.Atoi(n)
	}
}

func buildPodEvtHandle(plg *PodTopologySpread, pl listersv1.PodLister, poi informerv1.PodInformer, noi informerv1.NodeInformer, snapshot fwk.SharedLister) func(key cacheplugin.NamespaceedNameNode, t *PodTopologySpreadState, logger logr.Logger) {
	return func(key cacheplugin.NamespaceedNameNode, t *PodTopologySpreadState, logger logr.Logger) {
		reservedNode := key.ReservedNode
		key.ReservedNode = ""
		// p, err := pl.Pods(key.Namespace).Get(key.Name)
		p, err := cacheplugin.GetPodFromLister(key, pl)
		t.lock.Lock()
		defer t.lock.Unlock()
		if err != nil || reservedNode == "" {
			deleted := t.RemovePod(key)
			if len(deleted) > 0 {
				logger.V(5).Info("remove pre calculate score in cache", "pod", klog.KObj(p), "removed score", deleted)
			}
			return
		}
		deleted := t.RemovePod(key)
		// Bypass terminating Pod (see #87621).
		if p.DeletionTimestamp != nil || p.Namespace != t.namespace {
			return
		}
		ni, err := snapshot.NodeInfos().Get(reservedNode)
		if err != nil {
			return
		}
		if !plg.enableNodeInclusionPolicyInPodTopologySpread {
			// `node` should satisfy incoming pod's NodeSelector/NodeAffinity
			if match, _ := t.requiredNodeAffinity.Match(ni.Node()); !match {
				return
			}
		}
		// All topologyKeys need to be present in `node`
		if t.requireAllTopologies && !nodeLabelsMatchSpreadConstraints(ni.Node().Labels, t.constraints) {
			return
		}
		cache := make([]string, len(t.constraints))
		if len(t.preCalRes) < len(t.constraints) {
			klog.ErrorS(nil, "preCalRes's length is not equal to TopologyValueToPodCounts's length", "preCalRes", t.preCalRes, "constraints", t.constraints)
			origin := t.preCalRes
			t.preCalRes = make([]map[string]*int64, len(t.constraints))
			copy(t.preCalRes, origin)
		}
		for idx, c := range t.constraints {
			topologyValue := ni.Node().Labels[c.TopologyKey]
			if c.Selector.Matches(labels.Set(p.Labels)) {
				cache[idx] = topologyValue
				if t.preCalRes[idx] == nil {
					t.preCalRes[idx] = make(map[string]*int64)
				}
				if t.preCalRes[idx][topologyValue] == nil {
					t.preCalRes[idx][topologyValue] = pointer.To(int64(1))
				} else {
					atomic.AddInt64(t.preCalRes[idx][topologyValue], 1)
				}
			}
		}
		t.cachedPods[cacheplugin.NamespaceedNameNode{Namespace: key.Namespace, Name: key.Name, IsReservePod: key.IsReservePod, ReservationName: key.ReservationName}] = cache
		logger.V(5).Info("update pre calculate score in cache", "pod", klog.KObj(p),
			"updatedCache", cache, "removed", deleted)
	}
}

func NewPodTopologySpreadCacheProxy(ctx context.Context, plg *PodTopologySpread, pl listersv1.PodLister, poi informerv1.PodInformer, noi informerv1.NodeInformer, snapshot fwk.SharedLister) *PodTopologySpreadCacheProxy {
	c := cacheplugin.NewCache[string, *PodTopologySpreadState](ctx,
		"PodTopologySpread", cacheSize, workNum, func(hash string) string {
			return hash
		}, buildPodEvtHandle(plg, pl, poi, noi, snapshot))
	poi.Informer().AddEventHandler(cache.FilteringResourceEventHandler{
		FilterFunc: func(obj interface{}) bool {
			po := obj.(*v1.Pod)
			if po == nil || po.Spec.NodeName == "" {
				return false
			}
			return true
		},
		Handler: cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				po := obj.(*v1.Pod)
				if po == nil {
					return
				}
				c.ProcessUpdatePod(po, po.Spec.NodeName)
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				oldPo := oldObj.(*v1.Pod)
				newPo := newObj.(*v1.Pod)
				if oldPo == nil || newPo == nil {
					return
				}
				if !reflect.DeepEqual(oldPo.Labels, newPo.Labels) {
					c.ProcessUpdatePod(newPo, newPo.Spec.NodeName)
				}
			},
			DeleteFunc: func(obj interface{}) {
				var pod *v1.Pod
				switch t := obj.(type) {
				case *v1.Pod:
					pod = t
				case cache.DeletedFinalStateUnknown:
					pod, _ = t.Obj.(*v1.Pod)
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
			oldNo := oldObj.(*v1.Node)
			newNo := newObj.(*v1.Node)
			if oldNo == nil || newNo == nil {
				return
			}
			if !reflect.DeepEqual(oldNo.Labels, newNo.Labels) {
				c.Process(func(at string, tds *PodTopologySpreadState) {
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
	return &PodTopologySpreadCacheProxy{impl: c}
}
