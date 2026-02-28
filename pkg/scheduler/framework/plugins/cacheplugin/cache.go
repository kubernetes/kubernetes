package cacheplugin

import (
	"context"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"golang.org/x/time/rate"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
)

var CacheItemTimeout = time.Minute * 1

type CacheForPlugin interface {
	// these func should return quickly
	ProcessReservePod(new *corev1.Pod, reservedNode string)
	ProcessUnreservePod(new *corev1.Pod, reservedNode string)
}

// cache should be update when
// [Pod] assumed/bound/unresumed/delete/update
// [Node] update
// [Namespace] update
type CacheImpl[T any, V any] struct {
	rwlock sync.RWMutex

	Name          string
	Size          int
	PriorityQueue []string
	OriginMap     map[string]T
	HashMap       map[string]*ItemInfo[V]
	KeyFunc       func(T) string

	logger      logr.Logger
	workerSize  int
	wq          workqueue.RateLimitingInterface
	podEvHandle func(key NamespaceedNameNode, t V, logger logr.Logger)
}

type ItemInfo[V any] struct {
	Item           V
	LastAccessTime time.Time
}

type NamespaceedNameNode struct {
	Namespace    string
	Name         string
	ReservedNode string

	IsReservePod    bool
	ReservationName string
}

func NewCache[T any, V any](ctx context.Context, name string, size, workerSize int, keyFunc func(T) string, podEvHandle func(key NamespaceedNameNode, t V, logger logr.Logger)) *CacheImpl[T, V] {
	bucketWq := workqueue.NewRateLimitingQueue(&workqueue.BucketRateLimiter{Limiter: rate.NewLimiter(5000, 1000)})
	c := &CacheImpl[T, V]{
		Name:          name,
		Size:          size,
		PriorityQueue: make([]string, 0),
		HashMap:       make(map[string]*ItemInfo[V]),
		OriginMap:     make(map[string]T),
		KeyFunc:       keyFunc,
		workerSize:    workerSize,
		wq:            bucketWq,
		podEvHandle:   podEvHandle,

		logger: klog.FromContext(ctx).WithName(name),
	}
	cacheplugin.lock.Lock()
	cacheplugin.caches = append(cacheplugin.caches, c)
	cacheplugin.lock.Unlock()
	c.Run(context.Background().Done())
	go wait.UntilWithContext(context.Background(), func(ctx context.Context) {
		c.rwlock.Lock()
		defer c.rwlock.Unlock()
		now := time.Now()
		toDelete := []string{}
		for k, v := range c.HashMap {
			if now.Sub(v.LastAccessTime) > CacheItemTimeout {
				for i, value := range c.PriorityQueue {
					if value == k && i > 0 {
						c.PriorityQueue = append(c.PriorityQueue[:i], c.PriorityQueue[i+1:]...)
						break
					} else if value == k && i == 0 {
						c.PriorityQueue = c.PriorityQueue[1:]
						break
					}
				}
				toDelete = append(toDelete, k)
			}
		}
		for _, k := range toDelete {
			delete(c.HashMap, k)
			delete(c.OriginMap, k)
		}
	}, time.Minute)
	return c
}

func (c *CacheImpl[T, V]) ProcessReservePod(new *corev1.Pod, reservedNode string) {
	c.rwlock.Lock()
	for k, v := range c.HashMap {
		c.podEvHandle(NamespaceedNameNode{Namespace: new.Namespace, Name: new.Name, ReservedNode: reservedNode, IsReservePod: false},
			v.Item, c.logger.WithValues("cache item", k))
	}
	c.rwlock.Unlock()
}

func (c *CacheImpl[T, V]) ProcessUnreservePod(new *corev1.Pod, reservedNode string) {
	c.rwlock.Lock()
	for k, v := range c.HashMap {
		c.podEvHandle(NamespaceedNameNode{Namespace: new.Namespace, Name: new.Name, ReservedNode: "", IsReservePod: false},
			v.Item, c.logger.WithValues("cache item", k))
	}
	c.rwlock.Unlock()
}

func (c *CacheImpl[T, V]) ProcessUpdatePod(new *corev1.Pod, reservedNode string) {
	c.wq.AddRateLimited(NamespaceedNameNode{Namespace: new.Namespace, Name: new.Name, ReservedNode: reservedNode, IsReservePod: false})
}

func (c *CacheImpl[T, V]) Forget(f func(T, V) bool) {
	c.rwlock.Lock()
	defer c.rwlock.Unlock()
	ids := []string{}
	for k, v := range c.OriginMap {
		forget := f(v, c.HashMap[k].Item)
		if forget {
			c.logger.V(5).Info("cache forget key", "key", k)
			ids = append(ids, c.KeyFunc(v))
		}
	}
	for _, id := range ids {
		delete(c.OriginMap, id)
		delete(c.HashMap, id)
		for i, k := range c.PriorityQueue {
			if k == id {
				c.PriorityQueue = append(c.PriorityQueue[:i], c.PriorityQueue[i+1:]...)
				break
			}
		}
	}
}

func (c *CacheImpl[T, V]) Process(f func(T, V)) {
	c.rwlock.Lock()
	defer c.rwlock.Unlock()
	for k, v := range c.OriginMap {
		f(v, c.HashMap[k].Item)
	}
}

func (c *CacheImpl[T, V]) Run(stopCh <-chan struct{}) {
	for i := 0; i < c.workerSize; i++ {
		go func() {
			for {
				select {
				case <-stopCh:
					return
				default:
					ev, st := c.wq.Get()
					if st {
						return
					}
					pev := ev.(NamespaceedNameNode)
					c.rwlock.RLock()
					for k, v := range c.HashMap {
						c.podEvHandle(pev, v.Item, c.logger.WithValues("cache item", k))
					}
					c.rwlock.RUnlock()
					c.wq.Done(ev)
				}
			}
		}()
	}
}

func (c *CacheImpl[T, V]) Read(t T) V {
	c.rwlock.RLock()
	defer c.rwlock.RUnlock()
	key := c.KeyFunc(t)
	found := false
	for i, k := range c.PriorityQueue {
		if k != key {
			continue
		}
		found = true
		if i >= len(c.PriorityQueue)-1 {
			break
		}
		c.PriorityQueue = append(c.PriorityQueue[:i], c.PriorityQueue[i+1:]...)
		c.PriorityQueue = append(c.PriorityQueue, key)
		break
	}
	if found {
		// c.logger.V(6).Info("read data from cache", "key", key)
		c.HashMap[key].LastAccessTime = time.Now()
		return c.HashMap[key].Item
	} else {
		// c.logger.V(5).Info("cache missed", "key", key)
		var zeroValue V
		return zeroValue
	}
}

func (c *CacheImpl[T, V]) AddIfNotPresent(t T, v V) {
	c.rwlock.Lock()
	defer c.rwlock.Unlock()
	key := c.KeyFunc(t)
	if _, ok := c.HashMap[key]; ok {
		return
	}
	if len(c.PriorityQueue) >= c.Size {
		toDelete := c.PriorityQueue[0]
		c.logger.V(5).Info("insert data in cache and delete some data", "key", key, "value", v, "to delete", toDelete)
		delete(c.HashMap, toDelete)
		delete(c.OriginMap, toDelete)
		c.PriorityQueue = c.PriorityQueue[1:]
		c.PriorityQueue = append(c.PriorityQueue, key)
	} else {
		c.PriorityQueue = append(c.PriorityQueue, key)
		c.logger.V(5).Info("insert data in cache", "key", key, "value", v, "size", len(c.PriorityQueue))
	}
	c.HashMap[key] = &ItemInfo[V]{
		Item:           v,
		LastAccessTime: time.Now(),
	}
	c.OriginMap[key] = t
}

func (c *CacheImpl[T, V]) Write(t T, v V) {
	c.rwlock.Lock()
	defer c.rwlock.Unlock()
	key := c.KeyFunc(t)
	if _, ok := c.HashMap[key]; ok {
		for i, k := range c.PriorityQueue {
			if k != key {
				continue
			}
			c.logger.V(5).Info("update data in cache", "key", key, "value", v)
			if i >= len(c.PriorityQueue)-1 {
				break
			}
			c.PriorityQueue = append(c.PriorityQueue[:i], c.PriorityQueue[i+1:]...)
			c.PriorityQueue = append(c.PriorityQueue, key)
			break
		}
	} else {
		if len(c.PriorityQueue) >= c.Size {
			toDelete := c.PriorityQueue[0]
			c.logger.V(5).Info("insert data in cache and delete some data", "key", key, "value", v, "to delete", toDelete)
			delete(c.HashMap, toDelete)
			delete(c.OriginMap, toDelete)
			c.PriorityQueue = append(c.PriorityQueue[1:], key)
		} else {
			c.PriorityQueue = append(c.PriorityQueue, key)
			c.logger.V(5).Info("insert data in cache", "key", key, "value", v, "size", len(c.PriorityQueue))
		}
	}

	c.HashMap[key] = &ItemInfo[V]{
		Item:           v,
		LastAccessTime: time.Now(),
	}
	c.OriginMap[key] = t
}
