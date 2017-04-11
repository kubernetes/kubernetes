/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"flag"
	"time"

	"github.com/golang/glog"

	meta_v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/pkg/api"
	"k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/workqueue"
)

type Controller struct {
	indexer  cache.Indexer
	queue    workqueue.RateLimitingInterface
	informer cache.Controller
	f        ControllerFunc
}

func NewController(queue workqueue.RateLimitingInterface, indexer cache.Indexer, informer cache.Controller, f ControllerFunc) *Controller {
	return &Controller{
		informer: informer,
		indexer:  indexer,
		queue:    queue,
		f:        f,
	}
}

type ControllerFunc func(cache.Indexer, workqueue.RateLimitingInterface) bool

func (c *Controller) Run(threadiness int, stopCh chan struct{}) {
	// Let the workers stop when we are done
	defer c.queue.ShutDown()
	glog.Info("Starting Pod controller")

	go c.informer.Run(stopCh)

	for i := 0; i < threadiness; i++ {
		go wait.Until(c.runWorker, time.Second, stopCh)
	}

	<-stopCh
	glog.Info("Stopping Pod controller")
}

func (c *Controller) runWorker() {
	for c.f(c.indexer, c.queue) {
	}
}

func main() {
	var kubeconfig string
	var master string

	flag.StringVar(&kubeconfig, "kubeconfig", "", "absolute path to the kubeconfig file")
	flag.StringVar(&master, "master", "", "master url")
	flag.Parse()

	// Create the connection
	config, err := clientcmd.BuildConfigFromFlags(master, kubeconfig)
	if err != nil {
		glog.Fatal(err)
	}

	// creates the clientset
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		glog.Fatal(err)
	}

	// create the pod watcher
	podListWatcher := cache.NewListWatchFromClient(clientset.Core().RESTClient(), "pods", api.NamespaceDefault, fields.Everything())

	// create the workqueue
	queue := workqueue.NewRateLimitingQueue(workqueue.DefaultControllerRateLimiter())

	// Bind the workqueue to a cache with the help of an informer. This way we make sure than whenever the cache
	// is updated, the pod key is added to the workqueue. Note than when we finally process the item from the
	// workqueue we might see a newer version of the Pod than the version which was responsible for triggering the update.
	indexer, informer := cache.NewIndexerInformer(podListWatcher, &v1.Pod{}, 0, cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			key, err := cache.MetaNamespaceKeyFunc(obj)
			if err == nil {
				queue.Add(key)
			}
		},
		UpdateFunc: func(old interface{}, new interface{}) {
			key, err := cache.MetaNamespaceKeyFunc(new)
			if err == nil {
				queue.Add(key)
			}
		},
		DeleteFunc: func(obj interface{}) {
			// IndexerInformer uses a delta queue, therefore for deletes we have to use this
			// key function.
			key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
			if err == nil {
				queue.Add(key)
			}
		},
	}, cache.Indexers{})

	controller := NewController(queue, indexer, informer, func(indexer cache.Indexer, queue workqueue.RateLimitingInterface) bool {
		// Wait until there is a new item in the working queue
		key, quit := queue.Get()
		if quit {
			return false
		}
		// Tell the queue that we are done with processing this key. This unblocks the key for other workers
		// This allows safe parallel processing because two pods with the same key are never processed in
		// parallel.
		defer queue.Done(key)
		// Fetch the latest Pod state from cache
		obj, exists, err := indexer.GetByKey(key.(string))

		if err != nil {
			// TODO: Does this make sense?
			// Tricky what to do in this situation. One thing we can do, is enqueueing it a few times to
			// add some backoff delays on the invalid key. This way we avoid hotlooping
			// on invalid keys.
			if queue.NumRequeues(key) < 5 {
				queue.AddRateLimited(key)
			} else {
				queue.Forget(key)
			}
			glog.Fatalf("Fetching object with key %s from store failed", key.(string))
			return true
		}

		if !exists {
			// Below we will warm up our cache with a Pod, so that we will see a delete for one pod
			glog.Infof("Pod %s does not exist anymore\n", key.(string))
		} else {
			// Note that you also have to check the uid if you have a local controlled resource, which
			// is dependent on the actual instance, to detect that a Pod was recreated with the same name
			glog.Infof("Sync/Add/Update for Pod %s\n", obj.(*v1.Pod).GetName())
		}
		// On a successful run, forget the error history of the key
		queue.Forget(key)
		return true
	})

	// We can now warm up the cache for initial synchronization
	// Le's suppose that we knew about a pod mypod on our last run, so we add it to the cache
	indexer.Add(&v1.Pod{
		ObjectMeta: meta_v1.ObjectMeta{
			Name:      "mypod",
			Namespace: v1.NamespaceDefault,
		},
	})

	// Now let's start the controller
	stop := make(chan struct{})
	defer close(stop)
	go controller.Run(1, stop)

	// Wait forever
	select {}
}
