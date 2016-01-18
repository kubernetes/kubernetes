package event

import (
	"sort"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	gcCheckPeriod = 20 * time.Second
)

type GCController struct {
	kubeClient       client.Interface
	eventStore       cache.StoreToEventLister
	eventStoreSyncer *framework.Controller
	eventTTL         time.Duration
	// extract the shouldGC logic for injection for testing.
	shouldGC    func(api.Event) bool
	deleteEvent func(namespace, name string) error
}

func New(kubeClient client.Interface, resyncPeriod controller.ResyncPeriodFunc, eventTTL time.Duration) *GCController {
	gcc := &GCController{
		kubeClient: kubeClient,
		eventTTL:   eventTTL,
		shouldGC: func(event api.Event) bool {
			return time.Now().Sub(event.CreationTimestamp.Time) >= eventTTL
		},
		deleteEvent: func(namespace, name string) error {
			return kubeClient.Events(namespace).Delete(name)
		},
	}

	gcc.eventStore.Store, gcc.eventStoreSyncer = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return gcc.kubeClient.Events(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return gcc.kubeClient.Events(api.NamespaceAll).Watch(options)
			},
		},
		&api.Pod{},
		resyncPeriod(),
		framework.ResourceEventHandlerFuncs{},
	)
	return gcc
}

func (gcc *GCController) Run(stop <-chan struct{}) {
	go gcc.eventStoreSyncer.Run(stop)
	go util.Until(gcc.gc, gcCheckPeriod, stop)
	<-stop
}

func (gcc *GCController) gc() {
	events, _ := gcc.eventStore.List()
	sort.Sort(byCreationTimestamp(events.Items))

	index := sort.Search(len(events.Items), func(i int) bool {
		return !gcc.shouldGC(events.Items[i])
	})

	var wait sync.WaitGroup
	for i := 0; i < index; i++ {
		wait.Add(1)
		go func(namespace string, name string) {
			defer wait.Done()
			if err := gcc.deleteEvent(namespace, name); err != nil {
				// ignore not founds
				defer util.HandleError(err)
			}
		}(events.Items[i].Namespace, events.Items[i].Name)
	}
	wait.Wait()
}

// byCreationTimestamp sorts a list by creation timestamp, using their names as a tie breaker.
type byCreationTimestamp []api.Event

func (o byCreationTimestamp) Len() int      { return len(o) }
func (o byCreationTimestamp) Swap(i, j int) { o[i], o[j] = o[j], o[i] }

func (o byCreationTimestamp) Less(i, j int) bool {
	if o[i].CreationTimestamp.Equal(o[j].CreationTimestamp) {
		return o[i].Name < o[j].Name
	}
	return o[i].CreationTimestamp.Before(o[j].CreationTimestamp)
}
