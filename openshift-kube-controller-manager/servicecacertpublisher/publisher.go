package servicecacertpublisher

import (
	"context"
	"fmt"
	"os"
	"reflect"
	"strconv"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
)

// ServiceCACertConfigMapName is name of the configmap which stores certificates
// to validate service serving certificates issued by the service ca operator.
const ServiceCACertConfigMapName = "openshift-service-ca.crt"

func init() {
	registerMetrics()
}

// NewPublisher construct a new controller which would manage the configmap
// which stores certificates in each namespace. It will make sure certificate
// configmap exists in each namespace.
func NewPublisher(cmInformer coreinformers.ConfigMapInformer, nsInformer coreinformers.NamespaceInformer, cl clientset.Interface) (*Publisher, error) {
	e := &Publisher{
		client: cl,
		queue:  workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "service_ca_cert_publisher"),
	}

	cmInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		DeleteFunc: e.configMapDeleted,
		UpdateFunc: e.configMapUpdated,
	})
	e.cmLister = cmInformer.Lister()
	e.cmListerSynced = cmInformer.Informer().HasSynced

	nsInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    e.namespaceAdded,
		UpdateFunc: e.namespaceUpdated,
	})
	e.nsListerSynced = nsInformer.Informer().HasSynced

	e.syncHandler = e.syncNamespace

	return e, nil
}

// Publisher manages certificate ConfigMap objects inside Namespaces
type Publisher struct {
	client clientset.Interface

	// To allow injection for testing.
	syncHandler func(key string) error

	cmLister       corelisters.ConfigMapLister
	cmListerSynced cache.InformerSynced

	nsListerSynced cache.InformerSynced

	queue workqueue.RateLimitingInterface
}

// Run starts process
func (c *Publisher) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.Infof("Starting service CA certificate configmap publisher")
	defer klog.Infof("Shutting down service CA certificate configmap publisher")

	if !cache.WaitForNamedCacheSync("crt configmap", stopCh, c.cmListerSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(c.runWorker, time.Second, stopCh)
	}

	<-stopCh
}

func (c *Publisher) configMapDeleted(obj interface{}) {
	cm, err := convertToCM(obj)
	if err != nil {
		utilruntime.HandleError(err)
		return
	}
	if cm.Name != ServiceCACertConfigMapName {
		return
	}
	c.queue.Add(cm.Namespace)
}

func (c *Publisher) configMapUpdated(_, newObj interface{}) {
	cm, err := convertToCM(newObj)
	if err != nil {
		utilruntime.HandleError(err)
		return
	}
	if cm.Name != ServiceCACertConfigMapName {
		return
	}
	c.queue.Add(cm.Namespace)
}

func (c *Publisher) namespaceAdded(obj interface{}) {
	namespace := obj.(*v1.Namespace)
	c.queue.Add(namespace.Name)
}

func (c *Publisher) namespaceUpdated(oldObj interface{}, newObj interface{}) {
	newNamespace := newObj.(*v1.Namespace)
	if newNamespace.Status.Phase != v1.NamespaceActive {
		return
	}
	c.queue.Add(newNamespace.Name)
}

func (c *Publisher) runWorker() {
	for c.processNextWorkItem() {
	}
}

// processNextWorkItem deals with one key off the queue. It returns false when
// it's time to quit.
func (c *Publisher) processNextWorkItem() bool {
	key, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(key)

	if err := c.syncHandler(key.(string)); err != nil {
		utilruntime.HandleError(fmt.Errorf("syncing %q failed: %v", key, err))
		c.queue.AddRateLimited(key)
		return true
	}

	c.queue.Forget(key)
	return true
}

var (
	// default secure
	// This annotation prompts the service ca operator to inject
	// the service ca bundle into the configmap.
	injectionAnnotation = map[string]string{
		"service.beta.openshift.io/inject-cabundle": "true",
	}
	setAnnotationOnce = sync.Once{}
)

func getInjectionAnnotation() map[string]string {
	setAnnotationOnce.Do(func() {
		// this envvar can be used to get the kube-controller-manager to inject a vulnerable legacy service ca
		// the kube-controller-manager carries no existing patches to launch, so we aren't going add new
		// perma-flags.
		// it would be nicer to find a way to pass this more obviously.  This is a deep side-effect.
		// though ideally, we see this age out over time.
		useVulnerable := os.Getenv("OPENSHIFT_USE_VULNERABLE_LEGACY_SERVICE_CA_CRT")
		if len(useVulnerable) == 0 {
			return
		}
		useVulnerableBool, err := strconv.ParseBool(useVulnerable)
		if err != nil {
			// caller went crazy, don't use this unless you're careful
			panic(err)
		}
		if useVulnerableBool {
			// This annotation prompts the service ca operator to inject
			// the vulnerable, legacy service ca bundle into the configmap.
			injectionAnnotation = map[string]string{
				"service.alpha.openshift.io/inject-vulnerable-legacy-cabundle": "true",
			}
		}
	})

	return injectionAnnotation
}

func (c *Publisher) syncNamespace(ns string) (err error) {
	startTime := time.Now()
	defer func() {
		recordMetrics(startTime, ns, err)
		klog.V(4).Infof("Finished syncing namespace %q (%v)", ns, time.Since(startTime))
	}()

	annotations := getInjectionAnnotation()

	cm, err := c.cmLister.ConfigMaps(ns).Get(ServiceCACertConfigMapName)
	switch {
	case apierrors.IsNotFound(err):
		_, err = c.client.CoreV1().ConfigMaps(ns).Create(context.TODO(), &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:        ServiceCACertConfigMapName,
				Annotations: annotations,
			},
			// Create new configmaps with the field referenced by the default
			// projected volume. This ensures that pods - including the pod for
			// service ca operator - will be able to start during initial
			// deployment before the service ca operator has responded to the
			// injection annotation.
			Data: map[string]string{
				"service-ca.crt": "",
			},
		}, metav1.CreateOptions{})
		// don't retry a create if the namespace doesn't exist or is terminating
		if apierrors.IsNotFound(err) || apierrors.HasStatusCause(err, v1.NamespaceTerminatingCause) {
			return nil
		}
		return err
	case err != nil:
		return err
	}

	if reflect.DeepEqual(cm.Annotations, annotations) {
		return nil
	}

	// copy so we don't modify the cache's instance of the configmap
	cm = cm.DeepCopy()
	cm.Annotations = annotations

	_, err = c.client.CoreV1().ConfigMaps(ns).Update(context.TODO(), cm, metav1.UpdateOptions{})
	return err
}

func convertToCM(obj interface{}) (*v1.ConfigMap, error) {
	cm, ok := obj.(*v1.ConfigMap)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			return nil, fmt.Errorf("couldn't get object from tombstone %#v", obj)
		}
		cm, ok = tombstone.Obj.(*v1.ConfigMap)
		if !ok {
			return nil, fmt.Errorf("tombstone contained object that is not a ConfigMap %#v", obj)
		}
	}
	return cm, nil
}
