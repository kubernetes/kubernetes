package containerrestart

import (
	"fmt"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/util/metrics"
)

const (
	ContainerRestart = "deployment.kubernetes.io/container-restart"
)

type ContainerRestartController struct {
	kubeClient clientset.Interface

	podControl controller.PodControlInterface

	syncHandler func(podKey string) error

	// A store of pods, populated by the shared informer passed to NewReplicaSetController
	podLister corelisters.PodLister
	// podListerSynced returns true if the pod store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	podListerSynced cache.InformerSynced

	// Controllers that need to be synced
	queue workqueue.RateLimitingInterface
}

func NewContainerRestartController(podInformer coreinformers.PodInformer, kubeClient clientset.Interface) *ContainerRestartController {
	if kubeClient != nil && kubeClient.CoreV1().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("podreadiness-controller", kubeClient.CoreV1().RESTClient().GetRateLimiter())
	}

	prg := &ContainerRestartController{
		kubeClient: kubeClient,
		queue:      workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "pod_readiness"),
	}

	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		UpdateFunc: prg.updatePod,
	})
	prg.podLister = podInformer.Lister()
	prg.podListerSynced = podInformer.Informer().HasSynced

	prg.syncHandler = prg.syncReplicaSet

	return prg
}

func (crc *ContainerRestartController) SetEventRecorder(recorder record.EventRecorder) {
	// TODO: Hack. We can't cleanly shutdown the event recorder, so benchmarks
	// need to pass in a fake.
	crc.podControl = controller.RealPodControl{KubeClient: crc.kubeClient, Recorder: recorder}
}

// Run begins watching and syncing.
func (crc *ContainerRestartController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer crc.queue.ShutDown()

	klog.Infoln("Starting pod readiness controller")
	defer klog.Infoln("Shutting down pod readiness controller")

	if !controller.WaitForCacheSync("pods", stopCh, crc.podListerSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(crc.worker, time.Second, stopCh)
	}

	<-stopCh
}

func (crc *ContainerRestartController) updatePod(old, cur interface{}) {
	curPod := cur.(*v1.Pod)
	oldPod := old.(*v1.Pod)
	if curPod.ResourceVersion == oldPod.ResourceVersion {
		return
	}
	if curPod.DeletionTimestamp != nil {
		return
	}
	for _, tp := range curPod.Spec.ReadinessGates {
		if tp.ConditionType == ContainerRestart {
			crc.enqueue(curPod)
		}
	}
}

func (crc *ContainerRestartController) enqueue(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %+v: %v", obj, err))
		return
	}
	crc.queue.Add(key)
}

func (crc *ContainerRestartController) enqueueAfter(obj interface{}, after time.Duration) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %+v: %v", obj, err))
		return
	}
	crc.queue.AddAfter(key, after)
}

func (crc *ContainerRestartController) worker() {
	for crc.processNextWorkItem() {
	}
}

func (crc *ContainerRestartController) processNextWorkItem() bool {
	key, quit := crc.queue.Get()
	if quit {
		return false
	}
	defer crc.queue.Done(key)

	err := crc.syncHandler(key.(string))
	if err == nil {
		crc.queue.Forget(key)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("Sync %s failed with %v", key, err))
	crc.queue.AddRateLimited(key)

	return true
}

func (crc *ContainerRestartController) syncReplicaSet(key string) error {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing container restart %q (%v)", key, time.Since(startTime))
	}()

	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}
	pod, err := crc.podLister.Pods(namespace).Get(name)
	if errors.IsNotFound(err) {
		klog.V(4).Infof(" pod container restart  %v has been deleted", key)
		return nil
	}
	if err != nil {
		return err
	}
	ready := false
	var conditions []v1.PodCondition
	for _, condition := range pod.Status.Conditions {
		// container
		if ContainerRestart == condition.Type {
			continue
		}
		if condition.Type == v1.ContainersReady && condition.Status == v1.ConditionTrue {
			ready = true
		}
		conditions = append(conditions, condition)
	}

	if ready {
		conditions = append(conditions, v1.PodCondition{
			Type:   ContainerRestart,
			Status: "True",
			LastTransitionTime: metav1.Time{Time: time.Now(),
			}})
	}
	pod.Status.Conditions = conditions
	_, err = crc.kubeClient.CoreV1().Pods(pod.Namespace).UpdateStatus(pod)

	return err
}
