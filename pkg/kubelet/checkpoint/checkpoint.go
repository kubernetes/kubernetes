package checkpoint

import (
	"net/http"

	"github.com/emicklei/go-restful"
	v1 "k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
)

type Manager interface {
	HandleCheckpoint(*restful.Request, *restful.Response)
}

func NewManager(kubeClient clientset.Interface, podManager kubepod.Manager, checkpointFn checkpointFunction) Manager {
	return &checkpoint{
		kubeClient:   kubeClient,
		podManager:   podManager,
		checkpointFn: checkpointFn,
	}
}

type checkpointFunction func(*v1.Pod)

type checkpoint struct {
	kubeClient   clientset.Interface
	podManager   kubepod.Manager
	checkpointFn checkpointFunction
}

var _ Manager = &checkpoint{}

func (c *checkpoint) HandleCheckpoint(request *restful.Request, response *restful.Response) {
	pod, ok := c.podManager.GetPodByName(request.PathParameter("podNamespace"), request.PathParameter("podName"))
	if !ok {
		response.WriteHeader(http.StatusNotFound)
		return
	}
	if pod.Status.Phase != v1.PodRunning {
		response.WriteHeader(http.StatusConflict)
		return
	}
	c.checkpointFn(pod)
	response.WriteHeader(http.StatusOK)
	return
}
