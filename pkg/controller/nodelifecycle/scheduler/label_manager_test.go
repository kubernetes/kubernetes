package scheduler

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/clientcmd"
	controllerutil "k8s.io/kubernetes/pkg/controller/util/node"
)

func TestXxx(t *testing.T) {
	kubeconfigPath := os.Getenv("HOME") + "/.kube/config" // Adjust the path to your kubeconfig file
	config, err := clientcmd.BuildConfigFromFlags("", kubeconfigPath)
	if err != nil {
		panic(err.Error())
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		panic(err.Error())
	}

	ctx, _ := context.WithCancel(context.TODO())
	informerFactory := informers.NewSharedInformerFactory(clientset, 0)
	go informerFactory.Start(ctx.Done())

	podInformer := informerFactory.Core().V1().Pods()
	podInformer.Informer().AddIndexers(cache.Indexers{
		"spec.nodeName": func(obj interface{}) ([]string, error) {
			pod, ok := obj.(*v1.Pod)
			if !ok {
				return []string{}, nil
			}
			if len(pod.Spec.NodeName) == 0 {
				return []string{}, nil
			}
			return []string{pod.Spec.NodeName}, nil
		},
	})
	nodeInformer := informerFactory.Core().V1().Nodes()

	podIndexer := podInformer.Informer().GetIndexer()
	getPodsAssignedToNode := func(nodeName string) ([]*v1.Pod, error) {
		objs, err := podIndexer.ByIndex("spec.nodeName", nodeName)
		if err != nil {
			return nil, err
		}
		pods := make([]*v1.Pod, 0, len(objs))
		for _, obj := range objs {
			pod, ok := obj.(*v1.Pod)
			if !ok {
				continue
			}
			pods = append(pods, pod)
		}
		return pods, nil
	}
	podLister := podInformer.Lister()
	nodeLister := nodeInformer.Lister()

	go podInformer.Informer().Run(ctx.Done())
	go nodeInformer.Informer().Run(ctx.Done())

	labelManager := NewLabelManager(ctx, clientset, podLister, nodeLister, getPodsAssignedToNode)
	nodeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		UpdateFunc: controllerutil.CreateUpdateNodeHandler(func(oldNode, newNode *v1.Node) error {
			labelManager.NodeUpdated(oldNode, newNode)
			return nil
		}),
	})

	go labelManager.Run(ctx)

	time.Sleep(time.Second * 10)
	fmt.Println("ready!")

	time.Sleep(time.Minute)
}
