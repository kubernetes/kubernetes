package cluster

import (
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	//"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
	"k8s.io/kubernetes/pkg/util"
)

const (
	ClusterResyncPeriod = 30 * time.Second
)

type ClusterController struct {
	uberClient         *client.Client
	clusterController  *framework.Controller
	clustertore        cache.StoreToClusterLister
}

func New(uberClient *client.Client) *ClusterController {
	c := &ClusterController {
		uberClient: uberClient,
	}

	c.clustertore.Store, c.clusterController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return c.uberClient.Clusters().List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return c.uberClient.Clusters().Watch(options)
			},
		},
		&api.Cluster{},
		ClusterResyncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc: c.addCluster,
			UpdateFunc: c.updateCluster,
			DeleteFunc: c.deleteCluster,
		},
	)

	return c
}

func (c *ClusterController) Run() {
	go c.clusterController.Run(util.NeverStop)
	select {}
}

func (c *ClusterController) addCluster(obj interface{}) {
	cluster := obj.(*api.Cluster)
	glog.Infof("addCluster(%v)", cluster)
}

func (c *ClusterController) updateCluster(old, cur interface{}) {
	oldCluster := old.(*api.Cluster)
	curCluster := cur.(*api.Cluster)
	glog.Infof("updateCluster(%v, %v)", oldCluster, curCluster)
}

func (c *ClusterController) deleteCluster(obj interface{}) {
	cluster := obj.(*api.Cluster)
	glog.Infof("deleteCluster(%v)", cluster)

}
