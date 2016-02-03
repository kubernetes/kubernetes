package app

import (
	"k8s.io/kubernetes/cmd/uber-cluster-controller/app/options"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	"k8s.io/kubernetes/pkg/controller/cluster"
)

func Run(c *options.ClusterController) error {
	uberConfig, err := clientcmd.BuildConfigFromFlags(c.Ubernetes, "")
	if err != nil {
		return err
	}
	uberClient, err := client.New(uberConfig)
	if err != nil {
		return err
	}
	cc := cluster.New(uberClient)
	cc.Run()
	return nil
}