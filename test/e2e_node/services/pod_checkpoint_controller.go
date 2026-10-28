/*
Copyright The Kubernetes Authors.

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

package services

import (
	"context"

	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/controller/podcheckpoint"
	"k8s.io/kubernetes/test/e2e/framework"
)

// PodCheckpointController assigns checkpoint requests in the node test API server.
type PodCheckpointController struct {
	host   string
	cancel context.CancelFunc
	done   chan struct{}
}

func NewPodCheckpointController(host string) *PodCheckpointController {
	return &PodCheckpointController{host: host}
}

func (c *PodCheckpointController) Start(ctx context.Context) error {
	config := restclient.AddUserAgent(&restclient.Config{
		Host:        c.host,
		BearerToken: framework.TestContext.BearerToken,
		TLSClientConfig: restclient.TLSClientConfig{
			Insecure: true,
		},
	}, "podcheckpoint-controller")
	client, err := clientset.NewForConfig(config)
	if err != nil {
		return err
	}
	factory := informers.NewSharedInformerFactory(client, 0)
	controller := podcheckpoint.NewController(client, factory.Core().V1().Pods())
	ctx, c.cancel = context.WithCancel(ctx)
	c.done = make(chan struct{})
	factory.Start(ctx.Done())
	go func() {
		defer close(c.done)
		controller.Run(ctx, 1)
	}()
	return nil
}

func (c *PodCheckpointController) Stop() error {
	if c.cancel != nil {
		c.cancel()
		<-c.done
	}
	return nil
}

func (c *PodCheckpointController) Name() string {
	return "podcheckpoint-controller"
}
