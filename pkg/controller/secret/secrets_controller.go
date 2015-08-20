/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package secret

import (
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/secret"
	"k8s.io/kubernetes/pkg/secret/server_certificate/self_signed"
	"k8s.io/kubernetes/pkg/watch"
)

type SecretsControllerOptions struct {
	// SecretResync is the interval between full resyncs of Secrets.
	// If non-zero, all certificates will be re-listed this often.
	// Otherwise, re-list will be delayed as long as possible (until the watch is closed or times out).
	SecretResync time.Duration
}

func DefaultSecretsControllerOptions() SecretsControllerOptions {
	return SecretsControllerOptions{}
}

func loadSecretPlugins() []secret.SecretPlugin {
	allPlugins := []secret.SecretPlugin{}

	allPlugins = append(allPlugins, self_signed.ProbeSecretPlugins()...)

	return allPlugins
}

func NewSecretsController(cl client.Interface, options SecretsControllerOptions) *SecretsController {

	allPlugins := loadSecretPlugins()

	c := &SecretsController{
		client: cl,
	}

	c.pluginMgr.InitPlugins(allPlugins)

	_, certificateController := framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return cl.Secrets(api.NamespaceAll).List(labels.Everything(), fields.Everything())
			},
			WatchFunc: func(resourceVersion string) (watch.Interface, error) {
				return cl.Secrets(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), resourceVersion)
			},
		},
		&api.Secret{},
		options.SecretResync,
		framework.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				secret := obj.(*api.Secret)
				c.generateSecretIfRequired(secret)
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				secret := newObj.(*api.Secret)
				c.generateSecretIfRequired(secret)
			},
			DeleteFunc: func(obj interface{}) {
				secret := obj.(*api.Secret)
				glog.V(5).Infof("Deleting Secret[%s]\n", secret.Name)
				plugin, err := c.pluginMgr.FindPluginByType(secret.Type)
				if err != nil {
					glog.V(5).Infof("No secret plugin generator for %s\n", secret.Type)
					return
				}
				err = plugin.RevokeSecret(*secret)
				if err != nil {
					glog.V(2).Infof("Failed to revoke secret[%s]: %s\n", secret.Name, err)
				}
			},
		},
	)

	c.certificateController = certificateController
	return c
}

type SecretsController struct {
	certificateController *framework.Controller

	stopChan chan struct{}

	client client.Interface

	pluginMgr secret.SecretPluginMgr
}

// Run starts this recycler's control loops
func (c *SecretsController) Run() {
	glog.V(5).Infof("Starting SecretController\n")
	if c.stopChan == nil {
		c.stopChan = make(chan struct{})
		go c.certificateController.Run(c.stopChan)
	}
}

// Stop gracefully shuts down this binder
func (c *SecretsController) Stop() {
	glog.V(5).Infof("Stopping SecretsController\n")
	if c.stopChan != nil {
		close(c.stopChan)
		c.stopChan = nil
	}
}

func (c *SecretsController) generateSecretIfRequired(secret *api.Secret) {
	glog.V(5).Infof("Adding Secret[%s]\n", secret.Name)
	plugin, err := c.pluginMgr.FindPluginByType(secret.Type)
	if err != nil {
		glog.V(5).Infof("No secret plugin generator for %s\n", secret.Type)
		return
	}
	updatedSecret, err := plugin.GenerateSecret(*secret)
	if err != nil {
		glog.V(2).Infof("Failed to generate secret[%s]: %s\n", secret.Name, err)
		return
	}
	if updatedSecret != nil {
		_, err = c.client.Secrets(updatedSecret.ObjectMeta.Namespace).Update(updatedSecret)
		if err != nil {
			glog.V(2).Infof("Failed to update with generated secret[%s]: %s\n", secret.Name, err)
		}
	}
}
