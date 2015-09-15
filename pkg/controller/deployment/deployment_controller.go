/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package deployment

import (
	"fmt"
	"hash/adler32"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/experimental"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
)

type DeploymentController struct {
	client    client.Interface
	expClient client.ExperimentalInterface
}

func New(client client.Interface) *DeploymentController {
	return &DeploymentController{
		client:    client,
		expClient: client.Experimental(),
	}
}

func (d *DeploymentController) Run(syncPeriod time.Duration) {
	go util.Until(func() {
		if err := d.reconcileDeployments(); err != nil {
			glog.Errorf("Couldnt reconcile deployments: %v", err)
		}
	}, syncPeriod, util.NeverStop)
}

func (d *DeploymentController) reconcileDeployments() error {
	list, err := d.expClient.Deployments(api.NamespaceAll).List(labels.Everything(), fields.Everything())
	if err != nil {
		return fmt.Errorf("error listing deployments: %v", err)
	}
	for _, deployment := range list.Items {
		if err := d.reconcileDeployment(&deployment); err != nil {
			return fmt.Errorf("error in reconciling deployment: %v", err)
		}
	}
	return nil
}

func (d *DeploymentController) reconcileDeployment(deployment *experimental.Deployment) error {
	targetedRCs, err := d.getTargetedRCs(deployment)
	if err != nil {
		return err
	}
	desiredRC, err := d.getDesiredRC(deployment)
	if err != nil {
		return err
	}
	// TODO: Scale up and down the targeted and desired RCs.
	// For now, just print their names, until we start doing something useful.
	for _, targetedRC := range targetedRCs {
		glog.Infof("TargetedRC: %s", targetedRC.ObjectMeta.Name)
	}
	glog.Infof("DesiredRC: %s", desiredRC.ObjectMeta.Name)
	return nil
}

func (d *DeploymentController) getTargetedRCs(deployment *experimental.Deployment) ([]api.ReplicationController, error) {
	namespace := deployment.ObjectMeta.Namespace
	// 1. Find all pods whose labels match deployment.Spec.Selector
	podList, err := d.client.Pods(namespace).List(labels.SelectorFromSet(deployment.Spec.Selector), fields.Everything())
	if err != nil {
		return nil, fmt.Errorf("error listing pods: %v", err)
	}
	// 2. Find the corresponding RCs for pods in podList.
	// TODO: Right now we list all RCs and then filter. We should add an API for this.
	targetedRCs := map[string]api.ReplicationController{}
	rcList, err := d.client.ReplicationControllers(namespace).List(labels.Everything())
	if err != nil {
		return nil, fmt.Errorf("error listing replication controllers: %v", err)
	}
	for _, pod := range podList.Items {
		podLabelsSelector := labels.Set(pod.ObjectMeta.Labels)
		for _, rc := range rcList.Items {
			rcLabelsSelector := labels.SelectorFromSet(rc.Spec.Selector)
			if rcLabelsSelector.Matches(podLabelsSelector) {
				targetedRCs[rc.ObjectMeta.Name] = rc
				continue
			}
		}
	}
	requiredRCs := []api.ReplicationController{}
	for _, value := range targetedRCs {
		requiredRCs = append(requiredRCs, value)
	}
	return requiredRCs, nil
}

// Returns an RC that matches the intent of the given deployment.
// It creates a new RC if required.
func (d *DeploymentController) getDesiredRC(deployment *experimental.Deployment) (*api.ReplicationController, error) {
	namespace := deployment.ObjectMeta.Namespace
	// Find if the required RC exists already.
	rcList, err := d.client.ReplicationControllers(namespace).List(labels.Everything())
	if err != nil {
		return nil, fmt.Errorf("error listing replication controllers: %v", err)
	}
	for _, rc := range rcList.Items {
		if api.Semantic.DeepEqual(rc.Spec.Template, deployment.Spec.Template) {
			// This is the desired RC.
			return &rc, nil
		}
	}
	// desired RC does not exist, create a new one.
	podTemplateSpecHasher := adler32.New()
	util.DeepHashObject(podTemplateSpecHasher, deployment.Spec.Template)
	podTemplateSpecHash := podTemplateSpecHasher.Sum32()
	rcName := fmt.Sprintf("deploymentrc-%d", podTemplateSpecHash)
	desiredRC := api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name:      rcName,
			Namespace: namespace,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: 0,
			Template: deployment.Spec.Template,
		},
	}
	createdRC, err := d.client.ReplicationControllers(namespace).Create(&desiredRC)
	if err != nil {
		return nil, fmt.Errorf("error creating replication controller: %v", err)
	}
	return createdRC, nil
}
