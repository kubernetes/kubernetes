/*
Copyright 2019 The Kubernetes Authors.

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

package daemon

import (
	apps "k8s.io/api/apps/v1"
)

// sync is responsible for reconciling deployments on scaling events or when they
// are paused.
func (dsc *DaemonSetsController) sync(ds *apps.DaemonSet) error {
	return dsc.syncDaemonSetStatus(ds)
}

// syncDeploymentStatus checks if the status is up-to-date and sync it if necessary
func (dsc *DaemonSetsController) syncDaemonSetStatus(ds *apps.DaemonSet) error {
	toUpdate := ds.DeepCopy()
	// TODO: add Paused status in Daemonset Status.
	_, err := dsc.kubeClient.AppsV1().DaemonSets(ds.Namespace).UpdateStatus(toUpdate)
	return err
}
