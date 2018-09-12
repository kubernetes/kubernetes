/*
Copyright 2018 The Kubernetes Authors.

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

// Package reconciler implements interfaces that attempt to reconcile the
// desired state of the with the actual state of the world by triggering
// actions.
package reconciler

import (
	"fmt"
	"github.com/golang/glog"
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/controller/crdinstaller/crdgenerator"
	"time"
)

// Reconciler runs a periodic loop that compares the CRDs that should be
// installed with the CRDs actually installed and triggers installation as
// needed.
type Reconciler interface {
	// Starts running the reconciliation loop which executes periodically, checks
	// which CRDs need to be installed and install them if they don't exist.
	Run(stopCh <-chan struct{})
}

// NewReconciler returns a new instance of Reconciler that waits loopPeriod
// between successive executions.
// loopPeriod is the amount of time the reconciler loop waits between
// successive executions.
// crdClient is the client used to read/write apiextensions.k8s.io objects
// from the API server.
// crdGenerators is a list of generators used to generate the CRDs that
// should exist.
func NewReconciler(
	loopPeriod time.Duration,
	crdClient apiextensionsclient.Interface,
	crdGenerators []crdgenerator.ControllerCRDGenerator) Reconciler {
	return &reconciler{
		loopPeriod:    loopPeriod,
		crdClient:     crdClient,
		crdGenerators: crdGenerators,
	}
}

type reconciler struct {
	// loopPeriod is the amount of time the reconciler loop waits between
	// successive executions.
	loopPeriod time.Duration

	// crdClient is the client used to read/write apiextensions.k8s.io objects
	// from the API server.
	crdClient apiextensionsclient.Interface

	// crdGenerators is a list of generators used to generate the CRDs that
	// should exist.
	crdGenerators []crdgenerator.ControllerCRDGenerator
}

func (rc *reconciler) Run(stopCh <-chan struct{}) {
	wait.Until(rc.reconciliationLoopFunc(), rc.loopPeriod, stopCh)
}

// It periodically checks whether the attached volumes from actual state
// are still attached to the node and update the status if they are not.
func (rc *reconciler) reconciliationLoopFunc() func() {
	return func() {
		rc.reconcile()
	}
}

func (rc *reconciler) reconcile() {
	for _, crdGenerator := range rc.crdGenerators {
		// For each generator, get the list of CRDs that should exist
		crdsToCreate := crdGenerator.GetCRDs()

		// Verify each CRD exists, otherwise create it
		for i, v := range crdsToCreate {
			fmt.Printf("i: %v   v: %v \r\n", i, v) // TEMP REMOVE
			rc.installCRD("test", v)
		}
	}
}

func (rc *reconciler) installCRD(
	name string,
	crd *apiextensionsv1beta1.CustomResourceDefinition) error {
	res, err := rc.crdClient.ApiextensionsV1beta1().CustomResourceDefinitions().Create(crd)

	if err == nil {
		glog.Infof("%s CRD created successfully: %v", name, res)
	} else if apierrors.IsAlreadyExists(err) {
		// TODO (saad-ali): consider making this logic smarter and comparing expected vs
		// actual fields.
		glog.Warningf("%s CRD already exists: %#v, err: %v", name, res, err)
		fmt.Printf("OH MY\r\n") // TEMP REMOVE
	} else {
		glog.Errorf("failed to create %s CRD: %#v, err: %v", name, res, err)
		fmt.Printf("OH MY OH MY OH MY WHAT\r\n") // TEMP REMOVE
		return err
	}

	return nil
}
