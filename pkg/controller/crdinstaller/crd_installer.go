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

// Package crdinstaller implements a controller that manages the installation of
// CRDs that other core Kubernetes controllers depend on.
package crdinstaller

import (
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/util/runtime"
)

// CRDInstallationController defines the operations supported by this controller.
type CRDInstallationController interface {
	Run(stopCh <-chan struct{})
}

// NewCRDInstallationController returns a new instance of CRDInstaller.
func NewCRDInstallationController() (CRDInstallationController, error) {
	return &crdInstallationController{}, nil
}

type crdInstallationController struct {
}

func (cic *crdInstallationController) Run(stopCh <-chan struct{}) {
	defer runtime.HandleCrash()

	glog.Infof("Starting CRD Installation controller")
	defer glog.Infof("Shutting down CRD Installation controller")

	<-stopCh
}
