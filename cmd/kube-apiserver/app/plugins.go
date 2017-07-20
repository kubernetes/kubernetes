/*
Copyright 2014 The Kubernetes Authors.

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

package app

// This file exists to force the desired plugin implementations to be linked.
// This should probably be part of some configuration fed into the build for a
// given binary target.
import (
	// Cloud providers
	_ "k8s.io/kubernetes/pkg/cloudprovider/providers"

	// Admission policies
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/plugin/pkg/admission/admit"
	"k8s.io/kubernetes/plugin/pkg/admission/alwayspullimages"
	"k8s.io/kubernetes/plugin/pkg/admission/antiaffinity"
	"k8s.io/kubernetes/plugin/pkg/admission/defaulttolerationseconds"
	"k8s.io/kubernetes/plugin/pkg/admission/deny"
	"k8s.io/kubernetes/plugin/pkg/admission/exec"
	"k8s.io/kubernetes/plugin/pkg/admission/gc"
	"k8s.io/kubernetes/plugin/pkg/admission/imagepolicy"
	"k8s.io/kubernetes/plugin/pkg/admission/initialization"
	"k8s.io/kubernetes/plugin/pkg/admission/initialresources"
	"k8s.io/kubernetes/plugin/pkg/admission/limitranger"
	"k8s.io/kubernetes/plugin/pkg/admission/namespace/autoprovision"
	"k8s.io/kubernetes/plugin/pkg/admission/namespace/exists"
	"k8s.io/kubernetes/plugin/pkg/admission/noderestriction"
	"k8s.io/kubernetes/plugin/pkg/admission/persistentvolume/label"
	"k8s.io/kubernetes/plugin/pkg/admission/podnodeselector"
	"k8s.io/kubernetes/plugin/pkg/admission/podpreset"
	"k8s.io/kubernetes/plugin/pkg/admission/podtolerationrestriction"
	"k8s.io/kubernetes/plugin/pkg/admission/resourcequota"
	"k8s.io/kubernetes/plugin/pkg/admission/security/podsecuritypolicy"
	"k8s.io/kubernetes/plugin/pkg/admission/securitycontext/scdeny"
	"k8s.io/kubernetes/plugin/pkg/admission/serviceaccount"
	"k8s.io/kubernetes/plugin/pkg/admission/storageclass/setdefault"
	"k8s.io/kubernetes/plugin/pkg/admission/webhook"
)

// RegisterAllAdmissionPlugins registers all admission plugins
func RegisterAllAdmissionPlugins(plugins *admission.Plugins) {
	admit.Register(plugins)
	alwayspullimages.Register(plugins)
	antiaffinity.Register(plugins)
	defaulttolerationseconds.Register(plugins)
	deny.Register(plugins)
	exec.Register(plugins)
	gc.Register(plugins)
	imagepolicy.Register(plugins)
	initialization.Register(plugins)
	initialresources.Register(plugins)
	limitranger.Register(plugins)
	autoprovision.Register(plugins)
	exists.Register(plugins)
	noderestriction.Register(plugins)
	label.Register(plugins)
	podnodeselector.Register(plugins)
	podpreset.Register(plugins)
	podtolerationrestriction.Register(plugins)
	resourcequota.Register(plugins)
	podsecuritypolicy.Register(plugins)
	scdeny.Register(plugins)
	serviceaccount.Register(plugins)
	setdefault.Register(plugins)
	webhook.Register(plugins)
}
