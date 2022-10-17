/*
Copyright 2021 The Kubernetes Authors.

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

package podsecurity

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"

	// install conversions for types we need to convert
	_ "k8s.io/kubernetes/pkg/apis/apps/install"
	_ "k8s.io/kubernetes/pkg/apis/batch/install"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/features"

	admissionv1 "k8s.io/api/admission/v1"
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninit "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/warning"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/component-base/featuregate"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/core"
	podsecurityadmission "k8s.io/pod-security-admission/admission"
	podsecurityconfigloader "k8s.io/pod-security-admission/admission/api/load"
	podsecurityadmissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/pod-security-admission/metrics"
	"k8s.io/pod-security-admission/policy"
)

// PluginName is a string with the name of the plugin
const PluginName = "PodSecurity"

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(reader io.Reader) (admission.Interface, error) {
		return newPlugin(reader)
	})
}

// Plugin holds state for and implements the admission plugin.
type Plugin struct {
	*admission.Handler

	inspectedFeatureGates bool

	client          kubernetes.Interface
	namespaceLister corev1listers.NamespaceLister
	podLister       corev1listers.PodLister

	delegate *podsecurityadmission.Admission
}

var _ admission.ValidationInterface = &Plugin{}
var _ genericadmissioninit.WantsExternalKubeInformerFactory = &Plugin{}
var _ genericadmissioninit.WantsExternalKubeClientSet = &Plugin{}

var (
	defaultRecorder     *metrics.PrometheusRecorder
	defaultRecorderInit sync.Once
)

func getDefaultRecorder() metrics.Recorder {
	// initialize and register to legacy metrics once
	defaultRecorderInit.Do(func() {
		defaultRecorder = metrics.NewPrometheusRecorder(podsecurityadmissionapi.GetAPIVersion())
		defaultRecorder.MustRegister(legacyregistry.MustRegister)
	})
	return defaultRecorder
}

// newPlugin creates a new admission plugin.
func newPlugin(reader io.Reader) (*Plugin, error) {
	config, err := podsecurityconfigloader.LoadFromReader(reader)
	if err != nil {
		return nil, err
	}

	evaluator, err := policy.NewEvaluator(policy.DefaultChecks())
	if err != nil {
		return nil, fmt.Errorf("could not create PodSecurityRegistry: %w", err)
	}

	return &Plugin{
		Handler: admission.NewHandler(admission.Create, admission.Update),
		delegate: &podsecurityadmission.Admission{
			Configuration:    config,
			Evaluator:        evaluator,
			Metrics:          getDefaultRecorder(),
			PodSpecExtractor: SCCMutatingPodSpecExtractorInstance,
		},
	}, nil
}

// SetExternalKubeInformerFactory registers an informer
func (p *Plugin) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	namespaceInformer := f.Core().V1().Namespaces()
	p.namespaceLister = namespaceInformer.Lister()
	p.podLister = f.Core().V1().Pods().Lister()
	p.SetReadyFunc(namespaceInformer.Informer().HasSynced)
	p.updateDelegate()
}

// SetExternalKubeClientSet sets the plugin's client
func (p *Plugin) SetExternalKubeClientSet(client kubernetes.Interface) {
	p.client = client
	p.updateDelegate()
}

func (p *Plugin) updateDelegate() {
	// return early if we don't have what we need to set up the admission delegate
	if p.namespaceLister == nil {
		return
	}
	if p.podLister == nil {
		return
	}
	if p.client == nil {
		return
	}
	p.delegate.PodLister = podsecurityadmission.PodListerFromInformer(p.podLister)
	p.delegate.NamespaceGetter = podsecurityadmission.NamespaceGetterFromListerAndClient(p.namespaceLister, p.client)
}

func (c *Plugin) InspectFeatureGates(featureGates featuregate.FeatureGate) {
	c.inspectedFeatureGates = true
	policy.RelaxPolicyForUserNamespacePods(featureGates.Enabled(features.UserNamespacesPodSecurityStandards))
}

// ValidateInitialization ensures all required options are set
func (p *Plugin) ValidateInitialization() error {
	if !p.inspectedFeatureGates {
		return fmt.Errorf("%s did not see feature gates", PluginName)
	}
	if err := p.delegate.CompleteConfiguration(); err != nil {
		return fmt.Errorf("%s configuration error: %w", PluginName, err)
	}
	if err := p.delegate.ValidateConfiguration(); err != nil {
		return fmt.Errorf("%s invalid: %w", PluginName, err)
	}
	return nil
}

var (
	applicableResources = map[schema.GroupResource]bool{
		corev1.Resource("pods"):       true,
		corev1.Resource("namespaces"): true,
	}
)

func (p *Plugin) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	gr := a.GetResource().GroupResource()
	if !applicableResources[gr] && !p.delegate.PodSpecExtractor.HasPodSpec(gr) {
		return nil
	}

	result := p.delegate.Validate(ctx, &lazyConvertingAttributes{Attributes: a})
	for _, w := range result.Warnings {
		warning.AddWarning(ctx, "", w)
	}
	if len(result.AuditAnnotations) > 0 {
		annotations := make([]string, len(result.AuditAnnotations)*2)
		i := 0
		for k, v := range result.AuditAnnotations {
			annotations[i], annotations[i+1] = podsecurityadmissionapi.AuditAnnotationPrefix+k, v
			i += 2
		}
		audit.AddAuditAnnotations(ctx, annotations...)
	}
	if !result.Allowed {
		// start with a generic forbidden error
		retval := admission.NewForbidden(a, errors.New("Not allowed by PodSecurity")).(*apierrors.StatusError)
		// use message/reason/details/code from admission library if populated
		if result.Result != nil {
			if len(result.Result.Message) > 0 {
				retval.ErrStatus.Message = result.Result.Message
			}
			if len(result.Result.Reason) > 0 {
				retval.ErrStatus.Reason = result.Result.Reason
			}
			if result.Result.Details != nil {
				retval.ErrStatus.Details = result.Result.Details
			}
			if result.Result.Code != 0 {
				retval.ErrStatus.Code = result.Result.Code
			}
		}
		return retval
	}
	return nil
}

type lazyConvertingAttributes struct {
	admission.Attributes

	convertObjectOnce    sync.Once
	convertedObject      runtime.Object
	convertedObjectError error

	convertOldObjectOnce    sync.Once
	convertedOldObject      runtime.Object
	convertedOldObjectError error
}

func (l *lazyConvertingAttributes) GetObject() (runtime.Object, error) {
	l.convertObjectOnce.Do(func() {
		l.convertedObject, l.convertedObjectError = convert(l.Attributes.GetObject())
	})
	return l.convertedObject, l.convertedObjectError
}

func (l *lazyConvertingAttributes) GetOldObject() (runtime.Object, error) {
	l.convertOldObjectOnce.Do(func() {
		l.convertedOldObject, l.convertedOldObjectError = convert(l.Attributes.GetOldObject())
	})
	return l.convertedOldObject, l.convertedOldObjectError
}

func (l *lazyConvertingAttributes) GetOperation() admissionv1.Operation {
	return admissionv1.Operation(l.Attributes.GetOperation())
}

func (l *lazyConvertingAttributes) GetUserName() string {
	return l.GetUserInfo().GetName()
}

func convert(in runtime.Object) (runtime.Object, error) {
	var out runtime.Object
	switch in.(type) {
	case *core.Namespace:
		out = &corev1.Namespace{}
	case *core.Pod:
		out = &corev1.Pod{}
	case *core.ReplicationController:
		out = &corev1.ReplicationController{}
	case *core.PodTemplate:
		out = &corev1.PodTemplate{}
	case *apps.ReplicaSet:
		out = &appsv1.ReplicaSet{}
	case *apps.Deployment:
		out = &appsv1.Deployment{}
	case *apps.StatefulSet:
		out = &appsv1.StatefulSet{}
	case *apps.DaemonSet:
		out = &appsv1.DaemonSet{}
	case *batch.Job:
		out = &batchv1.Job{}
	case *batch.CronJob:
		out = &batchv1.CronJob{}
	default:
		return in, fmt.Errorf("unexpected type %T", in)
	}
	if err := legacyscheme.Scheme.Convert(in, out, nil); err != nil {
		return in, err
	}
	return out, nil
}
