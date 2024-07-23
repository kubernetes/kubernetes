/*
Copyright 2023 The Kubernetes Authors.

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

package validatingadmissionpolicystatus

import (
	"context"
	"fmt"
	"time"

	"k8s.io/api/admissionregistration/v1"
	kerrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	validatingadmissionpolicy "k8s.io/apiserver/pkg/admission/plugin/policy/validating"
	admissionregistrationv1apply "k8s.io/client-go/applyconfigurations/admissionregistration/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"

	kcpcache "github.com/kcp-dev/apimachinery/v2/pkg/cache"
	kcpinformerv1 "github.com/kcp-dev/client-go/informers/admissionregistration/v1"
	kcpadmissionregistrationv1 "github.com/kcp-dev/client-go/kubernetes/typed/admissionregistration/v1"
	"github.com/kcp-dev/logicalcluster/v3"
)

// ControllerName has "Status" in it to differentiate this controller with the other that runs in API server.
const ControllerName = "validatingadmissionpolicy-status"

// Controller is the ValidatingAdmissionPolicy Status controller that reconciles the Status field of each policy object.
// This controller runs type checks against referred types for each policy definition.
type Controller struct {
	policyInformer kcpinformerv1.ValidatingAdmissionPolicyClusterInformer
	policyQueue    workqueue.TypedRateLimitingInterface[string]

	policySynced cache.InformerSynced
	policyClient kcpadmissionregistrationv1.ValidatingAdmissionPolicyClusterInterface

	// typeChecker checks the policy's expressions for type errors.
	// Type of params is defined in policy.Spec.ParamsKind
	// Types of object are calculated from policy.Spec.MatchingConstraints
	typeCheckerFn func(clusterName logicalcluster.Path) (*validatingadmissionpolicy.TypeChecker, error)
}

func (c *Controller) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()

	if !cache.WaitForNamedCacheSync(ControllerName, ctx.Done(), c.policySynced) {
		return
	}

	defer c.policyQueue.ShutDown()
	for i := 0; i < workers; i++ {
		go wait.UntilWithContext(ctx, c.runWorker, time.Second)
	}

	<-ctx.Done()
}

func NewController(policyInformer kcpinformerv1.ValidatingAdmissionPolicyClusterInformer, policyClient kcpadmissionregistrationv1.ValidatingAdmissionPolicyClusterInterface, typeCheckerFn func(clusterName logicalcluster.Path) (*validatingadmissionpolicy.TypeChecker, error)) (*Controller, error) {
	c := &Controller{
		policyInformer: policyInformer,
		policyQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: ControllerName},
		),
		policyClient:  policyClient,
		typeCheckerFn: typeCheckerFn,
	}
	reg, err := policyInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			c.enqueuePolicy(obj)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			c.enqueuePolicy(newObj)
		},
	})
	if err != nil {
		return nil, err
	}
	c.policySynced = reg.HasSynced
	return c, nil
}

func (c *Controller) enqueuePolicy(policy any) {
	if policy, ok := policy.(*v1.ValidatingAdmissionPolicy); ok {
		// policy objects are cluster-scoped, no point include its namespace.
		key, err := kcpcache.MetaClusterNamespaceKeyFunc(policy)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("cannot get cluster namespace key from policy: %w", err))
		}
		if key == "" {
			utilruntime.HandleError(fmt.Errorf("cannot get name of object %v", policy))
		}
		c.policyQueue.Add(key)
	}
}

func (c *Controller) runWorker(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

func (c *Controller) processNextWorkItem(ctx context.Context) bool {
	key, shutdown := c.policyQueue.Get()
	if shutdown {
		return false
	}
	defer c.policyQueue.Done(key)

	err := func() error {
		clusterName, _, policyName, err := kcpcache.SplitMetaClusterNamespaceKey(key)
		if err != nil {
			return fmt.Errorf("failed to split key: %w", err)
		}

		policy, err := c.policyInformer.Lister().Cluster(clusterName).Get(policyName)
		if err != nil {
			if kerrors.IsNotFound(err) {
				// If not found, the policy is being deleting, do nothing.
				return nil
			}
			return err
		}
		return c.reconcile(ctx, policy)
	}()

	if err == nil {
		c.policyQueue.Forget(key)
		return true
	}

	utilruntime.HandleError(err)
	c.policyQueue.AddRateLimited(key)

	return true
}

func (c *Controller) reconcile(ctx context.Context, policy *v1.ValidatingAdmissionPolicy) error {
	if policy == nil {
		return nil
	}
	if policy.Generation <= policy.Status.ObservedGeneration {
		return nil
	}

	cluster := logicalcluster.From(policy)
	typeChecker, err := c.typeCheckerFn(cluster.Path())
	if err != nil {
		return err
	}

	warnings := typeChecker.Check(policy)
	warningsConfig := make([]*admissionregistrationv1apply.ExpressionWarningApplyConfiguration, 0, len(warnings))
	for _, warning := range warnings {
		warningsConfig = append(warningsConfig, admissionregistrationv1apply.ExpressionWarning().
			WithFieldRef(warning.FieldRef).
			WithWarning(warning.Warning))
	}
	applyConfig := admissionregistrationv1apply.ValidatingAdmissionPolicy(policy.Name).
		WithStatus(admissionregistrationv1apply.ValidatingAdmissionPolicyStatus().
			WithObservedGeneration(policy.Generation).
			WithTypeChecking(admissionregistrationv1apply.TypeChecking().
				WithExpressionWarnings(warningsConfig...)))
	_, err = c.policyClient.Cluster(cluster.Path()).ApplyStatus(ctx, applyConfig, metav1.ApplyOptions{FieldManager: ControllerName, Force: true})
	return err
}
