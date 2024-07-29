/*
Copyright 2017 The Kubernetes Authors.

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

package clusterroleaggregation

import (
	"context"
	"fmt"
	"sort"
	"time"

	rbacv1ac "k8s.io/client-go/applyconfigurations/rbac/v1"
	"k8s.io/klog/v2"

	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	rbacinformers "k8s.io/client-go/informers/rbac/v1"
	rbacclient "k8s.io/client-go/kubernetes/typed/rbac/v1"
	rbaclisters "k8s.io/client-go/listers/rbac/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"

	"k8s.io/kubernetes/pkg/controller"
)

// ClusterRoleAggregationController is a controller to combine cluster roles
type ClusterRoleAggregationController struct {
	clusterRoleClient  rbacclient.ClusterRolesGetter
	clusterRoleLister  rbaclisters.ClusterRoleLister
	clusterRolesSynced cache.InformerSynced

	syncHandler func(ctx context.Context, key string) error
	queue       workqueue.TypedRateLimitingInterface[string]
}

// NewClusterRoleAggregation creates a new controller
func NewClusterRoleAggregation(clusterRoleInformer rbacinformers.ClusterRoleInformer, clusterRoleClient rbacclient.ClusterRolesGetter) *ClusterRoleAggregationController {
	c := &ClusterRoleAggregationController{
		clusterRoleClient:  clusterRoleClient,
		clusterRoleLister:  clusterRoleInformer.Lister(),
		clusterRolesSynced: clusterRoleInformer.Informer().HasSynced,

		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{
				Name: "ClusterRoleAggregator",
			},
		),
	}
	c.syncHandler = c.syncClusterRole

	clusterRoleInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			c.enqueue()
		},
		UpdateFunc: func(old, cur interface{}) {
			c.enqueue()
		},
		DeleteFunc: func(uncast interface{}) {
			c.enqueue()
		},
	})
	return c
}

func (c *ClusterRoleAggregationController) syncClusterRole(ctx context.Context, key string) error {
	_, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}
	sharedClusterRole, err := c.clusterRoleLister.Get(name)
	if errors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}
	if sharedClusterRole.AggregationRule == nil {
		return nil
	}

	newPolicyRules := []rbacv1.PolicyRule{}
	for i := range sharedClusterRole.AggregationRule.ClusterRoleSelectors {
		selector := sharedClusterRole.AggregationRule.ClusterRoleSelectors[i]
		runtimeLabelSelector, err := metav1.LabelSelectorAsSelector(&selector)
		if err != nil {
			return err
		}
		clusterRoles, err := c.clusterRoleLister.List(runtimeLabelSelector)
		if err != nil {
			return err
		}
		sort.Sort(byName(clusterRoles))

		for i := range clusterRoles {
			if clusterRoles[i].Name == sharedClusterRole.Name {
				continue
			}

			for j := range clusterRoles[i].Rules {
				currRule := clusterRoles[i].Rules[j]
				if !ruleExists(newPolicyRules, currRule) {
					newPolicyRules = append(newPolicyRules, currRule)
				}
			}
		}
	}

	if equality.Semantic.DeepEqual(newPolicyRules, sharedClusterRole.Rules) {
		return nil
	}

	err = c.applyClusterRoles(ctx, sharedClusterRole.Name, newPolicyRules)
	if errors.IsUnsupportedMediaType(err) { // TODO: Remove this fallback at least one release after ServerSideApply GA
		// When Server Side Apply is not enabled, fallback to Update. This is required when running
		// 1.21 since api-server can be 1.20 during the upgrade/downgrade.
		// Since Server Side Apply is enabled by default in Beta, this fallback only kicks in
		// if the feature has been disabled using its feature flag.
		err = c.updateClusterRoles(ctx, sharedClusterRole, newPolicyRules)
	}
	return err
}

func (c *ClusterRoleAggregationController) applyClusterRoles(ctx context.Context, name string, newPolicyRules []rbacv1.PolicyRule) error {
	clusterRoleApply := rbacv1ac.ClusterRole(name).
		WithRules(toApplyPolicyRules(newPolicyRules)...)

	opts := metav1.ApplyOptions{FieldManager: "clusterrole-aggregation-controller", Force: true}
	_, err := c.clusterRoleClient.ClusterRoles().Apply(ctx, clusterRoleApply, opts)
	return err
}

func (c *ClusterRoleAggregationController) updateClusterRoles(ctx context.Context, sharedClusterRole *rbacv1.ClusterRole, newPolicyRules []rbacv1.PolicyRule) error {
	clusterRole := sharedClusterRole.DeepCopy()
	clusterRole.Rules = nil
	for _, rule := range newPolicyRules {
		clusterRole.Rules = append(clusterRole.Rules, *rule.DeepCopy())
	}
	_, err := c.clusterRoleClient.ClusterRoles().Update(ctx, clusterRole, metav1.UpdateOptions{})
	return err
}

func toApplyPolicyRules(rules []rbacv1.PolicyRule) []*rbacv1ac.PolicyRuleApplyConfiguration {
	var result []*rbacv1ac.PolicyRuleApplyConfiguration
	for _, rule := range rules {
		result = append(result, toApplyPolicyRule(rule))
	}
	return result
}

func toApplyPolicyRule(rule rbacv1.PolicyRule) *rbacv1ac.PolicyRuleApplyConfiguration {
	result := rbacv1ac.PolicyRule()
	result.Resources = rule.Resources
	result.ResourceNames = rule.ResourceNames
	result.APIGroups = rule.APIGroups
	result.NonResourceURLs = rule.NonResourceURLs
	result.Verbs = rule.Verbs
	return result
}

func ruleExists(haystack []rbacv1.PolicyRule, needle rbacv1.PolicyRule) bool {
	for _, curr := range haystack {
		if equality.Semantic.DeepEqual(curr, needle) {
			return true
		}
	}
	return false
}

// Run starts the controller and blocks until stopCh is closed.
func (c *ClusterRoleAggregationController) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrashWithContext(ctx)
	defer c.queue.ShutDown()

	logger := klog.FromContext(ctx)
	logger.Info("Starting ClusterRoleAggregator controller")
	defer logger.Info("Shutting down ClusterRoleAggregator controller")

	if !cache.WaitForNamedCacheSyncWithContext(ctx, c.clusterRolesSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.UntilWithContext(ctx, c.runWorker, time.Second)
	}

	<-ctx.Done()
}

func (c *ClusterRoleAggregationController) runWorker(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

func (c *ClusterRoleAggregationController) processNextWorkItem(ctx context.Context) bool {
	dsKey, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(dsKey)

	err := c.syncHandler(ctx, dsKey)
	if err == nil {
		c.queue.Forget(dsKey)
		return true
	}

	utilruntime.HandleErrorWithContext(ctx, err, "Syncing failed", "key", dsKey)
	c.queue.AddRateLimited(dsKey)

	return true
}

func (c *ClusterRoleAggregationController) enqueue() {
	// this is unusual, but since the set of all clusterroles is small and we don't know the dependency
	// graph, just queue up every thing each time.  This allows errors to be selectively retried if there
	// is a problem updating a single role
	allClusterRoles, err := c.clusterRoleLister.List(labels.Everything())
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't list all objects %v", err)) //nolint:logcheck // Not reached, listing from cache should never fail.
		return
	}
	for _, clusterRole := range allClusterRoles {
		// only queue ones that we may need to aggregate
		if clusterRole.AggregationRule == nil {
			continue
		}
		key, err := controller.KeyFunc(clusterRole)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %#v: %v", clusterRole, err)) //nolint:logcheck // Not reached, all objects have a key.
			return
		}
		c.queue.Add(key)
	}
}

type byName []*rbacv1.ClusterRole

func (a byName) Len() int           { return len(a) }
func (a byName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byName) Less(i, j int) bool { return a[i].Name < a[j].Name }
