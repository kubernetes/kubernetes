/*
Copyright 2016 The Kubernetes Authors.

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

package master

import (
	"encoding/json"
	"fmt"
	"runtime"
	"time"

	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	"k8s.io/kubernetes/pkg/api"
	apierrs "k8s.io/kubernetes/pkg/api/errors"
	unversionedapi "k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/util/wait"
)

const apiCallRetryInterval = 500 * time.Millisecond

func CreateClientAndWaitForAPI(adminConfig *clientcmdapi.Config) (*clientset.Clientset, error) {
	adminClientConfig, err := clientcmd.NewDefaultClientConfig(
		*adminConfig,
		&clientcmd.ConfigOverrides{},
	).ClientConfig()
	if err != nil {
		return nil, fmt.Errorf("<master/apiclient> failed to create API client configuration [%v]", err)
	}

	fmt.Println("<master/apiclient> created API client configuration")

	client, err := clientset.NewForConfig(adminClientConfig)
	if err != nil {
		return nil, fmt.Errorf("<master/apiclient> failed to create API client [%v]", err)
	}

	fmt.Println("<master/apiclient> created API client, waiting for the control plane to become ready")

	start := time.Now()
	wait.PollInfinite(apiCallRetryInterval, func() (bool, error) {
		cs, err := client.ComponentStatuses().List(api.ListOptions{})
		if err != nil {
			return false, nil
		}
		// TODO(phase2) must revisit this when we implement HA
		if len(cs.Items) < 3 {
			fmt.Println("<master/apiclient> not all control plane components are ready yet")
			return false, nil
		}
		for _, item := range cs.Items {
			for _, condition := range item.Conditions {
				if condition.Type != api.ComponentHealthy {
					fmt.Printf("<master/apiclient> control plane component %q is still unhealthy: %#v\n", item.ObjectMeta.Name, item.Conditions)
					return false, nil
				}
			}
		}

		fmt.Printf("<master/apiclient> all control plane components are healthy after %f seconds\n", time.Since(start).Seconds())
		return true, nil
	})

	fmt.Println("<master/apiclient> waiting for at least one node to register and become ready")
	start = time.Now()
	wait.PollInfinite(apiCallRetryInterval, func() (bool, error) {
		nodeList, err := client.Nodes().List(api.ListOptions{})
		if err != nil {
			fmt.Println("<master/apiclient> temporarily unable to list nodes (will retry)")
			return false, nil
		}
		if len(nodeList.Items) < 1 {
			return false, nil
		}
		n := &nodeList.Items[0]
		if !api.IsNodeReady(n) {
			fmt.Println("<master/apiclient> first node has registered, but is not ready yet")
			return false, nil
		}

		fmt.Printf("<master/apiclient> first node is ready after %f seconds\n", time.Since(start).Seconds())
		return true, nil
	})

	createDummyDeployment(client)

	return client, nil
}

func standardLabels(n string) map[string]string {
	return map[string]string{
		"component": n, "name": n, "k8s-app": n,
		"kubernetes.io/cluster-service": "true", "tier": "node",
	}
}

func NewDaemonSet(daemonName string, podSpec api.PodSpec) *extensions.DaemonSet {
	l := standardLabels(daemonName)
	return &extensions.DaemonSet{
		ObjectMeta: api.ObjectMeta{Name: daemonName},
		Spec: extensions.DaemonSetSpec{
			Selector: &unversionedapi.LabelSelector{MatchLabels: l},
			Template: api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{Labels: l},
				Spec:       podSpec,
			},
		},
	}
}

func NewService(serviceName string, spec api.ServiceSpec) *api.Service {
	l := standardLabels(serviceName)
	return &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name:   serviceName,
			Labels: l,
		},
		Spec: spec,
	}
}

func NewDeployment(deploymentName string, replicas int32, podSpec api.PodSpec) *extensions.Deployment {
	l := standardLabels(deploymentName)
	return &extensions.Deployment{
		ObjectMeta: api.ObjectMeta{Name: deploymentName},
		Spec: extensions.DeploymentSpec{
			Replicas: replicas,
			Selector: &unversionedapi.LabelSelector{MatchLabels: l},
			Template: api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{Labels: l},
				Spec:       podSpec,
			},
		},
	}
}

// It's safe to do this for alpha, as we don't have HA and there is no way we can get
// more then one node here (TODO(phase1+) use os.Hostname)
func findMyself(client *clientset.Clientset) (*api.Node, error) {
	nodeList, err := client.Nodes().List(api.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("unable to list nodes [%v]", err)
	}
	if len(nodeList.Items) < 1 {
		return nil, fmt.Errorf("no nodes found")
	}
	node := &nodeList.Items[0]
	return node, nil
}

func attemptToUpdateMasterRoleLabelsAndTaints(client *clientset.Clientset, schedulable bool) error {
	n, err := findMyself(client)
	if err != nil {
		return err
	}

	n.ObjectMeta.Labels[unversionedapi.NodeLabelKubeadmAlphaRole] = unversionedapi.NodeLabelRoleMaster

	if !schedulable {
		taintsAnnotation, _ := json.Marshal([]api.Taint{{Key: "dedicated", Value: "master", Effect: "NoSchedule"}})
		n.ObjectMeta.Annotations[api.TaintsAnnotationKey] = string(taintsAnnotation)
	}

	if _, err := client.Nodes().Update(n); err != nil {
		if apierrs.IsConflict(err) {
			fmt.Println("<master/apiclient> temporarily unable to update master node metadata due to conflict (will retry)")
			time.Sleep(apiCallRetryInterval)
			attemptToUpdateMasterRoleLabelsAndTaints(client, schedulable)
		} else {
			return err
		}
	}

	return nil
}

func UpdateMasterRoleLabelsAndTaints(client *clientset.Clientset, schedulable bool) error {
	// TODO(phase1+) use iterate instead of recursion
	err := attemptToUpdateMasterRoleLabelsAndTaints(client, schedulable)
	if err != nil {
		return fmt.Errorf("<master/apiclient> failed to update master node - %v", err)
	}
	return nil
}

func SetMasterTaintTolerations(meta *api.ObjectMeta) {
	tolerationsAnnotation, _ := json.Marshal([]api.Toleration{{Key: "dedicated", Value: "master", Effect: "NoSchedule"}})
	if meta.Annotations == nil {
		meta.Annotations = map[string]string{}
	}
	meta.Annotations[api.TolerationsAnnotationKey] = string(tolerationsAnnotation)
}

// SetNodeAffinity is a basic helper to set meta.Annotations[api.AffinityAnnotationKey] for one or more api.NodeSelectorRequirement(s)
func SetNodeAffinity(meta *api.ObjectMeta, expr ...api.NodeSelectorRequirement) {
	nodeAffinity := &api.NodeAffinity{
		RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{
			NodeSelectorTerms: []api.NodeSelectorTerm{{MatchExpressions: expr}},
		},
	}
	affinityAnnotation, _ := json.Marshal(api.Affinity{NodeAffinity: nodeAffinity})
	if meta.Annotations == nil {
		meta.Annotations = map[string]string{}
	}
	meta.Annotations[api.AffinityAnnotationKey] = string(affinityAnnotation)
}

// MasterNodeAffinity returns api.NodeSelectorRequirement to be used with SetNodeAffinity to set affinity to master node
func MasterNodeAffinity() api.NodeSelectorRequirement {
	return api.NodeSelectorRequirement{
		Key:      unversionedapi.NodeLabelKubeadmAlphaRole,
		Operator: api.NodeSelectorOpIn,
		Values:   []string{unversionedapi.NodeLabelRoleMaster},
	}
}

// NativeArchitectureNodeAffinity returns api.NodeSelectorRequirement to be used with SetNodeAffinity to nodes with CPU architecture
// the same as master node
func NativeArchitectureNodeAffinity() api.NodeSelectorRequirement {
	return api.NodeSelectorRequirement{
		Key: "beta.kubernetes.io/arch", Operator: api.NodeSelectorOpIn, Values: []string{runtime.GOARCH},
	}
}

func createDummyDeployment(client *clientset.Clientset) {
	fmt.Println("<master/apiclient> attempting a test deployment")
	dummyDeployment := NewDeployment("dummy", 1, api.PodSpec{
		SecurityContext: &api.PodSecurityContext{HostNetwork: true},
		Containers: []api.Container{{
			Name:  "dummy",
			Image: images.GetAddonImage("pause"),
		}},
	})

	wait.PollInfinite(apiCallRetryInterval, func() (bool, error) {
		// TODO: we should check the error, as some cases may be fatal
		if _, err := client.Extensions().Deployments(api.NamespaceSystem).Create(dummyDeployment); err != nil {
			fmt.Printf("<master/apiclient> failed to create test deployment [%v] (will retry)", err)
			return false, nil
		}
		return true, nil
	})

	wait.PollInfinite(apiCallRetryInterval, func() (bool, error) {
		d, err := client.Extensions().Deployments(api.NamespaceSystem).Get("dummy")
		if err != nil {
			fmt.Printf("<master/apiclient> failed to get test deployment [%v] (will retry)", err)
			return false, nil
		}
		if d.Status.AvailableReplicas < 1 {
			return false, nil
		}
		return true, nil
	})

	fmt.Println("<master/apiclient> test deployment succeeded")

	if err := client.Extensions().Deployments(api.NamespaceSystem).Delete("dummy", &api.DeleteOptions{}); err != nil {
		fmt.Printf("<master/apiclient> failed to delete test deployment [%v] (will ignore)", err)
	}
}
