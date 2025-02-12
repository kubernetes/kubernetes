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

package rc

import (
	"context"
	"fmt"

	"github.com/onsi/ginkgo/v2"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientset "k8s.io/client-go/kubernetes"
	scaleclient "k8s.io/client-go/scale"
	e2edebug "k8s.io/kubernetes/test/e2e/framework/debug"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2eresource "k8s.io/kubernetes/test/e2e/framework/resource"
	testutils "k8s.io/kubernetes/test/utils"
	"k8s.io/utils/pointer"
)

// ByNameContainer returns a ReplicationController with specified name and container
func ByNameContainer(name string, replicas int32, labels map[string]string, c v1.Container,
	gracePeriod *int64) *v1.ReplicationController {

	zeroGracePeriod := int64(0)

	if gracePeriod == nil {
		gracePeriod = &zeroGracePeriod
	}
	return &v1.ReplicationController{
		TypeMeta: metav1.TypeMeta{
			Kind:       "ReplicationController",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: pointer.Int32(replicas),
			Selector: labels,
			Template: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: v1.PodSpec{
					Containers:                    []v1.Container{c},
					TerminationGracePeriodSeconds: gracePeriod,
				},
			},
		},
	}
}

// DeleteRCAndWaitForGC deletes only the Replication Controller and waits for GC to delete the pods.
func DeleteRCAndWaitForGC(ctx context.Context, c clientset.Interface, ns, name string) error {
	// TODO (pohly): context support
	return e2eresource.DeleteResourceAndWaitForGC(ctx, c, schema.GroupKind{Kind: "ReplicationController"}, ns, name)
}

// ScaleRC scales Replication Controller to be desired size.
func ScaleRC(ctx context.Context, clientset clientset.Interface, scalesGetter scaleclient.ScalesGetter, ns, name string, size uint, wait bool) error {
	return e2eresource.ScaleResource(ctx, clientset, scalesGetter, ns, name, size, wait, schema.GroupKind{Kind: "ReplicationController"}, v1.SchemeGroupVersion.WithResource("replicationcontrollers"))
}

// RunRC Launches (and verifies correctness) of a Replication Controller
// and will wait for all pods it spawns to become "Running".
func RunRC(ctx context.Context, config testutils.RCConfig) error {
	ginkgo.By(fmt.Sprintf("creating replication controller %s in namespace %s", config.Name, config.Namespace))
	config.NodeDumpFunc = e2edebug.DumpNodeDebugInfo
	config.ContainerDumpFunc = e2ekubectl.LogFailedContainers
	return testutils.RunRC(ctx, config)
}
