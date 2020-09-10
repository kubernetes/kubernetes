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

package upgrades

import (
	"context"
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ereplicaset "k8s.io/kubernetes/test/e2e/framework/replicaset"
	"k8s.io/kubernetes/test/e2e/upgrades"

	"github.com/onsi/ginkgo"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	interval = 10 * time.Second
	timeout  = 5 * time.Minute
	rsName   = "rs"
	scaleNum = 2
)

// TODO: Test that the replicaset stays available during master (and maybe
// node and cluster upgrades).

// ReplicaSetUpgradeTest tests that a replicaset survives upgrade.
type ReplicaSetUpgradeTest struct {
	UID types.UID
}

// Name returns the tracking name of the test.
func (ReplicaSetUpgradeTest) Name() string { return "[sig-apps] replicaset-upgrade" }

// Setup creates a ReplicaSet and makes sure it's replicas ready.
func (r *ReplicaSetUpgradeTest) Setup(f *framework.Framework) {
	c := f.ClientSet
	ns := f.Namespace.Name
	nginxImage := imageutils.GetE2EImage(imageutils.Nginx)

	ginkgo.By(fmt.Sprintf("Creating replicaset %s in namespace %s", rsName, ns))
	replicaSet := newReplicaSet(rsName, ns, 1, map[string]string{"test": "upgrade"}, "nginx", nginxImage)
	rs, err := c.AppsV1().ReplicaSets(ns).Create(context.TODO(), replicaSet, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("Waiting for replicaset %s to have all of its replicas ready", rsName))
	framework.ExpectNoError(e2ereplicaset.WaitForReadyReplicaSet(c, ns, rsName))

	r.UID = rs.UID
}

// Test checks whether the replicasets are the same after an upgrade.
func (r *ReplicaSetUpgradeTest) Test(f *framework.Framework, done <-chan struct{}, upgrade upgrades.UpgradeType) {
	c := f.ClientSet
	ns := f.Namespace.Name
	rsClient := c.AppsV1().ReplicaSets(ns)

	// Block until upgrade is done
	ginkgo.By(fmt.Sprintf("Waiting for upgrade to finish before checking replicaset %s", rsName))
	<-done

	// Verify the RS is the same (survives) after the upgrade
	ginkgo.By(fmt.Sprintf("Checking UID to verify replicaset %s survives upgrade", rsName))
	upgradedRS, err := rsClient.Get(context.TODO(), rsName, metav1.GetOptions{})
	framework.ExpectNoError(err)
	if upgradedRS.UID != r.UID {
		framework.ExpectNoError(fmt.Errorf("expected same replicaset UID: %v got: %v", r.UID, upgradedRS.UID))
	}

	ginkgo.By(fmt.Sprintf("Waiting for replicaset %s to have all of its replicas ready after upgrade", rsName))
	framework.ExpectNoError(e2ereplicaset.WaitForReadyReplicaSet(c, ns, rsName))

	// Verify the upgraded RS is active by scaling up the RS to scaleNum and ensuring all pods are Ready
	ginkgo.By(fmt.Sprintf("Scaling up replicaset %s to %d", rsName, scaleNum))
	_, err = e2ereplicaset.UpdateReplicaSetWithRetries(c, ns, rsName, func(rs *appsv1.ReplicaSet) {
		*rs.Spec.Replicas = scaleNum
	})
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("Waiting for replicaset %s to have all of its replicas ready after scaling", rsName))
	framework.ExpectNoError(e2ereplicaset.WaitForReadyReplicaSet(c, ns, rsName))
}

// Teardown cleans up any remaining resources.
func (r *ReplicaSetUpgradeTest) Teardown(f *framework.Framework) {
	// rely on the namespace deletion to clean up everything
}

// newReplicaSet returns a new ReplicaSet.
func newReplicaSet(name, namespace string, replicas int32, podLabels map[string]string, imageName, image string) *appsv1.ReplicaSet {
	return &appsv1.ReplicaSet{
		TypeMeta: metav1.TypeMeta{
			Kind:       "ReplicaSet",
			APIVersion: "apps/v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Spec: appsv1.ReplicaSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: podLabels,
			},
			Replicas: &replicas,
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: podLabels,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:            imageName,
							Image:           image,
							SecurityContext: &v1.SecurityContext{},
						},
					},
				},
			},
		},
	}
}
