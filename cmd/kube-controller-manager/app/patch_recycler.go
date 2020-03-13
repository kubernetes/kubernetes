package app

import (
	"context"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/controller"
)

func createPVRecyclerSA(openshiftConfig string, clientBuilder controller.ControllerClientBuilder) error {
	if len(openshiftConfig) == 0 {
		return nil
	}

	//  the service account passed for the recyclable volume plugins needs to exist.  We want to do this via the init function, but its a kube init function
	// for the rebase, create that service account here
	coreClient, err := clientBuilder.Client("pv-recycler-controller-creator")
	if err != nil {
		return err
	}

	// Create the namespace if we can't verify it exists.
	// Tolerate errors, since we don't know whether this component has namespace creation permissions.
	_, _ = coreClient.CoreV1().Namespaces().Create(context.TODO(), &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "openshift-infra"}}, metav1.CreateOptions{})

	// Create the service account
	_, err = coreClient.CoreV1().ServiceAccounts("openshift-infra").Create(context.TODO(), &v1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Namespace: "openshift-infra", Name: "pv-recycler-controller"}}, metav1.CreateOptions{})
	if apierrors.IsAlreadyExists(err) {
		return nil
	}
	if err != nil {
		return err
	}

	return nil
}
