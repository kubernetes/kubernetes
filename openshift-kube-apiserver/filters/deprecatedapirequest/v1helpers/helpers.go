package v1helpers

import (
	"context"

	apiclientv1 "github.com/openshift/client-go/apiserver/clientset/versioned/typed/apiserver/v1"

	apiv1 "github.com/openshift/api/apiserver/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/util/retry"
)

type UpdateStatusFunc func(status *apiv1.DeprecatedAPIRequestStatus)

func ApplyStatus(ctx context.Context, client apiclientv1.DeprecatedAPIRequestInterface, name string, updateFuncs ...UpdateStatusFunc) (*apiv1.DeprecatedAPIRequestStatus, bool, error) {
	updated := false
	var updatedStatus *apiv1.DeprecatedAPIRequestStatus
	err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		check, err := client.Get(ctx, name, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			// on a not found, let's create this thing.
			requestCount := &apiv1.DeprecatedAPIRequest{
				ObjectMeta: metav1.ObjectMeta{Name: name},
				// TODO fix spec here
				Spec: apiv1.DeprecatedAPIRequestSpec{RemovedRelease: "1.1"},
			}
			check, err = client.Create(ctx, requestCount, metav1.CreateOptions{})
		}
		if err != nil {
			return err
		}
		oldStatus := check.Status
		newStatus := oldStatus.DeepCopy()
		for _, update := range updateFuncs {
			update(newStatus)
		}
		if equality.Semantic.DeepEqual(oldStatus, newStatus) {
			updatedStatus = newStatus
			return nil
		}
		check, err = client.Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		check.Status = *newStatus
		updatedCheck, err := client.UpdateStatus(ctx, check, metav1.UpdateOptions{})
		if err != nil {
			return err
		}
		updatedStatus = &updatedCheck.Status
		updated = true
		return err
	})
	return updatedStatus, updated, err
}
