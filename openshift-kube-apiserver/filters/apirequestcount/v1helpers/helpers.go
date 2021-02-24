package v1helpers

import (
	"context"

	apiv1 "github.com/openshift/api/apiserver/v1"
	apiv1client "github.com/openshift/client-go/apiserver/clientset/versioned/typed/apiserver/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/util/retry"
)

type UpdateStatusFunc func(maxNumUsers int, status *apiv1.APIRequestCountStatus)

func ApplyStatus(ctx context.Context, client apiv1client.APIRequestCountInterface, name string, statusDefaulter UpdateStatusFunc, updateFuncs ...UpdateStatusFunc) (*apiv1.APIRequestCountStatus, bool, error) {
	updated := false
	var updatedStatus *apiv1.APIRequestCountStatus
	err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		existingOrDefaultAPIRequestCount, err := client.Get(ctx, name, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			// APIRequestCount might have been purposely deleted. We will
			// try to create it again further below if there is a need to.
			existingOrDefaultAPIRequestCount = &apiv1.APIRequestCount{
				ObjectMeta: metav1.ObjectMeta{Name: name},
				Spec:       apiv1.APIRequestCountSpec{NumberOfUsersToReport: 10},
			}
			// make sure the status doesn't result in a diff on a no-op.
			statusDefaulter(10, &existingOrDefaultAPIRequestCount.Status)
		} else if err != nil {
			return err
		}
		oldStatus := existingOrDefaultAPIRequestCount.Status
		newStatus := oldStatus.DeepCopy()
		for _, update := range updateFuncs {
			update(int(existingOrDefaultAPIRequestCount.Spec.NumberOfUsersToReport), newStatus)
		}
		if equality.Semantic.DeepEqual(&oldStatus, newStatus) {
			updatedStatus = newStatus
			return nil
		}

		// At this point the status has been semantically changed by the updateFuncs,
		// possibly due to new requests, hourly log expiration, and so on.

		existingAPIRequestCount, err := client.Get(ctx, name, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			// APIRequestCount might have been purposely deleted, but new requests
			// have come in, so let's re-create the APIRequestCount resource.
			newAPIRequestCount := &apiv1.APIRequestCount{
				ObjectMeta: metav1.ObjectMeta{Name: name},
				Spec: apiv1.APIRequestCountSpec{
					NumberOfUsersToReport: 10,
				},
			}
			existingAPIRequestCount, err = client.Create(ctx, newAPIRequestCount, metav1.CreateOptions{})
		}
		if err != nil {
			return err
		}
		existingAPIRequestCount.Status = *newStatus
		updatedAPIRequestCount, err := client.UpdateStatus(ctx, existingAPIRequestCount, metav1.UpdateOptions{})
		if err != nil {
			return err
		}
		updatedStatus = &updatedAPIRequestCount.Status
		updated = true
		return err
	})
	return updatedStatus, updated, err
}
