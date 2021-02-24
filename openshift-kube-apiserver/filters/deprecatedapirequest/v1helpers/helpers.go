package v1helpers

import (
	"context"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/runtime"

	apiv1 "github.com/openshift/api/apiserver/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/util/retry"
)

type DeprecatedAPIRequestClient interface {
	Get(ctx context.Context, name string, opts metav1.GetOptions) (*apiv1.DeprecatedAPIRequest, error)
	UpdateStatus(ctx context.Context, deprecatedAPIRequest *apiv1.DeprecatedAPIRequest, opts metav1.UpdateOptions) (*apiv1.DeprecatedAPIRequest, error)
}

type UpdateStatusFunc func(status *apiv1.DeprecatedAPIRequestStatus)

func UpdateStatus(ctx context.Context, client DeprecatedAPIRequestClient, name string, updateFuncs ...UpdateStatusFunc) (*apiv1.DeprecatedAPIRequestStatus, bool, error) {
	updated := false
	var updatedStatus *apiv1.DeprecatedAPIRequestStatus
	err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		check, err := client.Get(ctx, name, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			// do not retry on NotFound
			runtime.HandleError(err)
			return nil
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
