package wait

import (
	"context"
	"fmt"
	"io"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	watch2 "k8s.io/apimachinery/pkg/watch"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/tools/watch"
)

type DeletionWaiter struct {
	errOut io.Writer
}

func NewDeletionWaiter(errOut io.Writer) Waiter {
	return DeletionWaiter{errOut}
}

func (d DeletionWaiter) VisitResource(info *resource.Info, o *WaitOptions) (runtime.Object, bool, error) {
	endTime := time.Now().Add(o.Timeout)
	for {
		if len(info.Name) == 0 {
			return info.Object, false, fmt.Errorf("resource name must be provided")
		}

		nameSelector := fields.OneTermEqualSelector("metadata.name", info.Name).String()

		// List with a name field selector to get the current resourceVersion to watch from (not the object's resourceVersion)
		gottenObjList, err := o.DynamicClient.Resource(info.Mapping.Resource).Namespace(info.Namespace).List(context.TODO(), v1.ListOptions{FieldSelector: nameSelector})
		if apierrors.IsNotFound(err) {
			return info.Object, true, nil
		}
		if err != nil {
			// TODO this could do something slightly fancier if we wish
			return info.Object, false, err
		}
		if len(gottenObjList.Items) != 1 {
			return info.Object, true, nil
		}
		gottenObj := &gottenObjList.Items[0]
		resourceLocation := ResourceLocation{
			GroupResource: info.Mapping.Resource.GroupResource(),
			Namespace:     gottenObj.GetNamespace(),
			Name:          gottenObj.GetName(),
		}
		if uid, ok := o.UIDMap[resourceLocation]; ok {
			if gottenObj.GetUID() != uid {
				return gottenObj, true, nil
			}
		}

		watchOptions := v1.ListOptions{}
		watchOptions.FieldSelector = nameSelector
		watchOptions.ResourceVersion = gottenObjList.GetResourceVersion()
		objWatch, err := o.DynamicClient.Resource(info.Mapping.Resource).Namespace(info.Namespace).Watch(context.TODO(), watchOptions)
		if err != nil {
			return gottenObj, false, err
		}

		timeout := endTime.Sub(time.Now())
		errWaitTimeoutWithName := extendErrWaitTimeout(wait.ErrWaitTimeout, info)
		if timeout < 0 {
			// we're out of time
			return gottenObj, false, errWaitTimeoutWithName
		}

		ctx, cancel := watch.ContextWithOptionalTimeout(context.Background(), o.Timeout)
		watchEvent, err := watch.UntilWithoutRetry(ctx, objWatch, d.IsDeleted)
		cancel()
		switch {
		case err == nil:
			return watchEvent.Object, true, nil
		case err == watch.ErrWatchClosed:
			continue
		case err == wait.ErrWaitTimeout:
			if watchEvent != nil {
				return watchEvent.Object, false, errWaitTimeoutWithName
			}
			return gottenObj, false, errWaitTimeoutWithName
		default:
			return gottenObj, false, err
		}
	}
}

func (d DeletionWaiter) OnWaitLoopCompletion(visitedCount int, err error) error {
	if apierrors.IsNotFound(err) || err == nil {
		return nil
	} else {
		return err
	}
}

// IsDeleted returns true if the object is deleted. It prints any errors it encounters.
func (d DeletionWaiter) IsDeleted(event watch2.Event) (bool, error) {
	switch event.Type {
	case watch2.Error:
		// keep waiting in the event we see an error - we expect the watch to be closed by
		// the server if the error is unrecoverable.
		err := apierrors.FromObject(event.Object)
		fmt.Fprintf(d.errOut, "error: An error occurred while waiting for the object to be deleted: %v", err)
		return false, nil
	case watch2.Deleted:
		return true, nil
	default:
		return false, nil
	}
}
