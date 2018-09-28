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

package etcd3

import (
	"errors"

	etcdrpc "github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
)

const (
	expiredList     = "The resourceVersion for the provided list is too old."
	expiredWatch    = "The resourceVersion for the provided watch is too old."
	continueExpired = "The provided continue parameter is too old " +
		"to display a consistent list result. You can start a new list without " +
		"the continue parameter."
	inconsistentContinue = "The provided continue parameter is too old " +
		"to display a consistent list result. You can start a new list without " +
		"the continue parameter, or use the continue token in this response to " +
		"retrieve the remainder of the results. Continuing with the provided " +
		"token results in an inconsistent list - objects that were created, " +
		"modified, or deleted between the time the first chunk was returned " +
		"and now may show up in the list."
	incompleteTTL = "store.updateState needs current objState for TTL calculation"
)

// errIncompleteTTL signals that we need to perform a TTL calculation on update,
// but lack all of the information required to determine the correct value.
// see store.updateState for further details
var errIncompleteTTL = errors.New(incompleteTTL)

func interpretWatchError(err error) error {
	switch {
	case err == etcdrpc.ErrCompacted:
		return apierrors.NewResourceExpired(expiredWatch)
	}
	return err
}

func interpretListError(err error, paging bool, continueKey, keyPrefix string) error {
	switch {
	case err == etcdrpc.ErrCompacted:
		if paging {
			return handleCompactedErrorForPaging(continueKey, keyPrefix)
		}
		return apierrors.NewResourceExpired(expiredList)
	}
	return err
}

func handleCompactedErrorForPaging(continueKey, keyPrefix string) error {
	// continueToken.ResoureVersion=-1 means that the apiserver can
	// continue the list at the latest resource version. We don't use rv=0
	// for this purpose to distinguish from a bad token that has empty rv.
	newToken, err := encodeContinue(continueKey, keyPrefix, -1)
	if err != nil {
		utilruntime.HandleError(err)
		return apierrors.NewResourceExpired(continueExpired)
	}
	statusError := apierrors.NewResourceExpired(inconsistentContinue)
	statusError.ErrStatus.ListMeta.Continue = newToken
	return statusError
}
