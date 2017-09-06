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
	"k8s.io/apimachinery/pkg/api/errors"

	etcdrpc "github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes"
)

func interpretWatchError(err error) error {
	switch {
	case err == etcdrpc.ErrCompacted:
		return errors.NewResourceExpired("The resourceVersion for the provided watch is too old.")
	}
	return err
}

func interpretListError(err error, paging bool) error {
	switch {
	case err == etcdrpc.ErrCompacted:
		if paging {
			return errors.NewResourceExpired("The provided from parameter is too old to display a consistent list result. You must start a new list without the from.")
		}
		return errors.NewResourceExpired("The resourceVersion for the provided list is too old.")
	}
	return err
}
