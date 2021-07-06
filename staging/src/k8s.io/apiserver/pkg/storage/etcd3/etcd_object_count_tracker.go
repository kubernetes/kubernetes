/*
Copyright 2021 The Kubernetes Authors.

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
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/klog/v2"
)

// ObjectCountTrackerFunc is used as a callback to get notified when
// we have a up to date value of the total number of objects in the
// storage for a given resource.
// {group}.{resource} combination is used as the unique key.
type ObjectCountTrackerFunc func(groupResource string, count int64)

func (f ObjectCountTrackerFunc) OnCount(groupResource string, count int64) {
	f(groupResource, count)
}

// WithObjectCountTracker takes the given storage.Interface and wraps it so
// we can get notified when Count is invoked.
func WithObjectCountTracker(delegate storage.Interface, callback ObjectCountTrackerFunc) storage.Interface {
	return &objectCountTracker{
		Interface: delegate,
		callback:  callback,
	}
}

type objectCountTracker struct {
	// embed because we only want to decorate Count
	storage.Interface
	callback ObjectCountTrackerFunc
}

func (s *objectCountTracker) Count(key string) (int64, error) {
	count, err := s.Interface.Count(key)
	if s.callback == nil {
		return count, err
	}

	if err != nil {
		klog.ErrorS(err, "Storage object OnCount callback not invoked", "key", key)
		return count, err
	}

	s.callback.OnCount(key, count)
	return count, err
}
