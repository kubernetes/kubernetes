/*
Copyright 2024 The Kubernetes Authors.

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

package watchlist

import (
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metainternalversionvalidation "k8s.io/apimachinery/pkg/apis/meta/internalversion/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientfeatures "k8s.io/client-go/features"
	"k8s.io/utils/ptr"
)

var scheme = runtime.NewScheme()

func init() {
	utilruntime.Must(metainternalversion.AddToScheme(scheme))
}

// PrepareWatchListOptionsFromListOptions creates a new ListOptions
// that can be used for a watch-list request from the given listOptions.
//
// This function also determines if the given listOptions can be used to form a watch-list request,
// which would result in streaming semantically equivalent data from the server.
func PrepareWatchListOptionsFromListOptions(listOptions metav1.ListOptions) (metav1.ListOptions, bool, error) {
	if !clientfeatures.FeatureGates().Enabled(clientfeatures.WatchListClient) {
		return metav1.ListOptions{}, false, nil
	}

	internalListOptions := &metainternalversion.ListOptions{}
	if err := scheme.Convert(&listOptions, internalListOptions, nil); err != nil {
		return metav1.ListOptions{}, false, err
	}
	if errs := metainternalversionvalidation.ValidateListOptions(internalListOptions, true); len(errs) > 0 {
		return metav1.ListOptions{}, false, nil
	}

	watchListOptions := listOptions
	// this is our legacy case, the cache ignores LIMIT for
	// ResourceVersion == 0 and RVM=unset|NotOlderThan
	if listOptions.Limit > 0 && listOptions.ResourceVersion != "0" {
		return metav1.ListOptions{}, false, nil
	}
	watchListOptions.Limit = 0

	// to ensure that we can create a watch-list request that returns
	// semantically equivalent data for the given listOptions,
	// we need to validate that the RVM for the list is supported by watch-list requests.
	if listOptions.ResourceVersionMatch == metav1.ResourceVersionMatchExact {
		return metav1.ListOptions{}, false, nil
	}
	watchListOptions.ResourceVersionMatch = metav1.ResourceVersionMatchNotOlderThan

	watchListOptions.Watch = true
	watchListOptions.AllowWatchBookmarks = true
	watchListOptions.SendInitialEvents = ptr.To(true)

	internalWatchListOptions := &metainternalversion.ListOptions{}
	if err := scheme.Convert(&watchListOptions, internalWatchListOptions, nil); err != nil {
		return metav1.ListOptions{}, false, err
	}
	if errs := metainternalversionvalidation.ValidateListOptions(internalWatchListOptions, true); len(errs) > 0 {
		return metav1.ListOptions{}, false, nil
	}

	return watchListOptions, true, nil
}
