/*
Copyright 2018 The Kubernetes Authors.

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

package utils

import (
	"fmt"
	"math/rand"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilwait "k8s.io/apimachinery/pkg/util/wait"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	apiextensionsv1beta1client "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset/typed/apiextensions/v1beta1"
)

var (
	// crdEstablishedNumRetries is the maximum number of attempts that a CRD will be given to establish
	crdEstablishedNumRetries = 5
	// crdEstablishedRetryInterval is the amount of time that will pass between retries
	crdEstablishedRetryInterval = 500 * time.Millisecond
	// crdEstablishedRetryFactor is the scalar by which the prevous interval will be increased
	crdEstablishedRetryFactor = 1.0
)

// WaitForEstablished  is a utility function that will ensure that an already created
// CustomResourceDefinition is ready for use by a Kubernetes cluster. If the CustomResourceDefinition
// cannot be established within a reasonable amount of retries, an error will be returned.
func WaitForEstablished(
	i apiextensionsv1beta1client.ApiextensionsV1beta1Interface,
	crd *apiextensionsv1beta1.CustomResourceDefinition,
	stopCh <-chan struct{},
) error {
	err := utilwait.ExponentialBackoffUntil(utilwait.Backoff{
		Factor:   crdEstablishedRetryFactor, // Even though we are using a factor of 1, ExponentialBackoff is preferred over PollImmediate as it provides jitter.
		Steps:    crdEstablishedNumRetries,
		Jitter:   rand.Float64(),
		Duration: crdEstablishedRetryInterval,
	}, func() (bool, error) {
		// Attempt to retrieve the CRD that was either already present or just created.
		crd, err := i.CustomResourceDefinitions().Get(crd.GetName(), metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		// Test if CRD is established. If it is not, attempt
		// to find a state in which it will never be established
		// and fail fast.
		for _, cond := range crd.Status.Conditions {
			switch cond.Type {
			case apiextensionsv1beta1.Established:
				// This is the state we are looking for.
				if cond.Status == apiextensionsv1beta1.ConditionTrue {
					return true, nil
				}
			case apiextensionsv1beta1.NamesAccepted:
				// If we have reached this state, the CRD will never become
				// established
				if cond.Status == apiextensionsv1beta1.ConditionFalse {
					return false, fmt.Errorf("due to the naming conflict %s, the CustomResourceDefinition %s will never become established", cond.Reason, crd.GetName())
				}
			}
		}
		return false, nil
	}, stopCh)

	if err == utilwait.ErrWaitTimeout {
		return fmt.Errorf("the CustomResourceDefinition %s was not established within a reasonable amount of time", crd.GetName())
	}
	return err
}
