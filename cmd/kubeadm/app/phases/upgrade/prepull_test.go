/*
Copyright 2017 The Kubernetes Authors.

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

package upgrade

import (
	"testing"
	"time"

	"github.com/pkg/errors"

	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	//"k8s.io/apimachinery/pkg/util/version"
)

// failedCreatePrepuller is a fake prepuller that errors for kube-controller-manager in the CreateFunc call
type failedCreatePrepuller struct{}

func NewFailedCreatePrepuller() Prepuller {
	return &failedCreatePrepuller{}
}

func (p *failedCreatePrepuller) CreateFunc(component string) error {
	if component == "kube-controller-manager" {
		return errors.New("boo")
	}
	return nil
}

func (p *failedCreatePrepuller) WaitFunc(component string) {}

func (p *failedCreatePrepuller) DeleteFunc(component string) error {
	return nil
}

// foreverWaitPrepuller is a fake prepuller that basically waits "forever" (10 mins, but longer than the 10sec timeout)
type foreverWaitPrepuller struct{}

func NewForeverWaitPrepuller() Prepuller {
	return &foreverWaitPrepuller{}
}

func (p *foreverWaitPrepuller) CreateFunc(component string) error {
	return nil
}

func (p *foreverWaitPrepuller) WaitFunc(component string) {
	time.Sleep(10 * time.Minute)
}

func (p *foreverWaitPrepuller) DeleteFunc(component string) error {
	return nil
}

// failedDeletePrepuller is a fake prepuller that errors for kube-scheduler in the DeleteFunc call
type failedDeletePrepuller struct{}

func NewFailedDeletePrepuller() Prepuller {
	return &failedDeletePrepuller{}
}

func (p *failedDeletePrepuller) CreateFunc(component string) error {
	return nil
}

func (p *failedDeletePrepuller) WaitFunc(component string) {}

func (p *failedDeletePrepuller) DeleteFunc(component string) error {
	if component == "kube-scheduler" {
		return errors.New("boo")
	}
	return nil
}

// goodPrepuller is a fake prepuller that works as expected
type goodPrepuller struct{}

func NewGoodPrepuller() Prepuller {
	return &goodPrepuller{}
}

func (p *goodPrepuller) CreateFunc(component string) error {
	time.Sleep(300 * time.Millisecond)
	return nil
}

func (p *goodPrepuller) WaitFunc(component string) {
	time.Sleep(300 * time.Millisecond)
}

func (p *goodPrepuller) DeleteFunc(component string) error {
	time.Sleep(300 * time.Millisecond)
	return nil
}

func TestPrepullImagesInParallel(t *testing.T) {
	tests := []struct {
		p           Prepuller
		timeout     time.Duration
		expectedErr bool
	}{
		{ // should error out; create failed
			p:           NewFailedCreatePrepuller(),
			timeout:     10 * time.Second,
			expectedErr: true,
		},
		{ // should error out; timeout exceeded
			p:           NewForeverWaitPrepuller(),
			timeout:     10 * time.Second,
			expectedErr: true,
		},
		{ // should error out; delete failed
			p:           NewFailedDeletePrepuller(),
			timeout:     10 * time.Second,
			expectedErr: true,
		},
		{ // should work just fine
			p:           NewGoodPrepuller(),
			timeout:     10 * time.Second,
			expectedErr: false,
		},
	}

	for _, rt := range tests {

		actualErr := PrepullImagesInParallel(rt.p, rt.timeout, append(constants.MasterComponents, constants.Etcd))
		if (actualErr != nil) != rt.expectedErr {
			t.Errorf(
				"failed TestPrepullImagesInParallel\n\texpected error: %t\n\tgot: %t",
				rt.expectedErr,
				(actualErr != nil),
			)
		}
	}
}
