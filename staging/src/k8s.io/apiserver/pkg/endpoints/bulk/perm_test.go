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

package bulk

import (
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	bulkapi "k8s.io/apiserver/pkg/apis/bulk"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/request"
)

type fakeAuthorizer struct {
	calls  []authorizer.Attributes
	auth   bool
	reason string
	err    error
}

func (f *fakeAuthorizer) Authorize(a authorizer.Attributes) (bool, string, error) {
	f.calls = append(f.calls, a)
	return f.auth, f.reason, f.err
}

func TestCheckWatchPermissions(t *testing.T) {
	assert := assert.New(t)

	fa := fakeAuthorizer{}
	m := APIManager{
		Authorizer:        &fa,
		PermissionRecheck: 10 * time.Millisecond,
	}

	usr := user.DefaultInfo{Name: "bulk api user"}
	ctx := request.NewContext()
	ctx = request.WithUser(ctx, &usr)

	selector := bulkapi.ResourceSelector{
		Group:    "the-group",
		Resource: "the-resource",
		Version:  "v3",
	}

	cf := m.newAuthorizationCheckerForWatch(ctx, selector)

	theError := errors.New("The Error")
	fa.err = theError

	err := cf()
	assert.EqualError(err, theError.Error())
	for i := 0; i < 5; i++ {
		_ = cf()
	}
	assert.Len(fa.calls, 1, "authorization result should be memoized")
	assert.Equal(fa.calls[0].GetUser().GetName(), usr.Name, "should get userInfo from context")
	assert.True(fa.calls[0].IsResourceRequest())
	assert.Equal(fa.calls[0].GetVerb(), "watch")
	assert.Equal(fa.calls[0].GetAPIGroup(), "the-group")
	assert.Equal(fa.calls[0].GetAPIVersion(), "v3")
	assert.Equal(fa.calls[0].GetResource(), "the-resource")

	time.Sleep(30 * time.Millisecond)
	fa.auth = false
	fa.reason = "some reason"
	fa.err = nil

	err = cf()
	assert.True(apierrors.IsForbidden(err))
	assert.Len(fa.calls, 2, "authorization result should be renewed")
}
