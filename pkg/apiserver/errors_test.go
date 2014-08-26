/*
Copyright 2014 Google Inc. All rights reserved.

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

package apiserver

import (
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	apierrors "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
)

func TestErrorNew(t *testing.T) {
	err := NewAlreadyExistsErr("test", "1")
	if !IsAlreadyExists(err) {
		t.Errorf("expected to be already_exists")
	}
	if IsConflict(err) {
		t.Errorf("expected to not be confict")
	}
	if IsNotFound(err) {
		t.Errorf(fmt.Sprintf("expected to not be %s", api.StatusReasonNotFound))
	}
	if IsInvalid(err) {
		t.Errorf("expected to not be invalid")
	}

	if !IsConflict(NewConflictErr("test", "2", errors.New("message"))) {
		t.Errorf("expected to be conflict")
	}
	if !IsNotFound(NewNotFoundErr("test", "3")) {
		t.Errorf("expected to be not found")
	}
	if !IsInvalid(NewInvalidErr("test", "2", nil)) {
		t.Errorf("expected to be invalid")
	}
}

func TestNewInvalidErr(t *testing.T) {
	testCases := []struct {
		Err     apierrors.ValidationError
		Details *api.StatusDetails
	}{
		{
			apierrors.NewDuplicate("field[0].name", "bar"),
			&api.StatusDetails{
				Kind: "kind",
				ID:   "name",
				Causes: []api.StatusCause{{
					Type:  api.CauseTypeFieldValueDuplicate,
					Field: "field[0].name",
				}},
			},
		},
		{
			apierrors.NewInvalid("field[0].name", "bar"),
			&api.StatusDetails{
				Kind: "kind",
				ID:   "name",
				Causes: []api.StatusCause{{
					Type:  api.CauseTypeFieldValueInvalid,
					Field: "field[0].name",
				}},
			},
		},
		{
			apierrors.NewNotFound("field[0].name", "bar"),
			&api.StatusDetails{
				Kind: "kind",
				ID:   "name",
				Causes: []api.StatusCause{{
					Type:  api.CauseTypeFieldValueNotFound,
					Field: "field[0].name",
				}},
			},
		},
		{
			apierrors.NewNotSupported("field[0].name", "bar"),
			&api.StatusDetails{
				Kind: "kind",
				ID:   "name",
				Causes: []api.StatusCause{{
					Type:  api.CauseTypeFieldValueNotSupported,
					Field: "field[0].name",
				}},
			},
		},
		{
			apierrors.NewRequired("field[0].name", "bar"),
			&api.StatusDetails{
				Kind: "kind",
				ID:   "name",
				Causes: []api.StatusCause{{
					Type:  api.CauseTypeFieldValueRequired,
					Field: "field[0].name",
				}},
			},
		},
	}
	for i := range testCases {
		vErr, expected := testCases[i].Err, testCases[i].Details
		expected.Causes[0].Message = vErr.Error()
		err := NewInvalidErr("kind", "name", apierrors.ErrorList{vErr})
		status := errToAPIStatus(err)
		if status.Code != 422 || status.Reason != api.StatusReasonInvalid {
			t.Errorf("unexpected status: %#v", status)
		}
		if !reflect.DeepEqual(expected, status.Details) {
			t.Errorf("expected %#v, got %#v", expected, status.Details)
		}
	}
}

func Test_errToAPIStatus(t *testing.T) {
	err := &apiServerError{}
	status := errToAPIStatus(err)
	if status.Reason != api.StatusReasonUnknown || status.Status != api.StatusFailure {
		t.Errorf("unexpected status object: %#v", status)
	}
}

func Test_reasonForError(t *testing.T) {
	if e, a := api.StatusReasonUnknown, reasonForError(nil); e != a {
		t.Errorf("unexpected reason type: %#v", a)
	}
}
