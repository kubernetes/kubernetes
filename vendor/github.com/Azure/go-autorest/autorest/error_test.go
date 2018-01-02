package autorest

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"fmt"
	"net/http"
	"reflect"
	"regexp"
	"testing"
)

func TestNewErrorWithError_AssignsPackageType(t *testing.T) {
	e := NewErrorWithError(fmt.Errorf("original"), "packageType", "method", nil, "message")

	if e.PackageType != "packageType" {
		t.Fatalf("autorest: Error failed to set package type -- expected %v, received %v", "packageType", e.PackageType)
	}
}

func TestNewErrorWithError_AssignsMethod(t *testing.T) {
	e := NewErrorWithError(fmt.Errorf("original"), "packageType", "method", nil, "message")

	if e.Method != "method" {
		t.Fatalf("autorest: Error failed to set method -- expected %v, received %v", "method", e.Method)
	}
}

func TestNewErrorWithError_AssignsMessage(t *testing.T) {
	e := NewErrorWithError(fmt.Errorf("original"), "packageType", "method", nil, "message")

	if e.Message != "message" {
		t.Fatalf("autorest: Error failed to set message -- expected %v, received %v", "message", e.Message)
	}
}

func TestNewErrorWithError_AssignsUndefinedStatusCodeIfRespNil(t *testing.T) {
	e := NewErrorWithError(nil, "packageType", "method", nil, "message")
	if e.StatusCode != UndefinedStatusCode {
		t.Fatalf("autorest: Error failed to set status code -- expected %v, received %v", UndefinedStatusCode, e.StatusCode)
	}
}

func TestNewErrorWithError_AssignsStatusCode(t *testing.T) {
	e := NewErrorWithError(fmt.Errorf("original"), "packageType", "method", &http.Response{
		StatusCode: http.StatusBadRequest,
		Status:     http.StatusText(http.StatusBadRequest)}, "message")

	if e.StatusCode != http.StatusBadRequest {
		t.Fatalf("autorest: Error failed to set status code -- expected %v, received %v", http.StatusBadRequest, e.StatusCode)
	}
}

func TestNewErrorWithError_AcceptsArgs(t *testing.T) {
	e := NewErrorWithError(fmt.Errorf("original"), "packageType", "method", nil, "message %s", "arg")

	if matched, _ := regexp.MatchString(`.*arg.*`, e.Message); !matched {
		t.Fatalf("autorest: Error failed to apply message arguments -- expected %v, received %v",
			`.*arg.*`, e.Message)
	}
}

func TestNewErrorWithError_AssignsError(t *testing.T) {
	err := fmt.Errorf("original")
	e := NewErrorWithError(err, "packageType", "method", nil, "message")

	if e.Original != err {
		t.Fatalf("autorest: Error failed to set error -- expected %v, received %v", err, e.Original)
	}
}

func TestNewErrorWithResponse_ContainsStatusCode(t *testing.T) {
	e := NewErrorWithResponse("packageType", "method", &http.Response{
		StatusCode: http.StatusBadRequest,
		Status:     http.StatusText(http.StatusBadRequest)}, "message")

	if e.StatusCode != http.StatusBadRequest {
		t.Fatalf("autorest: Error failed to set status code -- expected %v, received %v", http.StatusBadRequest, e.StatusCode)
	}
}

func TestNewErrorWithResponse_nilResponse_ReportsUndefinedStatusCode(t *testing.T) {
	e := NewErrorWithResponse("packageType", "method", nil, "message")

	if e.StatusCode != UndefinedStatusCode {
		t.Fatalf("autorest: Error failed to set status code -- expected %v, received %v", UndefinedStatusCode, e.StatusCode)
	}
}

func TestNewErrorWithResponse_Forwards(t *testing.T) {
	e1 := NewError("packageType", "method", "message %s", "arg")
	e2 := NewErrorWithResponse("packageType", "method", nil, "message %s", "arg")

	if !reflect.DeepEqual(e1, e2) {
		t.Fatal("autorest: NewError did not return an error equivelent to NewErrorWithError")
	}
}

func TestNewErrorWithError_Forwards(t *testing.T) {
	e1 := NewError("packageType", "method", "message %s", "arg")
	e2 := NewErrorWithError(nil, "packageType", "method", nil, "message %s", "arg")

	if !reflect.DeepEqual(e1, e2) {
		t.Fatal("autorest: NewError did not return an error equivelent to NewErrorWithError")
	}
}

func TestNewErrorWithError_DoesNotWrapADetailedError(t *testing.T) {
	e1 := NewError("packageType1", "method1", "message1 %s", "arg1")
	e2 := NewErrorWithError(e1, "packageType2", "method2", nil, "message2 %s", "arg2")

	if !reflect.DeepEqual(e1, e2) {
		t.Fatalf("autorest: NewErrorWithError incorrectly wrapped a DetailedError -- expected %v, received %v", e1, e2)
	}
}

func TestNewErrorWithError_WrapsAnError(t *testing.T) {
	e1 := fmt.Errorf("Inner Error")
	var e2 interface{} = NewErrorWithError(e1, "packageType", "method", nil, "message")

	if _, ok := e2.(DetailedError); !ok {
		t.Fatalf("autorest: NewErrorWithError failed to wrap a standard error -- received %T", e2)
	}
}

func TestDetailedError(t *testing.T) {
	err := fmt.Errorf("original")
	e := NewErrorWithError(err, "packageType", "method", nil, "message")

	if matched, _ := regexp.MatchString(`.*original.*`, e.Error()); !matched {
		t.Fatalf("autorest: Error#Error failed to return original error message -- expected %v, received %v",
			`.*original.*`, e.Error())
	}
}

func TestDetailedErrorConstainsPackageType(t *testing.T) {
	e := NewErrorWithError(fmt.Errorf("original"), "packageType", "method", nil, "message")

	if matched, _ := regexp.MatchString(`.*packageType.*`, e.Error()); !matched {
		t.Fatalf("autorest: Error#String failed to include PackageType -- expected %v, received %v",
			`.*packageType.*`, e.Error())
	}
}

func TestDetailedErrorConstainsMethod(t *testing.T) {
	e := NewErrorWithError(fmt.Errorf("original"), "packageType", "method", nil, "message")

	if matched, _ := regexp.MatchString(`.*method.*`, e.Error()); !matched {
		t.Fatalf("autorest: Error#String failed to include Method -- expected %v, received %v",
			`.*method.*`, e.Error())
	}
}

func TestDetailedErrorConstainsMessage(t *testing.T) {
	e := NewErrorWithError(fmt.Errorf("original"), "packageType", "method", nil, "message")

	if matched, _ := regexp.MatchString(`.*message.*`, e.Error()); !matched {
		t.Fatalf("autorest: Error#String failed to include Message -- expected %v, received %v",
			`.*message.*`, e.Error())
	}
}

func TestDetailedErrorConstainsStatusCode(t *testing.T) {
	e := NewErrorWithError(fmt.Errorf("original"), "packageType", "method", &http.Response{
		StatusCode: http.StatusBadRequest,
		Status:     http.StatusText(http.StatusBadRequest)}, "message")

	if matched, _ := regexp.MatchString(`.*400.*`, e.Error()); !matched {
		t.Fatalf("autorest: Error#String failed to include Status Code -- expected %v, received %v",
			`.*400.*`, e.Error())
	}
}

func TestDetailedErrorConstainsOriginal(t *testing.T) {
	e := NewErrorWithError(fmt.Errorf("original"), "packageType", "method", nil, "message")

	if matched, _ := regexp.MatchString(`.*original.*`, e.Error()); !matched {
		t.Fatalf("autorest: Error#String failed to include Original error -- expected %v, received %v",
			`.*original.*`, e.Error())
	}
}

func TestDetailedErrorSkipsOriginal(t *testing.T) {
	e := NewError("packageType", "method", "message")

	if matched, _ := regexp.MatchString(`.*Original.*`, e.Error()); matched {
		t.Fatalf("autorest: Error#String included missing Original error -- unexpected %v, received %v",
			`.*Original.*`, e.Error())
	}
}
