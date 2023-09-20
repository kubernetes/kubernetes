/*
Copyright (c) 2014 VMware, Inc. All Rights Reserved.

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

package soap

import (
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"

	"github.com/vmware/govmomi/vim25/types"
)

type regularError struct {
	err error
}

func (r regularError) Error() string {
	return r.err.Error()
}

type soapFaultError struct {
	fault *Fault
}

func (s soapFaultError) Error() string {
	msg := s.fault.String

	if msg == "" {
		if s.fault.Detail.Fault == nil {
			msg = "unknown fault"
		} else {
			msg = reflect.TypeOf(s.fault.Detail.Fault).Name()
		}
	}

	return fmt.Sprintf("%s: %s", s.fault.Code, msg)
}

func (s soapFaultError) MarshalJSON() ([]byte, error) {
	out := struct {
		Fault *Fault
	}{
		Fault: s.fault,
	}
	return json.Marshal(out)
}

type vimFaultError struct {
	fault types.BaseMethodFault
}

func (v vimFaultError) Error() string {
	typ := reflect.TypeOf(v.fault)
	for typ.Kind() == reflect.Ptr {
		typ = typ.Elem()
	}

	return typ.Name()
}

func (v vimFaultError) Fault() types.BaseMethodFault {
	return v.fault
}

func Wrap(err error) error {
	switch err.(type) {
	case regularError:
		return err
	case soapFaultError:
		return err
	case vimFaultError:
		return err
	}

	return WrapRegularError(err)
}

func WrapRegularError(err error) error {
	return regularError{err}
}

func IsRegularError(err error) bool {
	_, ok := err.(regularError)
	return ok
}

func ToRegularError(err error) error {
	return err.(regularError).err
}

func WrapSoapFault(f *Fault) error {
	return soapFaultError{f}
}

func IsSoapFault(err error) bool {
	_, ok := err.(soapFaultError)
	return ok
}

func ToSoapFault(err error) *Fault {
	return err.(soapFaultError).fault
}

func WrapVimFault(v types.BaseMethodFault) error {
	return vimFaultError{v}
}

func IsVimFault(err error) bool {
	_, ok := err.(vimFaultError)
	return ok
}

func ToVimFault(err error) types.BaseMethodFault {
	return err.(vimFaultError).fault
}

func IsCertificateUntrusted(err error) bool {
	// golang 1.20 introduce a new type to wrap 509 errors. So instead of
	// casting the type, now we check the error chain contains the
	// x509 error or not.
	x509UnknownAuthorityErr := &x509.UnknownAuthorityError{}
	ok := errors.As(err, x509UnknownAuthorityErr)
	if ok {
		return true
	}

	x509HostNameErr := &x509.HostnameError{}
	ok = errors.As(err, x509HostNameErr)
	if ok {
		return true
	}

	// The err variable may not be a special type of x509 or HTTP
	// error that can be validated by a type assertion. The err variable is
	// in fact be an *errors.errorString.

	msgs := []string{
		"certificate is not trusted",
		"certificate signed by unknown authority",
	}

	for _, msg := range msgs {
		if strings.HasSuffix(err.Error(), msg) {
			return true
		}
	}

	return false
}
