/*
Copyright 2020 The Kubernetes Authors.

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

package datapol

import (
	"crypto/x509"
	"net/http"
	"reflect"
)

var (
	httpHeader      = reflect.TypeOf(http.Header{})
	httpCookie      = reflect.TypeOf(http.Cookie{})
	x509Certificate = reflect.TypeOf(x509.Certificate{})
)

// GlobalDatapolicyMapping returns the list of sensitive datatypes are embedded
// in types not native to Kubernetes.
func GlobalDatapolicyMapping(v interface{}) []string {
	return byType(reflect.TypeOf(v))
}

func byType(t reflect.Type) []string {
	switch t {
	case httpHeader:
		return []string{"password", "token"}
	case httpCookie:
		return []string{"token"}
	case x509Certificate:
		return []string{"security-key"}
	default:
		return nil
	}
}
