/*
Copyright 2014 The Kubernetes Authors.

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

package validate

import (
	"context"
	"net"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	netutils "k8s.io/utils/net"
)

// IPSloppy verifies that the specified value is a valid IP address, but allows
// leading zeros on each octet value.  This should not be used for new APIs.
func IPSloppy[T ~string](ctx context.Context, op operation.Operation, fldPath *field.Path, value, _ *T) field.ErrorList {
	vt := *value
	vs := string(vt)
	_, errs := ipSloppy(ctx, op, fldPath, &vs, nil)
	return errs
}

func ipSloppy(ctx context.Context, op operation.Operation, fldPath *field.Path, value, _ *string) (net.IP, field.ErrorList) {
	if value == nil {
		return nil, nil
	}
	ip := netutils.ParseIPSloppy(*value)
	if ip == nil {
		return nil, field.ErrorList{
			field.Invalid(fldPath, *value, "must be a valid IP address (e.g. 10.9.8.7 or 2001:db8::ffff)").WithOrigin("format=k8s-ip"),
		}
	}
	return ip, nil
}
