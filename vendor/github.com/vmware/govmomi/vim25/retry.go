/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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

package vim25

import (
	"net"
	"net/url"
	"time"

	"github.com/vmware/govmomi/vim25/soap"
	"golang.org/x/net/context"
)

type RetryFunc func(err error) (retry bool, delay time.Duration)

// TemporaryNetworkError returns a RetryFunc that retries up to a maximum of n
// times, only if the error returned by the RoundTrip function is a temporary
// network error (for example: a connect timeout).
func TemporaryNetworkError(n int) RetryFunc {
	return func(err error) (retry bool, delay time.Duration) {
		var nerr net.Error
		var ok bool

		// Never retry if this is not a network error.
		switch rerr := err.(type) {
		case *url.Error:
			if nerr, ok = rerr.Err.(net.Error); !ok {
				return false, 0
			}
		case net.Error:
			nerr = rerr
		default:
			return false, 0
		}

		if !nerr.Temporary() {
			return false, 0
		}

		// Don't retry if we're out of tries.
		if n--; n <= 0 {
			return false, 0
		}

		return true, 0
	}
}

type retry struct {
	roundTripper soap.RoundTripper

	// fn is a custom function that is called when an error occurs.
	// It returns whether or not to retry, and if so, how long to
	// delay before retrying.
	fn RetryFunc
}

// Retry wraps the specified soap.RoundTripper and invokes the
// specified RetryFunc. The RetryFunc returns whether or not to
// retry the call, and if so, how long to wait before retrying. If
// the result of this function is to not retry, the original error
// is returned from the RoundTrip function.
func Retry(roundTripper soap.RoundTripper, fn RetryFunc) soap.RoundTripper {
	r := &retry{
		roundTripper: roundTripper,
		fn:           fn,
	}

	return r
}

func (r *retry) RoundTrip(ctx context.Context, req, res soap.HasFault) error {
	var err error

	for {
		err = r.roundTripper.RoundTrip(ctx, req, res)
		if err == nil {
			break
		}

		// Invoke retry function to see if another attempt should be made.
		if retry, delay := r.fn(err); retry {
			time.Sleep(delay)
			continue
		}

		break
	}

	return err
}
