/*
   Copyright The containerd Authors.

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

// Package cause is used to define root causes for errors
// common to errors packages like grpc and http.
package cause

import "fmt"

type ErrUnexpectedStatus struct {
	Status int
}

const UnexpectedStatusPrefix = "unexpected status "

func (e ErrUnexpectedStatus) Error() string {
	return fmt.Sprintf("%s%d", UnexpectedStatusPrefix, e.Status)
}

func (ErrUnexpectedStatus) Unknown() {}
