/*
Copyright 2019 The Kubernetes Authors.

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

package flowcontrol

import (
	"fmt"

	fcfmt "k8s.io/apiserver/pkg/util/flowcontrol/format"
)

var _ fmt.GoStringer = RequestDigest{}

// GoString produces a golang source expression of the value.
func (rd RequestDigest) GoString() string {
	return fmt.Sprintf("RequestDigest{RequestInfo: %#+v, User: %#+v}", rd.RequestInfo, rd.User)
}

var _ fmt.GoStringer = (*priorityLevelState)(nil)

// GoString produces a golang source expression of the value.
func (pls *priorityLevelState) GoString() string {
	if pls == nil {
		return "nil"
	}
	return fmt.Sprintf("&priorityLevelState{pl:%s, qsCompleter:%#+v, queues:%#+v, quiescing:%#v, numPending:%d}", fcfmt.Fmt(pls.pl), pls.qsCompleter, pls.queues, pls.quiescing, pls.numPending)
}
