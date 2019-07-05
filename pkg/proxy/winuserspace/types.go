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

package winuserspace

import (
	"fmt"

	"k8s.io/apimachinery/pkg/types"
)

// ServicePortPortalName carries a namespace + name + portname + portalip.  This is the unique
// identifier for a windows service port portal.
type ServicePortPortalName struct {
	types.NamespacedName
	Port         string
	PortalIPName string
}

func (spn ServicePortPortalName) String() string {
	return fmt.Sprintf("%s:%s:%s", spn.NamespacedName.String(), spn.Port, spn.PortalIPName)
}
