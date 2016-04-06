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

package pool

var poolNameHelp = `
POOL may be an absolute or relative path to a resource pool or a (clustered)
compute host. If it resolves to a compute host, the associated root resource
pool is returned. If a relative path is specified, it is resolved with respect
to the current datacenter's "host" folder (i.e. /ha-datacenter/host).

Paths to nested resource pools must traverse through the root resource pool of
the selected compute host, i.e. "compute-host/Resources/nested-pool".

The same globbing rules that apply to the "ls" command apply here. For example,
POOL may be specified as "*/Resources/*" to expand to all resource pools that
are nested one level under the root resource pool, on all (clustered) compute
hosts in the current datacenter.`

var poolCreateHelp = `
POOL may be an absolute or relative path to a resource pool. The parent of the
specified POOL must be an existing resource pool. If a relative path is
specified, it is resolved with respect to the current datacenter's "host"
folder (i.e. /ha-datacenter/host). The basename of the specified POOL is used
as the name for the new resource pool.

The same globbing rules that apply to the "ls" command apply here. For example,
the path to the parent resource pool in POOL may be specified as "*/Resources"
to expand to the root resource pools on all (clustered) compute hosts in the
current datacenter.

For example:
  */Resources/test             Create resource pool "test" on all (clustered)
                               compute hosts in the current datacenter.
  somehost/Resources/*/nested  Create resource pool "nested" in every
                               resource pool that is a direct descendant of
                               the root resource pool on "somehost".`
