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

/*
Package vim25 provides a minimal client implementation to use with other
packages in the vim25 tree. The code in this package intentionally does not
take any dependendies outside the vim25 tree.

The client implementation in this package embeds the soap.Client structure.
Additionally, it stores the value of the session's ServiceContent object. This
object stores references to a variety of subsystems, such as the root property
collector, the session manager, and the search index. The client is fully
functional after serialization and deserialization, without the need for
additional requests for initialization.
*/
package vim25
