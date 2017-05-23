/*
Copyright (c) 2014-2017 VMware, Inc. All Rights Reserved.

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
Package find implements inventory listing and searching.

The Finder is an alternative to the object.SearchIndex FindByInventoryPath() and FindChild() methods.
SearchIndex.FindByInventoryPath requires an absolute path, whereas the Finder also supports relative paths
and patterns via path.Match.
SearchIndex.FindChild requires a parent to find the child, whereas the Finder also supports an ancestor via
recursive object traversal.

The various Finder methods accept a "path" argument, which can absolute or relative to the Folder for the object type.
The Finder supports two modes, "list" and "find".  The "list" mode behaves like the "ls" command, only searching within
the immediate path.  The "find" mode behaves like the "find" command, with the search starting at the immediate path but
also recursing into sub Folders relative to the Datacenter.  The default mode is "list" if the given path contains a "/",
otherwise "find" mode is used.

The exception is to use a "..." wildcard with a path to find all objects recursively underneath any root object.
For example: VirtualMachineList("/DC1/...")

See also: https://github.com/vmware/govmomi/blob/master/govc/README.md#usage
*/
package find
