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

package progress

/*
The progress package contains functionality to deal with progress reporting.
The functionality is built to serve progress reporting for infrastructure
operations when talking the vSphere API, but is generic enough to be used
elsewhere.

At the core of this progress reporting API lies the Sinker interface. This
interface is implemented by any object that can act as a sink for progress
reports. Callers of the Sink() function receives a send-only channel for
progress reports. They are responsible for closing the channel when done.
This semantic makes it easy to keep track of multiple progress report channels;
they are only created when Sink() is called and assumed closed when any
function that receives a Sinker parameter returns.
*/
