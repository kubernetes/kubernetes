/*
Copyright 2014 Google Inc. All rights reserved.

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
Package cmd provides one stop shopping for a command line executable to bind the correct flags,
build the client config, and create a working client. The code for usage looks like this:

    clientBuilder := clientcmd.NewBuilder(clientcmd.NewDefaultAuthLoader())
    clientBuilder.BindFlags(cmds.PersistentFlags())
    apiClient, err := clientBuilder.Client()
*/
package clientcmd
