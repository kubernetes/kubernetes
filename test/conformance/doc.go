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

/*
This stand-alone package is utilized for dynamically generating/maintaining a list of
conformance tests. It utilizes a two step approach:
  - The test binary is built
  - The test binary is run in dry mode with a custom ginkgo reporter dumping out
    types.SpecSummary objects which contain full test names and file/code information.
  - The SpecSummary information is parsed to get file/line info on Conformance tests and
    then we use a simplified AST parser to grab the comments above the test.

Due to the complicated nature of how tests can be declared/wrapped in various contexts,
this approach is much simpler to maintain than a pure-AST parser and allows us to easily
capture the full test names/locations of the tests using the pre-existing ginkgo logic.
*/
package main
