// Copyright 2014 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

module("tabIndex values");

test("tabIndex is 0 or -1 passes the audit", function() {
  var rule = axs.AuditRules.getRule("tabIndexGreaterThanZero");
  var fixture = document.getElementById('qunit-fixture');

  var listItem = document.createElement("li");
  var heading = document.createElement("h2");
  fixture.appendChild(listItem);
  fixture.appendChild(heading);
  listItem.tabIndex = -1;
  heading.tabIndex = 0;

  var output = rule.run({ scope: fixture });
  equal(output.result, axs.constants.AuditResult.PASS);
});

test("tabIndex with a positive integer fails the audit", function() {
  var rule = axs.AuditRules.getRule("tabIndexGreaterThanZero");
  var fixture = document.getElementById('qunit-fixture');

  var listItem = document.createElement("li");
  fixture.appendChild(listItem);
  listItem.tabIndex = 1;

  var output = rule.run({ scope: fixture });
  equal(output.result, axs.constants.AuditResult.FAIL);
  deepEqual(output.elements, [listItem]);
});
