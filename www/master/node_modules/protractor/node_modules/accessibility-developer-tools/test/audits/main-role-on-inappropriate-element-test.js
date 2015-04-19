// Copyright 2013 Google Inc.
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

module('MainRoleOnInappropriateElement');

test('No role=main -> no relevant elements', function() {
  var fixture = document.getElementById('qunit-fixture');
  var div = document.createElement('div');
  fixture.appendChild(div);

  deepEqual(
    axs.AuditRules.getRule('mainRoleOnInappropriateElement').run({ scope: fixture }),
    { result: axs.constants.AuditResult.NA }
  );
});

test('role=main on empty element === fail', function() {
  var fixture = document.getElementById('qunit-fixture');
  var div = document.createElement('div');
  div.setAttribute('role', 'main');
  fixture.appendChild(div);

  deepEqual(
    axs.AuditRules.getRule('mainRoleOnInappropriateElement').run({ scope: fixture }),
    { elements: [div], result: axs.constants.AuditResult.FAIL }
  );
});

test('role=main on element with textContent < 50 characters === pass', function() {
  var fixture = document.getElementById('qunit-fixture');
  var div = document.createElement('div');
  div.setAttribute('role', 'main');
  div.textContent = 'Lorem ipsum dolor sit amet.';
  fixture.appendChild(div);

  deepEqual(
    axs.AuditRules.getRule('mainRoleOnInappropriateElement').run({ scope: fixture }),
    { elements: [div], result: axs.constants.AuditResult.FAIL }
  );
});

test('role=main on element with textContent >= 50 characters === pass', function() {
  var fixture = document.getElementById('qunit-fixture');
  var div = document.createElement('div');
  div.setAttribute('role', 'main');
  div.textContent = 'Lorem ipsum dolor sit amet, consectetur cras amet.';
  fixture.appendChild(div);

  deepEqual(
    axs.AuditRules.getRule('mainRoleOnInappropriateElement').run({ scope: fixture }),
    { elements: [], result: axs.constants.AuditResult.PASS }
  );
});

test('role=main on inline element === fail', function() {
  var fixture = document.getElementById('qunit-fixture');
  var span = document.createElement('span');
  span.setAttribute('role', 'main');
  span.textContent = 'Lorem ipsum dolor sit amet, consectetur cras amet.';
  fixture.appendChild(span);

  deepEqual(
    axs.AuditRules.getRule('mainRoleOnInappropriateElement').run({ scope: fixture }),
    { elements: [span], result: axs.constants.AuditResult.FAIL });
});
