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

module('Scroll area', {
  setup: function() {
    var fixture = document.createElement('div');

    fixture.style.top = '0';
    fixture.style.left = '0';
    fixture.style.overflow = 'scroll';

    document.getElementById('qunit-fixture').appendChild(fixture);
    this.fixture_ = fixture;

    document.getElementById('qunit-fixture').style.top = 0;
    document.getElementById('qunit-fixture').style.left = 0;
  },
  teardown: function() {
    document.getElementById('qunit-fixture').style.removeProperty('top');
    document.getElementById('qunit-fixture').style.removeProperty('left');
  }
});
test('Inside scroll area = no problem', function() {
  var input = document.createElement('input');
  this.fixture_.appendChild(input);

  equal(axs.utils.elementIsOutsideScrollArea(input), false);
});
test('Outside scroll area = bad', function() {
  var input = document.createElement('input');
  this.fixture_.appendChild(input);
  input.style.top = '-1000px';
  input.style.left = '-1000px';
  input.style.position = 'absolute';
  equal(axs.utils.elementIsOutsideScrollArea(input), true);
});
test('In scroll area for element with overflow:auto or overflow:scroll = ok', function() {
  var longDiv = document.createElement('div');
  this.fixture_.appendChild(longDiv);
  longDiv.style.overflow = 'auto';
  longDiv.style.position = 'absolute';
  longDiv.style.left = '0';
  longDiv.style.top = '0';
  longDiv.style.height = '1000px';
  for (var i = 0; i < 1000; i++) {
    var filler = document.createElement('div');
    filler.innerText = 'spam';
    longDiv.appendChild(filler);
  }
  var input = document.createElement('input');
  longDiv.appendChild(input);
  equal(axs.utils.elementIsOutsideScrollArea(input), false);

  longDiv.style.overflow = 'scroll';
  equal(axs.utils.elementIsOutsideScrollArea(input), false);
});
test('In scroll area for element but that element is not inside scroll area = bad', function() {
  var longDiv = document.createElement('div');
  this.fixture_.appendChild(longDiv);
  longDiv.style.overflow = 'auto';
  longDiv.style.position = 'absolute';
  longDiv.style.left = '-10000px';
  longDiv.style.top = '-10000px';
  longDiv.style.height = '1000px';
  longDiv.style.width = '1000px';
  for (var i = 0; i < 1000; i++) {
    var filler = document.createElement('div');
    filler.innerText = 'spam';
    longDiv.appendChild(filler);
  }
  var input = document.createElement('input');
  longDiv.appendChild(input);
  equal(axs.utils.elementIsOutsideScrollArea(input), true);
});
test('Clipped by element = bad even if inside scroll area', function() {
  this.fixture_.innerHTML =
    '<style>\n' +
    'div {\n' +
    '    border: 1px solid #009;\n' +
    '    padding: 20px;\n' +
    '}\n' +
    'button {\n' +
    '    margin: 20px;\n' +
    '    display: block;\n' +
    '}\n' +
    '.container {\n' +
    '    overflow: hidden;\n' +
    '    position: relative;\n' +
    '    left: 400px;\n' +
    '}\n' +
    '.b2 {\n' +
    '    position: relative;\n' +
    '    left: -400px;\n' +
    '}\n' +
    '</style>\n' +
    '<div class="container">\n' +
    '    <button class="b1">This button is offscreen</button>\n' +
    '    <button class="b2">This button is onscreen but clipped.</button>\n' +
    '</div>';
  var button = document.querySelector('.b2');
  equal(axs.utils.elementIsOutsideScrollArea(button), true);

  var container = document.querySelector('.container');
  container.style.overflow = 'scroll';
  equal(axs.utils.elementIsOutsideScrollArea(button), true);

  var container = document.querySelector('.container');
  container.style.overflow = 'auto';
  equal(axs.utils.elementIsOutsideScrollArea(button), true);

  var container = document.querySelector('.container');
  container.style.overflow = 'visible';
  equal(axs.utils.elementIsOutsideScrollArea(button), false);
});
