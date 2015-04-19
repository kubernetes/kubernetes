// Copyright 2012 Google Inc.
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

module("Contrast Ratio", {
  setup: function () {
    var fixture = document.createElement('div');
    document.getElementById('qunit-fixture').appendChild(fixture);
    this.fixture_ = fixture;
    this.black_ = {"red": 0, "green": 0, "blue": 0, "alpha": 1};
    this.white_ = {"red": 255, "green": 255, "blue": 255, "alpha": 1};
  }
});
test("Black and white.", function () {
  equal(axs.utils.calculateContrastRatio(this.white_, this.black_), 21);
  equal(axs.utils.calculateContrastRatio(this.black_, this.white_), 21);
});
test("Same color === no contrast.", function () {
  equal(axs.utils.calculateContrastRatio(this.white_, this.white_), 1);
  equal(axs.utils.calculateContrastRatio(this.black_, this.black_), 1);
});
test("Transparent foreground === no contrast.", function () {
  equal(axs.utils.calculateContrastRatio({"red": 0, "green": 0, "blue": 0, "alpha": 0}, this.white_), 1);
});

module("Zero Area", {
  setup: function () {
    var fixture = document.createElement('div');
    document.getElementById('qunit-fixture').appendChild(fixture);
    this.fixture_ = fixture;
  }
});
test("Large element has non-zero area.", function () {
  this.fixture_.style.display = "block";
  this.fixture_.style.width = "500px";
  this.fixture_.style.height = "500px";
  equal(axs.utils.elementHasZeroArea(this.fixture_), false);
});
test("Small element has non-zero area.", function () {
  this.fixture_.style.display = "block";
  this.fixture_.style.width = "1px";
  this.fixture_.style.height = "1px";
  equal(axs.utils.elementHasZeroArea(this.fixture_), false);
});
test("Empty element has zero area.", function () {
  equal(axs.utils.elementHasZeroArea(this.fixture_), true);
});
test("Inline element has non-zero area.", function () {
  this.fixture_.style.display = "inline";
  this.fixture_.appendChild(document.createTextNode('Size!'));
  equal(axs.utils.elementHasZeroArea(this.fixture_), false);
});

module("Transparency", {
  setup: function () {
    var fixture = document.createElement('div');
    document.getElementById('qunit-fixture').appendChild(fixture);
    this.fixture_ = fixture;
  }
});
test("Transparent elements are transparent.", function () {
  this.fixture_.style.opacity = 0;
  equal(axs.utils.elementIsTransparent(this.fixture_), true);
});
test("Hidden, but opaque elements are not transparent.", function () {
  this.fixture_.style.display = 'none';
  this.fixture_.style.opacity = 1;
  equal(axs.utils.elementIsTransparent(this.fixture_), false);
});
test("Non-transparent elements are non-transparent.", function () {
  for (var i = 0.001; i <= 1; i += 0.001) {
    this.fixture_.style.opacity = i;
    equal(axs.utils.elementIsTransparent(this.fixture_), false);
  }
});

module("Control labels", {
  setup: function () {
  }
});
test("Input type=submit has a label.", function() {
  var element = document.createElement("input");
  element.type = "submit";
  equal(axs.utils.hasLabel(element), true);
});
test("A placeholder does not count as a label.", function() {
  var element0 = document.createElement("textarea");
  element0.placeholder = "Your life story";
  equal(axs.utils.hasLabel(element0), false);

  var element1 = document.createElement("input");
  element1.placeholder = "First name";
  equal(axs.utils.hasLabel(element1), false);

  var element2 = document.createElement("input");
  element2.type = "url";
  element2.placeholder = "Homepage";
  equal(axs.utils.hasLabel(element2), false);
});
test('axs.utils.hasLabel() does not crash for element with numeric id attribute', function() {
    var element = document.createElement('input');
    element.setAttribute('id', '123_user');

    try {
        equal(axs.utils.hasLabel(element), false);
    } catch(e) {
        ok(false, 'Threw exception: ' + e);
    }
});

module("getQuerySelectorText", {
  setup: function () {
    this.fixture_ = document.getElementById('qunit-fixture');
  }
});
test("returns the selector text for a nested object with a class attribute", function() {
  var targetNode = document.createElement('em');
  targetNode.setAttribute('class', 'foo');
  var targetParentNode = document.createElement('p');
  targetParentNode.appendChild(targetNode);
  this.fixture_.appendChild(targetParentNode);

  equal(axs.utils.getQuerySelectorText(targetNode), "#qunit-fixture > P > .foo");
});
test("nth-of-type does not refer to a selector but a tagName", function() {
  var html = '<ul><li>One</li><li class="thing">Two</li><li class="thing">Three</li></ul>';
  this.fixture_.innerHTML = html;

  var lis = document.querySelectorAll('li');
  var lastLi = lis[lis.length - 1];
  var selector = axs.utils.getQuerySelectorText(lastLi);
  equal(lastLi, document.querySelector(selector),
        'selector "' + selector + '" does not match element');
});

module("parseColor");
test("parses alpha values correctly", function() {
  var colorString = 'rgba(255, 255, 255, .47)';
  var color = axs.utils.parseColor(colorString);
  equal(color.red, 255);
  equal(color.blue, 255);
  equal(color.green, 255);
  equal(color.alpha, .47);
});

module("getIdReferrers", {
  setup: function () {
    this.fixture_ = document.getElementById('qunit-fixture');
  }
});
test("returns the aria owners for a given element", function() {
  var owned = document.createElement("div");
  var ownerCount = 5;
  owned.id = "theOwned";
  this.fixture_.appendChild(owned);
  for(var i = 0; i < ownerCount; i++) {
    var owner = document.createElement("div");
    owner.setAttribute("aria-owns", "theOwned");
    owner.setAttribute("class", "owner");
    this.fixture_.appendChild(owner);
  }
  var expected = this.fixture_.querySelectorAll(".owner");
  var actual = axs.utils.getIdReferrers("aria-owns", owned);
  equal(expected.length, ownerCount);  // sanity check the test itself
  equal(actual.length, ownerCount);
  var allFound = Array.prototype.every.call(expected, function(element){
      return (Array.prototype.indexOf.call(actual, element) >= 0);
  });
  equal(allFound, true);
});
test("returns the elements this element labels", function() {
  var label = document.createElement("div");
  var labelledCount = 2;
  label.id = "theLabel";
  this.fixture_.appendChild(label);
  for(var i = 0; i < labelledCount; i++) {
    var labelled = document.createElement("div");
    labelled.setAttribute("aria-labelledby", "theLabel notPresentInDom");
    labelled.setAttribute("class", "labelled");
    this.fixture_.appendChild(labelled);
  }
  var expected = this.fixture_.querySelectorAll(".labelled");
  var actual = axs.utils.getIdReferrers("aria-labelledby", label);
  equal(expected.length, labelledCount);  // sanity check the test itself
  equal(actual.length, labelledCount);
  var allFound = Array.prototype.every.call(expected, function(element){
      return (Array.prototype.indexOf.call(actual, element) >= 0);
  });
  equal(allFound, true);
});
module("getAriaPropertiesByValueType", {
  setup: function () {

  }
});
test("Returns idref and idref_list types.", function() {
  var expected = ["activedescendant", "controls", "describedby", "flowto", "labelledby", "owns"];
  var actual = axs.utils.getAriaPropertiesByValueType(["idref", "idref_list"]);
  actual = Object.keys(actual);
  actual.sort();
  deepEqual(actual, expected);
});
test("Returns idref types.", function() {
  var expected = ["activedescendant"];
  var actual = axs.utils.getAriaPropertiesByValueType(["idref"]);
  actual = Object.keys(actual);
  actual.sort();
  deepEqual(actual, expected);
});

module("getSelectorForAriaProperties", {
  setup: function () {

  }
});
test("Returns a selector to match all aria properties.", function() {
  var expected = "[aria-activedescendant],[aria-atomic],[aria-autocomplete],[aria-busy],[aria-checked],[aria-controls],";
  expected += "[aria-describedby],[aria-disabled],[aria-dropeffect],[aria-expanded],[aria-flowto],[aria-grabbed],";
  expected += "[aria-haspopup],[aria-hidden],[aria-invalid],[aria-label],[aria-labelledby],[aria-level],[aria-live],";
  expected += "[aria-multiline],[aria-multiselectable],[aria-orientation],[aria-owns],[aria-posinset],[aria-pressed],";
  expected += "[aria-readonly],[aria-relevant],[aria-required],[aria-selected],[aria-setsize],[aria-sort],[aria-valuemax],";
  expected += "[aria-valuemin],[aria-valuenow],[aria-valuetext]";
  var actual = axs.utils.getSelectorForAriaProperties(axs.constants.ARIA_PROPERTIES);
  deepEqual(actual, expected);
});
test("Returns a selector to match all aria idref properties.", function() {
  var expected = "[aria-activedescendant]";
  var actual = axs.utils.getSelectorForAriaProperties(axs.utils.getAriaPropertiesByValueType(["idref"]));
  deepEqual(actual, expected);
});
