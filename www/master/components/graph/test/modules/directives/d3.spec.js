/**
 Copyright 2015 Google Inc. All rights reserved.

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

'use strict';

describe('D3 directive', function() {
  var $compile;
  var $rootScope;
  var viewModelService;

  var MOCK_SAMPLE_DATA = [{
    'nodes': [
      {'name': 'service: guestbook', 'radius': 16, 'fill': 'olivedrab', 'id': 5},
      {'name': 'pod: guestbook-controller', 'radius': 20, 'fill': 'palegoldenrod', 'id': 2},
    ],
    'links': [],
    'configuration': {'settings': {'clustered': false, 'showEdgeLabels': true, 'showNodeLabels': true}}
  }];

  // Work around to get ngLodash correctly injected.
  beforeEach(function() { angular.module('testModule', ['ngLodash', 'kubernetesApp.components.graph']); });

  beforeEach(module('testModule'));

  beforeEach(inject(function(_$compile_, _$rootScope_, _viewModelService_) {
    $compile = _$compile_;
    $rootScope = _$rootScope_;
    viewModelService = _viewModelService_;
  }));

  it('should replace the element with the appropriate svg content in response to the viewModel being set', function() {
    // Compile some HTML containing the directive.
    var element = $compile('<div><d3-visualization></d3-visualization></div>')($rootScope);

    $rootScope.viewModelService = viewModelService;

    // Test that the element hasn't been compiled yet.
    expect(element.html()).toEqual('<d3-visualization></d3-visualization>');

    // Request the viewModelService to update the view model with the specified data.
    viewModelService.setViewModel(MOCK_SAMPLE_DATA[0]);

    // Test that the element still hasn't been compiled yet.
    expect(element.html()).toEqual('<d3-visualization></d3-visualization>');

    // Fire all the watches.
    $rootScope.$digest();

    // Test that the element has been compiled and contains the svg content.
    expect(element.html()).toContain('<svg');
    expect(element.html()).toContain('service: guestbook');
    expect(element.html()).toContain('pod: guestbook-controller');
  });

  it('should set the node selection in response to the selectionIdList being set', function() {
    // Compile some HTML containing the directive.
    var element = $compile('<div><d3-visualization></d3-visualization></div>')($rootScope);

    $rootScope.viewModelService = viewModelService;

    // Request the viewModelService to update the view model with the specified data. No initial selections.
    viewModelService.setViewModel(MOCK_SAMPLE_DATA[0]);

    // Fire all the watches.
    $rootScope.$digest();

    // Test that each node has an opacity of 1.
    var nodeList = element[0].querySelectorAll("g > g");

    for (var i = 0; i < nodeList.length; i++) {
      var node = angular.element(nodeList[i]);
      if (node.style !== undefined) {
        expect(node.style).toEqual('opacity: 1;'); 
      }
    }

    // Set a new selection id list that should trigger a watch.
    $rootScope.selectionIdList = [2];

    // Fire all the watches.
    $rootScope.$digest();

    // Test that at least one node has an opacity of 1 and at least one node has an opacity of less than 1.
    var foundOpacityOfOne = false;
    var foundOpacityLessThanOne = false;
    nodeList = element[0].querySelectorAll("g > g");

    for (var i = 0; i < nodeList.length; i++) {
      if (nodeList[i].getAttribute("style") === "opacity: 1;") {
        foundOpacityOfOne = true;
      } else if (nodeList[i].getAttribute("style") === "opacity: 0.2;") {
        foundOpacityLessThanOne = true;
      }
    }

    expect(foundOpacityOfOne).toBeTruthy();
    expect(foundOpacityLessThanOne).toBeTruthy();
  });
});
