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

describe("Inspect node controller", function() {
  var inspectNodeService = {};
  var scope, location, controller;
  var mockNodeDetail = {
    'id': 1,
    'metadata': 'data'
  };
  // Work around to get ngLodash correctly injected.
  beforeEach(function() {
    angular.module('testModule', ['ngLodash', 'kubernetesApp.components.graph', 'kubernetesApp.config']);
  });

  beforeEach(module('testModule'));

  beforeEach(inject(function(_inspectNodeService_, $controller, $location, $rootScope) {
    inspectNodeService = _inspectNodeService_;
    // Mock the node detail data returned by the service.
    inspectNodeService.setDetailData(mockNodeDetail);
    scope = $rootScope.$new();
    location = $location;
    controller = $controller('InspectNodeCtrl', {$scope: scope, $location: location});
  }));

  it("should work as intended", function() {
    // Test if the controller sets the correct model values.
    expect(scope.element).toEqual(mockNodeDetail.id);
    expect(scope.metadata).toEqual(mockNodeDetail.metadata);

    // Test if the controller changed the location correctly.
    scope.backToGraph();
    expect(location.path()).toEqual('/graph');
  });
});
