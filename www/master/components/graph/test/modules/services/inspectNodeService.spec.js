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

describe("Inspect node service", function() {
  var inspectNodeService;

  beforeEach(module('kubernetesApp.components.graph.services'));
  beforeEach(inject(function(_inspectNodeService_) { inspectNodeService = _inspectNodeService_; }));

  it("should set and get data as intended", function() {
    var data = {
      'name': 'pod',
      'id': 1
    };
    inspectNodeService.setDetailData(data);
    var getData = inspectNodeService.getDetailData();
    expect(data).toEqual(getData);
  });
});
