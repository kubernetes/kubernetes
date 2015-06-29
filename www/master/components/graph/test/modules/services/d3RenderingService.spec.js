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

// TODO(duftler):
//  Add tests for:
//    clustered view

describe('D3 rendering service', function() {
  var d3RenderingService;
  var d3UtilitiesService;
  var parentDiv;
  var graphDiv;
  var scope;
  var d3Rendering;

  var MOCK_SAMPLE_DATA = [
    {
      'nodes': [
        {'name': 'service: guestbook', 'radius': 16, 'fill': 'olivedrab', 'id': 1, 'selected': true},
        {'name': 'pod: guestbook-controller', 'radius': 20, 'fill': 'palegoldenrod', 'id': 2, 'selected': true},
        {'name': 'pod: guestbook-controller', 'radius': 20, 'fill': 'palegoldenrod', 'id': 3, 'selected': true},
        {'name': 'pod: guestbook-controller', 'radius': 20, 'fill': 'palegoldenrod', 'id': 55},
        {'name': 'container: php-redis', 'radius': 24, 'fill': 'cornflowerblue', 'id': 77}
      ],
      'links': [
        {'source': 0, 'target': 1, 'width': 2, 'stroke': 'black', 'distance': 80},
        {'source': 0, 'target': 2, 'width': 2, 'stroke': 'black', 'distance': 80},
        {'source': 1, 'target': 3, 'width': 2, 'stroke': 'black', 'distance': 80}
      ],
      'configuration': {'settings': {'clustered': false, 'showEdgeLabels': true, 'showNodeLabels': true}}
    }
  ];

  // Work around to get ngLodash correctly injected.
  beforeEach(function() { angular.module('testModule', ['ngLodash', 'kubernetesApp.components.graph']); });

  beforeEach(module('testModule'));

  beforeEach(inject(function(_d3RenderingService_, _d3UtilitiesService_) {
    d3RenderingService = _d3RenderingService_;
    d3UtilitiesService = _d3UtilitiesService_;

    // Build the parent <div> and graph <div> to hold the <svg> element.
    parentDiv = d3.select('body').append('div');
    parentDiv.style('width', '500px');
    parentDiv.style('height', '500px');
    graphDiv = parentDiv.append('div');

    // Create the mock scope.
    scope = {
      viewModelService: {
        viewModel: {
          data: MOCK_SAMPLE_DATA[0]
        },
        setSelectionIdList: function() {}
      }
    };

    // Construct and configure the d3 rendering service.
    d3Rendering = d3RenderingService.rendering().controllerScope(scope).directiveElement(graphDiv.node());

    // Set the mock data in the scope.
    scope.viewModelService.viewModel.data = MOCK_SAMPLE_DATA[0];

    // Render the graph.
    d3Rendering();
  }));

  afterEach(function() { parentDiv.remove(); });

  it('should locate the dimensions of the parent', function() {
    // Test that container dimensions are properly calculated after rendering.
    var containerDimensionsAfterRendering = d3Rendering.getParentContainerDimensions();
    expect(containerDimensionsAfterRendering[0]).toEqual(500);
    expect(containerDimensionsAfterRendering[1]).toEqual(500);
  });

  it('should resize the graph implicitly and explicitly', function() {
    // Test that the initial graph size is calculated properly.
    var initialGraphSize = d3Rendering.graphSize();
    // The initial width is calculated by subtracting 16 from the parent container width.
    // TODO(duftler): Use same constant here as in d3RenderingService.
    expect(initialGraphSize[0]).toEqual(484);
    // The initial height defaults to 700.
    // TODO(duftler): Use same constant here as in d3RenderingService.
    expect(initialGraphSize[1]).toEqual(700);

    // Explicitly set the graph size.
    d3Rendering.graphSize([750, 750]);

    // Test that the modified graph size is calculated properly.
    var modifiedGraphSize = d3Rendering.graphSize();
    expect(modifiedGraphSize[0]).toEqual(750);
    expect(modifiedGraphSize[1]).toEqual(750);
  });

  it('should respect "selected" property in view model', function() {
    // Test that the initial selection is calculated properly.
    var initialNodeSelection = d3Rendering.nodeSelection();
    expect(initialNodeSelection.size).toEqual(3);
    expect(d3UtilitiesService.setHas(initialNodeSelection, {id: 1})).toBeTruthy();
    expect(d3UtilitiesService.setHas(initialNodeSelection, {id: 2})).toBeTruthy();
    expect(d3UtilitiesService.setHas(initialNodeSelection, {id: 3})).toBeTruthy();
  });

  it('should completely replace node selection when explicitly set', function() {
    // Create and set a new node selection.
    var newNodeSelection = new Set();
    newNodeSelection.add({id: 2});
    newNodeSelection.add({id: 55});
    d3Rendering.nodeSelection(newNodeSelection);

    // Test that the updated selection is calculated properly.
    var updatedNodeSelection = d3Rendering.nodeSelection();
    expect(updatedNodeSelection.size).toEqual(2);
    expect(d3UtilitiesService.setHas(updatedNodeSelection, {id: 2})).toBeTruthy();
    expect(d3UtilitiesService.setHas(updatedNodeSelection, {id: 55})).toBeTruthy();
  });

  it('should select appropriate edges with respect to selected nodes in view model', function() {
    // Test that the expected edges are selected and no more.
    var initialEdgeSelectionIterator = d3Rendering.edgeSelection().values();

    // Test that the first selected edge goes from node with id 1 to node with id 2.
    var current = initialEdgeSelectionIterator.next();
    expect(current.done).toBeFalsy();
    expect(current.value.source.id).toEqual(1);
    expect(current.value.target.id).toEqual(2);

    // Test that the second selected edge goes from node with id 1 to node with id 3.
    current = initialEdgeSelectionIterator.next();
    expect(current.done).toBeFalsy();
    expect(current.value.source.id).toEqual(1);
    expect(current.value.target.id).toEqual(3);

    // Test that there are only 2 edges selected.
    current = initialEdgeSelectionIterator.next();
    expect(current.done).toBeTruthy();
  });

  it('should select appropriate edges with respect to explicitly set node selection', function() {
    // Create and set a new node selection.
    var newNodeSelection = new Set();
    newNodeSelection.add({id: 2});
    newNodeSelection.add({id: 55});
    d3Rendering.nodeSelection(newNodeSelection);

    // Test that the expected edges are selected and no more.
    var updatedEdgeSelectionIterator = d3Rendering.edgeSelection().values();

    // Test that the first selected edge goes from node with id 2 to node with id 55.
    var current = updatedEdgeSelectionIterator.next();
    expect(current.done).toBeFalsy();
    expect(current.value.source.id).toEqual(2);
    expect(current.value.target.id).toEqual(55);

    // Test that there is only 1 edge selected.
    current = updatedEdgeSelectionIterator.next();
    expect(current.done).toBeTruthy();
  });

  it('should select appropriate edgelabels with respect to selected nodes in view model', function() {
    // Test that the expected edgelabels are selected and no more.
    var initialEdgelabelsSelectionIterator = d3Rendering.edgelabelsSelection().values();

    // Test that the first selected edgelabel goes from node with id 1 to node with id 2.
    var current = initialEdgelabelsSelectionIterator.next();
    expect(current.done).toBeFalsy();
    expect(current.value.source.id).toEqual(1);
    expect(current.value.target.id).toEqual(2);

    // Test that the second selected edgelabel goes from node with id 1 to node with id 3.
    current = initialEdgelabelsSelectionIterator.next();
    expect(current.done).toBeFalsy();
    expect(current.value.source.id).toEqual(1);
    expect(current.value.target.id).toEqual(3);

    // Test that there are only 2 edgelabels selected.
    current = initialEdgelabelsSelectionIterator.next();
    expect(current.done).toBeTruthy();
  });

  it('should select appropriate edgelabels with respect to explicitly set node selection', function() {
    // Create and set a new node selection.
    var newNodeSelection = new Set();
    newNodeSelection.add({id: 2});
    newNodeSelection.add({id: 55});
    d3Rendering.nodeSelection(newNodeSelection);

    // Test that the expected edgelabels are selected and no more.
    var updatedEdgelabelsSelectionIterator = d3Rendering.edgelabelsSelection().values();

    // Test that the first selected edgelabel goes from node with id 2 to node with id 55.
    var current = updatedEdgelabelsSelectionIterator.next();
    expect(current.done).toBeFalsy();
    expect(current.value.source.id).toEqual(2);
    expect(current.value.target.id).toEqual(55);

    // Test that there is only 1 edgelabel selected.
    current = updatedEdgelabelsSelectionIterator.next();
    expect(current.done).toBeTruthy();
  });

  it('should set opacity of selected nodes to 1, and opacity of all others to something else', function() {
    // Test that the initial selection is calculated properly.
    var initialNodeSelection = d3Rendering.nodeSelection();
    expect(initialNodeSelection.size).toEqual(3);

    graphDiv.selectAll('.node').each(function(e) {
      var opacity = d3.select(this).style('opacity');

      if (opacity === '1') {
        expect(d3UtilitiesService.setHas(initialNodeSelection, e)).toBeTruthy();
      } else {
        expect(d3UtilitiesService.setHas(initialNodeSelection, e)).toBeFalsy();
      }
    });

    // Create and set a new node selection.
    var newNodeSelection = new Set();
    newNodeSelection.add({id: 2});
    newNodeSelection.add({id: 55});
    d3Rendering.nodeSelection(newNodeSelection);

    // Test that the updated node selection is calculated properly.
    var updatedNodeSelection = d3Rendering.nodeSelection();
    expect(updatedNodeSelection.size).toEqual(2);

    graphDiv.selectAll('.node').each(function(e) {
      var opacity = d3.select(this).style('opacity');

      if (opacity === '1') {
        expect(d3UtilitiesService.setHas(updatedNodeSelection, e)).toBeTruthy();
      } else {
        expect(d3UtilitiesService.setHas(updatedNodeSelection, e)).toBeFalsy();
      }
    });
  });

  it('should set opacity of selected edges to 1, and opacity of all others to something else', function() {
    // Test that the initial selection is calculated properly.
    var initialEdgeSelection = d3Rendering.edgeSelection();
    expect(initialEdgeSelection.size).toEqual(2);

    graphDiv.selectAll('.link').each(function(e) {
      var opacity = d3.select(this).style('opacity');

      if (opacity === '1') {
        expect(d3UtilitiesService.setHas(initialEdgeSelection, e)).toBeTruthy();
      } else {
        expect(d3UtilitiesService.setHas(initialEdgeSelection, e)).toBeFalsy();
      }
    });

    // Create and set a new node selection.
    var newNodeSelection = new Set();
    newNodeSelection.add({id: 2});
    newNodeSelection.add({id: 55});
    d3Rendering.nodeSelection(newNodeSelection);

    // Test that the updated edge selection is calculated properly.
    var updatedEdgeSelection = d3Rendering.edgeSelection();
    expect(updatedEdgeSelection.size).toEqual(1);

    graphDiv.selectAll('.link').each(function(e) {
      var opacity = d3.select(this).style('opacity');

      if (opacity === '1') {
        expect(d3UtilitiesService.setHas(updatedEdgeSelection, e)).toBeTruthy();
      } else {
        expect(d3UtilitiesService.setHas(updatedEdgeSelection, e)).toBeFalsy();
      }
    });
  });

  it('should set opacity of selected edgelabels to 1, and opacity of all others to something else', function() {
    // Test that the initial selection is calculated properly.
    var initialEdgelabelsSelection = d3Rendering.edgelabelsSelection();
    expect(initialEdgelabelsSelection.size).toEqual(2);

    graphDiv.selectAll('.edgelabel')
        .each(function(e) {
          var opacity = d3.select(this).style('opacity');

          if (opacity === '1') {
            expect(d3UtilitiesService.setHas(initialEdgelabelsSelection, e)).toBeTruthy();
          } else {
            expect(d3UtilitiesService.setHas(initialEdgelabelsSelection, e)).toBeFalsy();
          }
        });

    // Create and set a new node selection.
    var newNodeSelection = new Set();
    newNodeSelection.add({id: 2});
    newNodeSelection.add({id: 55});
    d3Rendering.nodeSelection(newNodeSelection);

    // Test that the updated edgelabels selection is calculated properly.
    var updatedEdgelabelsSelection = d3Rendering.edgeSelection();
    expect(updatedEdgelabelsSelection.size).toEqual(1);

    graphDiv.selectAll('.edgelabel')
        .each(function(e) {
          var opacity = d3.select(this).style('opacity');

          if (opacity === '1') {
            expect(d3UtilitiesService.setHas(updatedEdgelabelsSelection, e)).toBeTruthy();
          } else {
            expect(d3UtilitiesService.setHas(updatedEdgelabelsSelection, e)).toBeFalsy();
          }
        });
  });

  it('should set opacity of all nodes, edges, edgelabels and images to 1 when nothing is selected', function() {
    // Set the node selection to an empty set.
    d3Rendering.nodeSelection(new Set());

    graphDiv.selectAll('.node').each(function(e) { expect(d3.select(this).style('opacity')).toEqual('1'); });

    graphDiv.selectAll('.link').each(function(e) { expect(d3.select(this).style('opacity')).toEqual('1'); });

    graphDiv.selectAll('.edgelabel').each(function(e) { expect(d3.select(this).style('opacity')).toEqual('1'); });

    graphDiv.selectAll('image').each(function(e) { expect(d3.select(this).style('opacity')).toEqual('1'); });
  });

  it('should update node settings cache when node is pinned and unpinned', function() {
    // Pin a node.
    d3Rendering.togglePinned({id: 2, fixed: 0});

    // Test that it's cached as fixed.
    var updatedNodeSettingsCache = d3Rendering.nodeSettingsCache();
    expect(updatedNodeSettingsCache[2].fixed).toBeTruthy();

    // Unpin the same node.
    d3Rendering.togglePinned({id: 2, fixed: 8});

    // Test that it's not cached as fixed.
    updatedNodeSettingsCache = d3Rendering.nodeSettingsCache();
    expect(updatedNodeSettingsCache[2].fixed).toBeFalsy();
  });

  it('should update node settings cache when pins are reset', function() {
    // Pin two nodes.
    d3Rendering.togglePinned({id: 2, fixed: 0});
    d3Rendering.togglePinned({id: 3, fixed: 0});

    // Test that they are cached as fixed.
    var updatedNodeSettingsCache = d3Rendering.nodeSettingsCache();
    expect(updatedNodeSettingsCache[2].fixed).toBeTruthy();
    expect(updatedNodeSettingsCache[3].fixed).toBeTruthy();

    // Reset all pins.
    d3Rendering.resetPins();

    // Test that no nodes are cached as fixed.
    updatedNodeSettingsCache = d3Rendering.nodeSettingsCache();

    for (var nodeId in updatedNodeSettingsCache) {
      expect(updatedNodeSettingsCache[nodeId].fixed).toBeFalsy();
    }
  });

  // This is the equivalent of the old waitsFor/runs syntax
  // which was removed from Jasmine 2. From https://gist.github.com/abreckner/110e28897d42126a3bb9.
  var waitsForAndRuns = function(escapeFunction, runFunction, escapeTime) {
    // check the escapeFunction every millisecond so as soon as it is met we can escape the function
    var interval = setInterval(function() {
      if (escapeFunction()) {
        clearWaitsForAndRuns();
        runFunction();
      }
    }, 1);
   
    // in case we never reach the escapeFunction, we will time out
    // at the escapeTime
    var timeOut = setTimeout(function() {
      clearWaitsForAndRuns();
      runFunction();
    }, escapeTime);
   
    // clear the interval and the timeout
    function clearWaitsForAndRuns(){
      clearInterval(interval);
      clearTimeout(timeOut);
    }
  };

  it('should update view settings cache when image is zoomed', function() {
    // Adjust the zoom to 75%.
    d3Rendering.adjustZoom(0.75);
    waitsForAndRuns(
      function() { return d3Rendering.viewSettingsCache().scale < 0.77; }, 
      function() { expect(d3Rendering.viewSettingsCache().scale).toBeLessThan(0.76); }, 
      1000);
  });
});
