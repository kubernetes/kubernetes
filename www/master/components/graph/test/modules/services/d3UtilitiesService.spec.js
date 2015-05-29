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

describe('D3 utilities service', function() {
  var d3UtilitiesService;

  // Work around to get ngLodash correctly injected.
  beforeEach(function() { angular.module('testModule', ['ngLodash', 'kubernetesApp.components.graph']); });

  beforeEach(module('testModule'));

  beforeEach(inject(function(_d3UtilitiesService_) { d3UtilitiesService = _d3UtilitiesService_; }));

  it('should generate starting positions within specified radius of origin', function() {
    // Get random starting positions for 10,000 nodes.
    var maxDistance = 0;

    for (var i = 0; i < 50000; i++) {
      var startingPosition = d3UtilitiesService.getRandomStartingPosition(200);
      var x = startingPosition[0];
      var y = startingPosition[1];
      var distance = Math.sqrt(x * x + y * y);

      maxDistance = Math.max(distance, maxDistance);
    }

    // Test that all nodes are positioned within 200 pixels (the specified radius) of the origin.
    expect(maxDistance).toBeLessThan(200);
  });

  it('should determine whether nodes are neighbors', function() {
    var linkedByIndex = {'0,0': 1, '1,1': 1, '2,2': 1, '3,3': 1, '4,4': 1, '0,3': 1, '3,4': 1};

    // Test that 0 does not neighbor 3 when selectionHops == 0.
    var isNeighboring = d3UtilitiesService.neighboring({index: 0}, {index: 3}, linkedByIndex, 0);
    expect(isNeighboring).toBeFalsy();

    // Test that 3 does not neighbor 4 when selectionHops == 0.
    isNeighboring = d3UtilitiesService.neighboring({index: 3}, {index: 4}, linkedByIndex, 0);
    expect(isNeighboring).toBeFalsy();

    // Test that 0 does not neighbor 4 when selectionHops == 0.
    isNeighboring = d3UtilitiesService.neighboring({index: 0}, {index: 4}, linkedByIndex, 0);
    expect(isNeighboring).toBeFalsy();

    // Test that 0 does neighbor 3 when selectionHops == 1.
    isNeighboring = d3UtilitiesService.neighboring({index: 0}, {index: 3}, linkedByIndex, 1);
    expect(isNeighboring).toBeTruthy();

    // Test that 3 does neighbor 4 when selectionHops == 1.
    isNeighboring = d3UtilitiesService.neighboring({index: 3}, {index: 4}, linkedByIndex, 1);
    expect(isNeighboring).toBeTruthy();

    // Test that 0 still does not neighbor 4 when selectionHops == 1.
    isNeighboring = d3UtilitiesService.neighboring({index: 0}, {index: 4}, linkedByIndex, 1);
    expect(isNeighboring).toBeFalsy();
  });

  it('should find matches in search set', function() {
    var searchSet = new Set();
    var itemOne = {id: '1'};
    var itemTwo = {id: '2'};
    var itemThree = {id: '3'};

    searchSet.add(itemOne);
    searchSet.add({id: '2'});

    // Test that a match is returned when the actual object is in the set.
    expect(d3UtilitiesService.setHas(searchSet, itemOne)).toBeTruthy();

    // Test that a match is returned when an item with the same id is in the set, even if it is not the actual object.
    expect(d3UtilitiesService.setHas(searchSet, itemTwo)).toBeTruthy();

    // Test that a match is not returned if the object is not in the set and no item in the set has the same id.
    expect(d3UtilitiesService.setHas(searchSet, itemThree)).toBeFalsy();
  });

  it('should properly build clusters', function() {
    var nodes = [
      {cluster: 0, radius: 5},
      {cluster: 0, radius: 10},
      {cluster: 0, radius: 15},
      {cluster: 1, radius: 3},
      {cluster: 1, radius: 9},
      {cluster: 1, radius: 6},
      {cluster: 2, radius: 6},
      {cluster: 2, radius: 4},
      {cluster: 2, radius: 2},
    ];

    var builtClusters = d3UtilitiesService.buildClusters(nodes);

    // Test that 3 clusters are identified.
    expect(builtClusters.clusters.length).toEqual(3);

    // Test that the node with the largest radius in cluster 0 is the one with a radius of 15.
    expect(builtClusters.clusters[0].radius).toEqual(15);

    // Test that the node with the largest radius in cluster 1 is the one with a radius of 9.
    expect(builtClusters.clusters[1].radius).toEqual(9);

    // Test that the node with the largest radius in cluster 2 is the one with a radius of 6.
    expect(builtClusters.clusters[2].radius).toEqual(6);

    // Test that the maximum radius identified is 15.
    expect(builtClusters.maxRadius).toEqual(15);
  });
});
