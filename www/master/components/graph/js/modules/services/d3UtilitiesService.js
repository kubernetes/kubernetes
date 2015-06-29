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

/**=========================================================
 * Module: Graph
 * Visualizer for force directed graph.
 * This is a service that provides stateless utility
 * functions for use by the d3 visualization directive.
 =========================================================*/

(function() {
  'use strict';

  var d3UtilitiesService = function() {
    // Return a random position [x,y] within radius of the origin.
    function getRandomStartingPosition(radius) {
      var t = 2 * Math.PI * Math.random();
      var u = Math.random() + Math.random();
      var r = u > 1 ? 2 - u : u;

      return [r * Math.cos(t) * radius, r * Math.sin(t) * radius];
    }

    // This function looks up whether a pair of nodes are neighbours.
    function neighboring(a, b, linkedByIndex, selectionHops) {
      // TODO(duftler): Add support for > 1 hops.
      if (selectionHops) {
        return linkedByIndex[a.index + ',' + b.index];
      } else {
        return false;
      }
    }

    // Match on Set.has() or id.
    function setHas(searchSet, item) {
      if (searchSet.has(item)) {
        return true;
      }

      var found = false;

      searchSet.forEach(function(e) {
        if (e.id !== undefined && e.id === item.id) {
          found = true;
          return;
        }
      });

      return found;
    }

    // Returns an object containing:
    //   clusters:  An array where each index is a cluster number and the value stored at that index is the node with
    //              the maximum radius in that cluster.
    //   maxRadius: The maximum radius of all the nodes.
    //
    function buildClusters(nodes) {
      var maxRadius = -1;
      var maxCluster = -1;

      nodes.forEach(function(d) {
        maxCluster = Math.max(maxCluster, d.cluster);
        maxRadius = Math.max(maxRadius, d.radius);
      });

      var clusters = new Array(maxCluster + 1);

      nodes.forEach(function(d) {
        if (!clusters[d.cluster] || (d.radius > clusters[d.cluster].radius)) {
          clusters[d.cluster] = d;
        }
      });

      return {clusters: clusters, maxRadius: maxRadius};
    }

    // Move d to be adjacent to the cluster node.
    function cluster(builtClusters, alpha) {
      return function(d) {
        var cluster = builtClusters.clusters[d.cluster];
        if (cluster === d) return;
        if (d.x == cluster.x && d.y == cluster.y) {
          d.x += 0.1;
        }
        var x = d.x - cluster.x, y = d.y - cluster.y, l = Math.sqrt(x * x + y * y), r = d.radius + cluster.radius;
        if (l != r) {
          l = (l - r) / l * alpha;
          d.x -= x *= l;
          d.y -= y *= l;
          cluster.x += x;
          cluster.y += y;
        }
      };
    }

    // Resolves collisions between d and all other nodes.
    function collide(d3, nodes, builtClusters, alpha, clusterInnerPadding, clusterOuterPadding) {
      var quadtree = d3.geom.quadtree(nodes);
      return function(d) {
        var r = d.radius + builtClusters.maxRadius + Math.max(clusterInnerPadding, clusterOuterPadding), nx1 = d.x - r,
            nx2 = d.x + r, ny1 = d.y - r, ny2 = d.y + r;
        quadtree.visit(function(quad, x1, y1, x2, y2) {
          if (quad.point && (quad.point !== d)) {
            var x = d.x - quad.point.x, y = d.y - quad.point.y, l = Math.sqrt(x * x + y * y),
                r = d.radius + quad.point.radius +
                    (d.cluster === quad.point.cluster ? clusterInnerPadding : clusterOuterPadding);
            if (l < r) {
              l = (l - r) / l * alpha;
              d.x -= x *= l;
              d.y -= y *= l;
              quad.point.x += x;
              quad.point.y += y;
            }
          }
          return x1 > nx2 || x2 < nx1 || y1 > ny2 || y2 < ny1;
        });
      };
    }

    function showContextMenu(d3, data, index, contextMenu) {
      var elm = this;

      d3.selectAll('.d3-context-menu').html('');
      var list = d3.selectAll('.d3-context-menu').append('ul');
      list.selectAll('li')
          .data(contextMenu)
          .enter()
          .append('li')
          .html(function(d) { return (typeof d.title === 'string') ? d.title : d.title(data); })
          .on('click', function(d, i) {
            d.action(elm, data, index);
            d3.select('.d3-context-menu').style('display', 'none');
          });

      // Display context menu.
      d3.select('.d3-context-menu')
          .style('left', (d3.event.pageX - 2) + 'px')
          .style('top', (d3.event.pageY - 2) + 'px')
          .style('display', 'block')
          .on('contextmenu', function() { d3.event.preventDefault(); });

      d3.event.preventDefault();
    }

    return {
      'getRandomStartingPosition': getRandomStartingPosition,
      'neighboring': neighboring,
      'setHas': setHas,
      'buildClusters': buildClusters,
      'cluster': cluster,
      'collide': collide,
      'showContextMenu': showContextMenu
    };
  };

  angular.module('kubernetesApp.components.graph.services.d3', []).service('d3UtilitiesService', [d3UtilitiesService]);

})();
