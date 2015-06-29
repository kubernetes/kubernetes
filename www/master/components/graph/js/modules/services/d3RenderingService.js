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
 * This is a service that provides rendering capabilities
 * for use by the d3 visualization directive.
 =========================================================*/
(function() {
  'use strict';

  var d3RenderingService = function(lodash, d3UtilitiesService, $location, $rootScope, inspectNodeService) {

    function rendering() {
      var CONSTANTS = {
        FIXED_DRAGGING_BIT: 2,
        FIXED_MOUSEOVER_BIT: 4,
        FIXED_PINNED_BIT: 8,
        SHOWPIN_MOUSEOVER_BIT: 2,
        SHOWPIN_METAKEYDOWN_BIT: 4,
        OPACITY_MOUSEOVER: 0.7,
        OPACITY_DESELECTED: 0.2,
        // TODO(duftler): Externalize these defaults.
        DEFAULTS: {
          RENDER_NODE_ICONS: true,
          BOUNDING_BOX: [30, 30],
          USE_RADIUS_FOR_BOUNDING_BOX: true,
          SVG_INITIAL_HEIGHT: 700,
          FORCE_CLUSTERED_GRAVITY: 0.80,
          FORCE_CLUSTERED_CHARGE: 0,
          FORCE_CLUSTERED_THETA: 0.1,
          FORCE_CLUSTERED_REFRESH_STARTING_ALPHA: 0.03,
          FORCE_CLUSTERED_REDRAW_STARTING_ALPHA: 0.9,
          FORCE_NONCLUSTERED_GRAVITY: 0.9,
          FORCE_NONCLUSTERED_CHARGE: -1500,
          // FORCE_NONCLUSTERED_CHARGE_DISTANCE: 350,
          FORCE_NONCLUSTERED_THETA: 0.01,
          FORCE_NONCLUSTERED_FRICTION: 0.7,
          FORCE_NONCLUSTERED_REFRESH_STARTING_ALPHA: 0.05,
          FORCE_NONCLUSTERED_REDRAW_STARTING_ALPHA: 0.3,
          FORCE_REFRESH_THRESHOLD_PERCENTAGE: 1,
          CLUSTER_INNER_PADDING: 4,
          CLUSTER_OUTER_PADDING: 32
        }
      };

      var directiveElement;
      var controllerScope;

      var nodeIconCache = {};

      // Used to maintain settings that must survive refresh.
      var viewSettingsCache = {};
      var nodeSettingsCache = {};

      // Contains the currently-seleted resources.
      var selection = {
        nodes: new Set(),
        edges: new Set(),
        edgelabels: new Set()
      };

      var node;
      var circle;
      var pin;
      var transform;
      var link;
      var edgepaths;
      var edgelabels;
      var force;
      var zoom;
      var g;
      var center;

      // Used to store the largest node for each cluster.
      var builtClusters;
      // The configured padding between nodes within a cluster.
      var clusterInnerPadding;
      // The configured padding between clusters.
      var clusterOuterPadding;

      // Select all edges and edgelabels where both the source and target nodes are selected.
      function selectEdgesInScope() {
        selection.edges.clear();
        selection.edgelabels.clear();

        // Add each edge where both the source and target nodes are selected.
        if (link) {
          link.each(function(e) {
            if (d3UtilitiesService.setHas(selection.nodes, e.source) &&
              d3UtilitiesService.setHas(selection.nodes, e.target)) {
              selection.edges.add(e);
            }
          });
        }

        // Add each edge label where both the source and target nodes are selected.
        if (edgelabels) {
          edgelabels.each(function(e) {
            if (d3UtilitiesService.setHas(selection.nodes, e.source) &&
              d3UtilitiesService.setHas(selection.nodes, e.target)) {
              selection.edgelabels.add(e);
            }
          });
        }
      }

      // Adjust the opacity of all resources to indicate selected items.
      function applySelectionToOpacity() {
        var notSelectedOpacity = CONSTANTS.OPACITY_DESELECTED;

        // If nothing is selected, show everything.
        if (!selection.nodes.size && !selection.edges.size && !selection.edgelabels.size) {
          notSelectedOpacity = 1;
        }

        // Reduce the opacity of all but the selected nodes.
        node.style('opacity', function(e) {
          var newOpacity = d3UtilitiesService.setHas(selection.nodes, e) ? 1 : notSelectedOpacity;

          if (e.originalOpacity) {
            e.originalOpacity = newOpacity;
          }

          return newOpacity;
        });

        // Reduce the opacity of all but the selected edges.
        if (link) {
          link.style('opacity', function(e) {
            return d3UtilitiesService.setHas(selection.edges, e) ? 1 : notSelectedOpacity;
          });
        }

        // Reduce the opacity of all but the selected edge labels.
        if (edgelabels) {
          edgelabels.style('opacity', function(e) {
            return d3UtilitiesService.setHas(selection.edgelabels, e) ? 1 : notSelectedOpacity;
          });
        }

        var selectionIdList = [];

        selection.nodes.forEach(function(e) {
          if (e.id !== undefined) {
            selectionIdList.push(e.id);
          }
        });

        controllerScope.viewModelService.setSelectionIdList(selectionIdList);

        _.defer(function() {
          $rootScope.$apply();
          autosizeSVG(d3, false);
        });
      }

      // Return the dimensions of the parent element of the d3 visualization directive.
      function getParentContainerDimensions(d3) {
        var parentNode = d3.select(directiveElement.parentNode);
        var width = parseInt(parentNode.style('width'));
        var height = parseInt(parentNode.style('height'));

        return [width, height];
      }

      // Resize the svg element.
      function resizeSVG(d3, newSVGDimensions) {
        var svg = d3.select(directiveElement).select('svg');
        var width = newSVGDimensions[0];
        var height = newSVGDimensions[1];

        svg.attr('width', width);
        svg.attr('height', height);

        // We want the width and height to survive redraws.
        viewSettingsCache.width = width;
        viewSettingsCache.height = height;

        force.size([width, height]);
      }

      // Adjust the size of the svg element to a new size derived from the dimensions of the parent.
      function autosizeSVG(d3, windowWasResized) {
        var containerDimensions = getParentContainerDimensions(d3);
        var width = containerDimensions[0] - 16;
        var height = containerDimensions[1] - 19;

        resizeSVG(d3, [width, height]);
        if (windowWasResized) {
          force.resume();
        }
      }

      // Get or set the directive element. Returns the rendering service when acting as a setter.
      graph.directiveElement = function(newDirectiveElement) {
        if (!arguments.length) return directiveElement;
        directiveElement = newDirectiveElement;
        return this;
      };

      // Get or set the controller scope. Returns the rendering service when acting as a setter.
      graph.controllerScope = function(newControllerScope) {
        if (!arguments.length) return controllerScope;
        controllerScope = newControllerScope;
        return this;
      };

      // Return the dimensions of the parent container.
      graph.getParentContainerDimensions = function() {
        return getParentContainerDimensions(window.d3);
      };

      // Get or set the size of the svg element. Returns the rendering service when acting as a setter.
      graph.graphSize = function(newGraphSize) {
        if (!arguments.length) {
          var svg = window.d3.select(directiveElement).select('svg');
          return [parseInt(svg.attr('width')), parseInt(svg.attr('height'))];
        } else {
          resizeSVG(window.d3, newGraphSize);
          return this;
        }
      };

      // Get or set the node selection. Returns the rendering service when acting as a setter.
      graph.nodeSelection = function(newNodeSelection) {
        if (!arguments.length) return selection.nodes;
        selection.nodes = newNodeSelection;
        selectEdgesInScope();
        applySelectionToOpacity();
        return this;
      };

      // Get or set the edge selection. Returns the rendering service when acting as a setter.
      graph.edgeSelection = function(newEdgeSelection) {
        if (!arguments.length) return selection.edges;
        selection.edges = newEdgeSelection;
        return this;
      };

      // Get or set the edgelabels selection. Returns the rendering service when acting as a setter.
      graph.edgelabelsSelection = function(newEdgelabelsSelection) {
        if (!arguments.length) return selection.edgelabels;
        selection.edgelabels = newEdgelabelsSelection;
        return this;
      };

      // Toggle the pinned state of this node.
      function togglePinned(d) {
        if (!nodeSettingsCache[d.id]) {
          nodeSettingsCache[d.id] = {};
        }

        if (d.fixed & CONSTANTS.FIXED_PINNED_BIT) {
          d.fixed &= ~CONSTANTS.FIXED_PINNED_BIT;
          force.start().alpha(CONSTANTS.DEFAULTS.FORCE_CLUSTERED_REFRESH_STARTING_ALPHA * 2);
          nodeSettingsCache[d.id].fixed = false;
        } else {
          d.fixed |= CONSTANTS.FIXED_PINNED_BIT;
          nodeSettingsCache[d.id].fixed = true;
          tick();
        }
      }

      graph.togglePinned = togglePinned;

      // Clear all pinned nodes.
      function resetPins() {
        node.each(function(d) {
          // Unset the appropriate bit on each node.
          d.fixed &= ~CONSTANTS.FIXED_PINNED_BIT;

          // Ensure the node is not marked in the cache as fixed.
          if (nodeSettingsCache[d.id]) {
            nodeSettingsCache[d.id].fixed = false;
          }
        });

        force.start().alpha(0.01);
      }

      graph.resetPins = resetPins;

      function getBoundingBox(d) {
        if (d) {
          if (_.isArray(d.size)) {
            return d.size;
          }

          if (CONSTANTS.DEFAULTS.USE_RADIUS_FOR_BOUNDING_BOX && d.radius) {
            return [d.radius * 2, d.radius * 2];
          }
        }

        return CONSTANTS.DEFAULTS.BOUNDING_BOX;
      }

      // Render the graph.
      function graph() {
        // Adjust selection in response to a single-click on a node.
        function toggleSelected(d) {
          var isSelected = d3UtilitiesService.setHas(selection.nodes, d);
          // Select if no nodes are currently selected or this node is not selected or the shift key is pressed.
          var selectOperation = !isSelected || d3.event.shiftKey;
          if (selectOperation) {
            // Add the clicked node.
            if (!isSelected) {
              selection.nodes.add(d);
              d.selectionHops = 0;
            } else {
              d.selectionHops = (d.selectionHops || 0) + 1;
            }

            // Add each node within the appropriate number of hops from the clicked node.
            if (d.selectionHops > 0) {
              node.each(function(e) {
                if (d3UtilitiesService.neighboring(d, e, linkedByIndex, d.selectionHops) |
                  d3UtilitiesService.neighboring(e, d, linkedByIndex, d.selectionHops)) {
                  selection.nodes.add(e);
                }
              });
            }
          } else {
            // De-select the clicked node.
            selection.nodes.delete(d);
            var numberOfHops = d.selectionHops || 0;
            delete d.selectionHops;

            // Remove each node within the appropriate number of hops from the clicked node.
            if (numberOfHops > 0) {
              node.each(function(e) {
                if (d3UtilitiesService.neighboring(d, e, linkedByIndex, numberOfHops) |
                  d3UtilitiesService.neighboring(e, d, linkedByIndex, numberOfHops)) {
                  selection.nodes.delete(e);
                }
              });
            }
          }

          selectEdgesInScope();
          applySelectionToOpacity();
        }

        // Clear all selected resources.
        function resetSelection() {
          // Show everything.
          _.forEach(selection.nodes, function(d) {
            delete d.selectionHops;
          });
          selection.nodes.clear();
          selection.edges.clear();
          selection.edgelabels.clear();
          applySelectionToOpacity();
        }

        // Return the configured padding between nodes within a cluster.
        function getClusterInnerPadding() {
          var result = CONSTANTS.DEFAULTS.CLUSTER_INNER_PADDING;
          if (graph.configuration.settings.clusterSettings &&
            graph.configuration.settings.clusterSettings.innerPadding !== undefined) {
            result = graph.configuration.settings.clusterSettings.innerPadding;
          }

          return result;
        }

        // Return the configured padding between clusters.
        function getClusterOuterPadding() {
          var result = CONSTANTS.DEFAULTS.CLUSTER_OUTER_PADDING;
          if (graph.configuration.settings.clusterSettings &&
            graph.configuration.settings.clusterSettings.outerPadding !== undefined) {
            result = graph.configuration.settings.clusterSettings.outerPadding;
          }

          return result;
        }

        // The context menu to display when not right-clicking on a node.
        var canvasContextMenu = [{
          title: 'Reset Zoom/Pan',
          action: function(elm, d, i) {
            adjustZoom();
          }
        }, {
          title: 'Reset Selection',
          action: function(elm, d, i) {
            resetSelection();
          }
        }, {
          title: 'Reset Pins',
          action: function(elm, d, i) {
            resetPins();
          }
        }];

        // The context menu to display when right-clicking on a node.
        var nodeContextMenu = [{
          title: function(d) {
            return 'Inspect Node';
          },
          action: function(elm, d, i) {
            inspectNode(d);
          }
        }];

        // Display 'Inspect' view for this node.
        function inspectNode(d, tagName) {
          if (tagName) {
            // Clone the node.
            d = JSON.parse(JSON.stringify(d));
            if (d.metadata && d.metadata[tagName]) {
              // Prefix the tag name with asterisks so it stands out in the details view.
              d.metadata['** ' + tagName] = d.metadata[tagName];

              // Remove the non-decorated tag.
              delete d.metadata[tagName];
            }
          }

          // Add the node details into the service, to be consumed by the
          // next controller.
          inspectNodeService.setDetailData(d);

          // Redirect to the detail view page.
          $location.path('/graph/inspect');
          $rootScope.$apply();
        }

        function wheelScrollHandler() {
          var origTranslate = zoom.translate();
          zoom.translate([origTranslate[0] - window.event.deltaX, origTranslate[1] - window.event.deltaY]);
          zoomed();
        }

        function dragstarted(d) {
          d3.event.sourceEvent.stopPropagation();
          d.fixed |= CONSTANTS.FIXED_DRAGGING_BIT;
          d.dragging = true;
        }

        function dragmove(d) {
          d.dragMoved = true;
          d.px = d3.event.x, d.py = d3.event.y;
          force.start().alpha(CONSTANTS.DEFAULTS.FORCE_CLUSTERED_REFRESH_STARTING_ALPHA * 2);
        }

        function dragended(d) {
          d.fixed &= ~(CONSTANTS.FIXED_DRAGGING_BIT + CONSTANTS.FIXED_MOUSEOVER_BIT);
          d.dragging = false;
          d.dragMoved = false;
        }

        function getNodeFill(d) {
          return d.fill || 'white';
        }

        function getNodeStroke(d) {
          return d.stroke || 'black';
        }

        function getNodeStrokeWidth(d) {
          return d.strokeWidth || '1';
        }

        function d3_layout_forceMouseover(d) {
          // If we use Cmd-Tab but don't navigate away from this window, the keyup event won't have a chance to fire.
          // Unsetting this bit here ensures that the Pin cursor won't be displayed when mousing over a node, unless
          // the Cmd key is down.
          if (!d3.event.metaKey) {
            showPin &= ~CONSTANTS.SHOWPIN_METAKEYDOWN_BIT;
          }

          showPin |= CONSTANTS.SHOWPIN_MOUSEOVER_BIT;

          // We show the Pin cursor if the cursor is over the node and the command key is depressed.
          if (showPin === (CONSTANTS.SHOWPIN_MOUSEOVER_BIT + CONSTANTS.SHOWPIN_METAKEYDOWN_BIT)) {
            svg.attr('class', 'graph pin-cursor');
          }

          d.fixed |= CONSTANTS.FIXED_MOUSEOVER_BIT;
          d.px = d.x;
          d.py = d.y;

          var gSelection = d3.select(this);

          // We capture the original opacity so we have a value to return to after removing the cursor from this node.
          d.originalOpacity = gSelection.style('opacity') || 1;
          d.opacity = CONSTANTS.OPACITY_MOUSEOVER;
          if (CONSTANTS.DEFAULTS.RENDER_NODE_ICONS && d.icon) {
            // Circles also get an outline.
            var circleSelection = gSelection.select('circle');
            if (circleSelection) {
              d.originalStroke = getNodeStroke(d);
              d.originalStrokeWidth = getNodeStrokeWidth(d);
              circleSelection
                .style('stroke', 'black')
                .style('stroke-width', '2');
            }
          }

          gSelection.style('opacity', d.opacity);
          tick();
        }

        function d3_layout_forceMouseout(d) {
          showPin &= ~CONSTANTS.SHOWPIN_MOUSEOVER_BIT;
          svg.attr('class', 'graph');

          d.fixed &= ~CONSTANTS.FIXED_MOUSEOVER_BIT;

          var gSelection = d3.select(this);
          if (d.originalOpacity) {
            d.opacity = d.originalOpacity;
            delete d.originalOpacity;
            gSelection
              .style('opacity', d.opacity);
          }

          var circleSelection = gSelection.select('circle');
          if (circleSelection) {
            if (d.originalStroke) {
              d.stroke = d.originalStroke;
              delete d.originalStroke;
              circleSelection
                .style('stroke', getNodeStroke(d));
            }

            if (d.originalStrokeWidth) {
              d.strokeWidth = d.originalStrokeWidth;
              delete d.originalStrokeWidth;
              circleSelection
                .style('stroke-width', getNodeStrokeWidth(d));
            }
          }

          tick();
        }

        // Resize the svg element in response to the window resizing.
        function windowWasResized() {
          autosizeSVG(d3, true);
        }

        // Apply all cached settings to nodes, giving precedence to properties explicitly specified in the view model.
        // Return true if the given node has neither a specified nor cached position. Return false otherwise.
        function applyCachedSettingsToNodes(n, selectedNodeSet) {
          var noSpecifiedOrCachedPosition = false;
          var cachedSettings;

          if (n.id) {
            cachedSettings = nodeSettingsCache[n.id];
          }

          if (n.fixed) {
            // If view model specifies node is fixed, it's fixed.
            n.fixed = CONSTANTS.FIXED_PINNED_BIT;
          } else if (cachedSettings && cachedSettings.fixed) {
            // Otherwise, take into account the fixed property from the cache.
            n.fixed = CONSTANTS.FIXED_PINNED_BIT;
          }

          if (n.position) {
            // If view model specifies position use that as the starting position.
            n.x = n.position[0];
            n.y = n.position[1];

            noSpecifiedOrCachedPosition = true;
          } else if (cachedSettings) {
            // Otherwise, take into account the position from the cache.
            var cachedPosition = cachedSettings.position;

            if (cachedPosition) {
              n.x = cachedPosition[0];
              n.y = cachedPosition[1];
            }
          }

          // If we have neither a view model specified position, nor a cached position, use a random starting position
          // within some radius of the canvas center.
          if (!n.x && !n.y) {
            var radius = graph.nodes.length * 3;
            var startingPosition = d3UtilitiesService.getRandomStartingPosition(radius);

            n.x = center[0] + startingPosition[0];
            n.y = center[1] + startingPosition[1];

            noSpecifiedOrCachedPosition = true;
          }

          // Build up a set of nodes the view model specifies are to be selected.
          if (n.selected && n.id !== 'undefined') {
            selectedNodeSet.add({
              id: n.id
            });
          }

          return noSpecifiedOrCachedPosition;
        }

        // We want to stop any prior simulation before starting a new one.
        if (force) {
          force.stop();
        }

        var d3 = window.d3;
        d3.select(window).on('resize', windowWasResized);

        // TODO(duftler): Derive the initial svg height from the container rather than the other way around.
        var width = viewSettingsCache.width || (getParentContainerDimensions(d3)[0] - 16);
        var height = viewSettingsCache.height || CONSTANTS.DEFAULTS.SVG_INITIAL_HEIGHT;

        center = [width / 2, height / 2];

        var color = d3.scale.category20();

        d3.select(directiveElement).select('svg').remove();

        var svg = d3.select(directiveElement)
          .append('svg')
          .attr('width', width)
          .attr('height', height)
          .attr('class', 'graph');

        svg.append('defs')
          .selectAll('marker')
          .data(['suit', 'licensing', 'resolved'])
          .enter()
          .append('marker')
          .attr('id', function(d) {
            return d;
          })
          .attr('viewBox', '0 -5 10 10')
          .attr('refX', 60)
          .attr('refY', 0)
          .attr('markerWidth', 6)
          .attr('markerHeight', 6)
          .attr('orient', 'auto')
          .attr('markerUnits', 'userSpaceOnUse')
          .append('path')
          .attr('d', 'M0,-5L10,0L0,5 L10,0 L0, -5')
          .style('stroke', 'black')
          .style('opacity', '1');

        svg.on('contextmenu', function(data, index) {
          if (d3.select('.d3-context-menu').style('display') !== 'block') {
            d3UtilitiesService.showContextMenu(d3, data, index, canvasContextMenu);
          }

          // Even if we don't show a new context menu, we don't want the browser's default context menu shown.
          d3.event.preventDefault();
        });

        zoom = d3.behavior.zoom().scaleExtent([0.5, 12]).on('zoom', zoomed);

        if (viewSettingsCache.translate && viewSettingsCache.scale) {
          zoom.translate(viewSettingsCache.translate).scale(viewSettingsCache.scale);
        }

        g = svg.append('g');

        svg.call(zoom).on('dblclick.zoom', null).call(zoom.event);

        var origWheelZoomHandler = svg.on('wheel.zoom');
        svg.on('wheel.zoom', wheelScrollHandler);

        var showPin = 0;

        d3.select('body')
          .on('keydown',
            function() {
              if (d3.event.ctrlKey) {
                svg.on('wheel.zoom', origWheelZoomHandler);
                svg.attr('class', 'graph zoom-cursor');
              } else if (d3.event.metaKey) {
                showPin |= CONSTANTS.SHOWPIN_METAKEYDOWN_BIT;

                if (showPin === (CONSTANTS.SHOWPIN_MOUSEOVER_BIT + CONSTANTS.SHOWPIN_METAKEYDOWN_BIT)) {
                  svg.attr('class', 'graph pin-cursor');
                }
              }
            })
          .on('keyup', function() {
            if (!d3.event.ctrlKey) {
              svg.on('wheel.zoom', wheelScrollHandler);
              svg.attr('class', 'graph');
            }

            if (!d3.event.metaKey) {
              showPin &= ~CONSTANTS.SHOWPIN_METAKEYDOWN_BIT;
              svg.attr('class', 'graph');
            }
          });

        function windowBlur() {
          // If we Cmd-Tab away from this window, the keyup event won't have a chance to fire.
          // Unsetting this bit here ensures that the Pin cursor won't be displayed when focus returns to this window.
          showPin &= ~CONSTANTS.SHOWPIN_METAKEYDOWN_BIT;
          svg.attr('class', 'graph');
        }

        window.addEventListener('blur', windowBlur);

        var drag = d3.behavior.drag()
          .origin(function(d) {
            return d;
          })
          .on('dragstart', dragstarted)
          .on('drag', dragmove)
          .on('dragend', dragended);

        var graph = undefined;
        if (controllerScope.viewModelService) {
          graph = controllerScope.viewModelService.viewModel.data;
        }

        if (graph === undefined) {
          return;
        }

        force = d3.layout.force().size([width, height]).on('tick', tick);

        if (graph.configuration.settings.clustered) {
          force.gravity(CONSTANTS.DEFAULTS.FORCE_CLUSTERED_GRAVITY)
            .charge(CONSTANTS.DEFAULTS.FORCE_CLUSTERED_CHARGE)
            .theta(CONSTANTS.DEFAULTS.FORCE_CLUSTERED_THETA);

          clusterInnerPadding = getClusterInnerPadding();
          clusterOuterPadding = getClusterOuterPadding();
        } else {
          force.gravity(CONSTANTS.DEFAULTS.FORCE_NONCLUSTERED_GRAVITY)
            .charge(CONSTANTS.DEFAULTS.FORCE_NONCLUSTERED_CHARGE)
            // .chargeDistance(CONSTANTS.DEFAULTS.FORCE_NONCLUSTERED_CHARGE_DISTANCE)
            .theta(CONSTANTS.DEFAULTS.FORCE_NONCLUSTERED_THETA)
            .friction(CONSTANTS.DEFAULTS.FORCE_NONCLUSTERED_FRICTION)
            .linkDistance(function(d) {
              return d.distance;
            })
            .links(graph.links);

          // Create all the line svgs but without locations yet.
          link = g.selectAll('.link')
            .data(graph.links)
            .enter()
            .append('line')
            .attr('class', 'link')
            .style('marker-end',
              function(d) {
                if (d.directed) {
                  return 'url(#suit)';
                }
                return 'none';
              })
            .style('stroke', function(d) {
              return getNodeStroke(d);
            })
            .style('stroke-dasharray', function(d) {
              return d.dash || ('1, 0');
            })
            .style('stroke-linecap', function(d) {
              return d.linecap || 'round';
            })
            .style('stroke-width', function(d) {
              return d.width;
            });
        }

        var selectedNodeSet = new Set();
        var newPositionCount = 0;

        // Apply all cached settings and count number of nodes with new positions.
        graph.nodes.forEach(function(n) {
          if (applyCachedSettingsToNodes(n, selectedNodeSet)) {
            ++newPositionCount;
          }
        });

        // If any nodes in the graph are explicitly selected, the cached selection is overridden.
        if (selectedNodeSet.size) {
          selection.nodes = selectedNodeSet;
        }

        force.nodes(graph.nodes);

        var startingAlpha = graph.configuration.settings.clustered ?
          CONSTANTS.DEFAULTS.FORCE_CLUSTERED_REDRAW_STARTING_ALPHA :
          CONSTANTS.DEFAULTS.FORCE_NONCLUSTERED_REDRAW_STARTING_ALPHA;
        if (newPositionCount <= (CONSTANTS.DEFAULTS.FORCE_REFRESH_THRESHOLD_PERCENTAGE * graph.nodes.length)) {
          startingAlpha = graph.configuration.settings.clustered ?
            CONSTANTS.DEFAULTS.FORCE_CLUSTERED_REFRESH_STARTING_ALPHA :
            CONSTANTS.DEFAULTS.FORCE_NONCLUSTERED_REFRESH_STARTING_ALPHA;
        }

        force.start().alpha(startingAlpha);

        if (graph.configuration.settings.clustered) {
          builtClusters = d3UtilitiesService.buildClusters(graph.nodes);
        }

        node = g.selectAll('.node')
          .data(graph.nodes)
          .enter()
          .append('g')
          .attr('class', 'node')
          .on('mouseover', d3_layout_forceMouseover)
          .on('mouseout', d3_layout_forceMouseout)
          .on('mouseup', mouseup)
          .call(drag);

        function mouseup(d) {
          if (!d3.event.metaKey) {
            if (d.dragMoved === undefined || !d.dragMoved) {
              toggleSelected(d);
            }
          } else {
            togglePinned(d);
          }
        }

        // Create the div element that will hold the context menu.
        d3.selectAll('.d3-context-menu').data([1]).enter().append('div').attr('class', 'd3-context-menu');

        // Close context menu.
        d3.select('body')
          .on('click.d3-context-menu', function() {
            d3.select('.d3-context-menu').style('display', 'none');
          });

        var attachIconToNode = function(gSelection, svgElement) {
          gSelection.append('rect')
            .attr('width', function(d) {
              var size = getBoundingBox(d)[0];
              return size > 10 ? size - 10 : size;
            })
            .attr('height', function(d) {
              var size = getBoundingBox(d)[1];
              return size > 10 ? size - 10 : size;
            })
            .attr('x', function(d) {
              var size = getBoundingBox(d)[0];
              return size > 10 ? 5 : 0;
            })
            .attr('y', function(d) {
              var size = getBoundingBox(d)[1];
              return size > 10 ? 5 : 0;
            })
            .style('stroke', 'white')
            .style('fill', 'white');

          var newElement = svgElement.cloneNode(true);
          gSelection.node().appendChild(newElement);
          var svgSelection = gSelection.select("svg");
          svgSelection
            .attr('width', function(d) {
              return getBoundingBox(d)[0];
            })
            .attr('height', function(d) {
              return getBoundingBox(d)[1];
            })
            .style('stroke', function(d) {
              return getNodeStroke(d);
            })
            .style('fill', function(d) {
              return getNodeFill(d);
            })
            .on('contextmenu', function(data, index) {
              d3UtilitiesService.showContextMenu(d3, data, index, nodeContextMenu);
            });
        };

        node.each(function(n) {
          var gSelection = d3.select(this);
          gSelection.attr('class', 'transform');
          var iconPath = n.icon;
          if (CONSTANTS.DEFAULTS.RENDER_NODE_ICONS && iconPath) {
            var nodeIconCacheEntry = nodeIconCache[iconPath];
            // Ignoring poossible race condition where d3.xml returns between 
            // the following two tests and the line that pushes the selection. 
            if (!nodeIconCacheEntry) {
              nodeIconCacheEntry = {
                "nodesWaitingForThisIcon": [this],
                "svgElement": undefined
              };
              nodeIconCache[iconPath] = nodeIconCacheEntry;
              d3.xml(iconPath, function(error, documentFragment) {
                if (error) {
                  console.log(error);
                  return;
                }

                var nodeIconCacheEntry = nodeIconCache[iconPath];
                if (nodeIconCacheEntry) {
                  var svgElement = documentFragment.getElementsByTagName("svg")[0];
                  if (svgElement) {
                    nodeIconCacheEntry.svgElement = svgElement;
                    _.forEach(nodeIconCacheEntry.nodesWaitingForThisIcon, function(node) {
                      attachIconToNode(d3.select(node), svgElement);
                    });
                  }
                }
              });
            } else {
              if (!nodeIconCacheEntry.svgElement) {
                nodeIconCacheEntry.nodesWaitingForThisIcon.push(this);
              } else {
                attachIconToNode(gSelection, nodeIconCacheEntry.svgElement);
              }
            }
          } else {
            gSelection.append('circle')
              .attr('r', function(d) {
                return d.radius;
              })
              .style('stroke', function(d) {
                return getNodeStroke(d);
              })
              .style('fill', function(d) {
                return d.fill || 'white';
              })
              .on('contextmenu', function(data, index) {
                d3UtilitiesService.showContextMenu(d3, data, index, nodeContextMenu);
              });
          }
        });

        transform = d3.selectAll('.transform');
        circle = g.selectAll('circle');

        var text = node.append('text')
          .attr('x', function(d) {
            var offset = getBoundingBox(d)[0];
            if (!CONSTANTS.DEFAULTS.RENDER_NODE_ICONS || !d.icon) {
              offset -= d.radius;
            }

            return offset;
          })
          .attr('y', function(d) {
            var offset = getBoundingBox(d)[1];
            if (!CONSTANTS.DEFAULTS.RENDER_NODE_ICONS || !d.icon) {
              offset -= d.radius;
            }

            return offset;
          });

        text.text(function(d) {
          return graph.configuration.settings.showNodeLabels && !d.hideLabel ? d.name : '';
        });

        text.each(function(e) {
          var singleText = d3.select(this);
          var parentNode = singleText.node().parentNode;

          d3.select(parentNode)
            .append('image')
            .attr('class', 'pin-icon')
            .attr('xlink:href', function(d) {
              return 'components/graph/img/Pin.svg';
            })
            .attr('display', function(d) {
              if (d.fixed) {
                if (d.fixed & CONSTANTS.FIXED_PINNED_BIT) {
                  return null;
                }
              }

              return 'none';
            })
            .attr('width', function(d) {
              return '13px';
            })
            .attr('height', function(d) {
              return '13px';
            })
            .attr('x', function(d) {
              var offset = -10;
              if (!CONSTANTS.DEFAULTS.RENDER_NODE_ICONS || !d.icon) {
                offset -= d.radius;
              }

              return offset;
            })
            .attr('y', function(d) {
              var offset = (getBoundingBox(d)[1] / 2) - 13;
              if (!CONSTANTS.DEFAULTS.RENDER_NODE_ICONS || !d.icon) {
                offset -= d.radius;
              }

              return offset;
            });
        });

        pin = d3.selectAll('.pin-icon');

        if (!graph.configuration.settings.clustered && graph.configuration.settings.showEdgeLabels) {
          edgepaths = g.selectAll('.edgepath')
            .data(graph.links)
            .enter()
            .append('path')
            .attr({
              d: function(d) {
                return 'M ' + d.source.x + ' ' + d.source.y + ' L ' + d.target.x + ' ' + d.target.y;
              },
              class: 'edgepath',
              'fill-opacity': 0,
              'stroke-opacity': 0,
              fill: 'blue',
              stroke: 'red',
              id: function(d, i) {
                return 'edgepath' + i;
              }
            })
            .style('pointer-events', 'none');

          edgelabels = g.selectAll('.edgelabel')
            .data(graph.links)
            .enter()
            .append('text')
            .style('pointer-events', 'none')
            .attr({
              class: 'edgelabel',
              id: function(d, i) {
                return 'edgelabel' + i;
              },
              dx: function(d) {
                return d.distance / 3;
              },
              dy: 0
            });

          edgelabels.append('textPath')
            .attr('xlink:href', function(d, i) {
              return '#edgepath' + i;
            })
            .style('pointer-events', 'none')
            .text(function(d, i) {
              return d.label;
            });
        }

        // If zero nodes are in the current selection, reset the selection.
        var nodeMatches = new Set();

        node.each(function(e) {
          if (d3UtilitiesService.setHas(selection.nodes, e)) {
            nodeMatches.add(e);
          }
        });

        if (!nodeMatches.size) {
          resetSelection();
        } else {
          selection.nodes = nodeMatches;
          selectEdgesInScope();
          applySelectionToOpacity();
        }

        // Create an array logging what is connected to what.
        var linkedByIndex = {};
        for (var i = 0; i < graph.nodes.length; i++) {
          // TODO: Should this be an array instead of a string map?
          linkedByIndex[i + ',' + i] = 1;
        }

        if (graph.links) {
          graph.links.forEach(function(d) {
            linkedByIndex[d.source.index + ',' + d.target.index] = 1;
          });
        }
      }

      function tick(e) {
        var forceAlpha = force.alpha();

        node.style('opacity', function(e) {
          if (e.opacity) {
            var opacity = e.opacity;
            delete e.opacity;
            return opacity;
          }

          return window.d3.select(this).style('opacity');
        });

        if (controllerScope.viewModelService.viewModel.data.configuration.settings.clustered) {
          circle.each(d3UtilitiesService.cluster(builtClusters, 10 * forceAlpha * forceAlpha))
            .each(d3UtilitiesService.collide(d3, controllerScope.viewModelService.viewModel.data.nodes,
              builtClusters, .5, clusterInnerPadding, clusterOuterPadding));

          pin.each(d3UtilitiesService.cluster(builtClusters, 10 * forceAlpha * forceAlpha))
            .each(d3UtilitiesService.collide(d3, controllerScope.viewModelService.viewModel.data.nodes,
              builtClusters, .5, clusterInnerPadding, clusterOuterPadding));
        } else {
          link
            .attr('x1', function(d) {
              var offsetX = (CONSTANTS.DEFAULTS.RENDER_NODE_ICONS && d.source.icon) ? getBoundingBox(d.source)[0] / 2 : 0;
              return d.source.x + offsetX;
            })
            .attr('y1', function(d) {
              var offsetY = (CONSTANTS.DEFAULTS.RENDER_NODE_ICONS && d.source.icon) ? getBoundingBox(d.source)[1] / 2 : 0;
              return d.source.y + offsetY;
            })
            .attr('x2', function(d) {
              var offsetX = (CONSTANTS.DEFAULTS.RENDER_NODE_ICONS && d.target.icon) ? getBoundingBox(d.target)[0] / 2 : 0;
              return d.target.x + offsetX;
            })
            .attr('y2', function(d) {
              var offsetY = (CONSTANTS.DEFAULTS.RENDER_NODE_ICONS && d.target.icon) ? getBoundingBox(d.target)[1] / 2 : 0;
              return d.target.y + offsetY;
            });

          if (edgepaths) {
            edgepaths.attr('d', function(d) {
              var path = 'M ' + d.source.x + ' ' + d.source.y + ' L ' + d.target.x + ' ' + d.target.y;
              return path
            });

            edgelabels.attr('transform', function(d, i) {
              if (d.target.x < d.source.x) {
                var bbox = this.getBBox();
                var rx = bbox.x + bbox.width / 2;
                var ry = bbox.y + bbox.height / 2;

                return 'rotate(180 ' + rx + ' ' + ry + ')';
              } else {
                return 'rotate(0)';
              }
            });
          }
        }

        transform.attr('transform', function(d) {
          return sprintf('translate(%s,%s)', d.x, d.y);
        });

        pin.attr('display', function(d) {
          return d.fixed & CONSTANTS.FIXED_PINNED_BIT ? null : 'none';
        });

        if (forceAlpha < 0.04) {
          controllerScope.viewModelService.viewModel.data.nodes.forEach(function(n) {
            if (n.id) {
              if (!nodeSettingsCache[n.id]) {
                nodeSettingsCache[n.id] = {};
              }

              nodeSettingsCache[n.id].position = [n.x, n.y];
            }
          });
        }
      }

      // Get or set the node settings cache. Returns the rendering service when acting as a setter.
      graph.nodeSettingsCache = function(newNodeSettingsCache) {
        if (!arguments.length) return nodeSettingsCache;
        nodeSettingsCache = newNodeSettingsCache;

        return this;
      };

      // Get or set the view settings cache. Returns the rendering service when acting as a setter.
      graph.viewSettingsCache = function(newViewSettingsCache) {
        if (!arguments.length) return viewSettingsCache;
        viewSettingsCache = newViewSettingsCache;

        return this;
      };

      function zoomed() {
        var translate = zoom.translate();
        var scale = zoom.scale();

        g.attr('transform', 'translate(' + translate + ')scale(' + scale + ')');

        viewSettingsCache.translate = translate;
        viewSettingsCache.scale = scale;
      }

      function adjustZoom(factor) {
        var scale = zoom.scale(),
          extent = zoom.scaleExtent(),
          translate = zoom.translate(),
          x = translate[0],
          y = translate[1],
          target_scale = scale * factor;

        var reset = !factor;

        if (reset) {
          target_scale = 1;
          factor = target_scale / scale;
        }

        // If we're already at an extent, done.
        if (target_scale === extent[0] || target_scale === extent[1]) {
          return false;
        }
        // If the factor is too much, scale it down to reach the extent exactly.
        var clamped_target_scale = Math.max(extent[0], Math.min(extent[1], target_scale));
        if (clamped_target_scale != target_scale) {
          target_scale = clamped_target_scale;
          factor = target_scale / scale;
        }

        // Center each vector, stretch, then put back.
        x = (x - center[0]) * factor + center[0];
        y = (y - center[1]) * factor + center[1];

        if (reset) {
          x = 0;
          y = 0;
        }

        // Transition to the new view over 350ms
        window.d3.transition().duration(350).tween('zoom', function() {
          var interpolate_scale = window.d3.interpolate(scale, target_scale);
          var interpolate_trans = window.d3.interpolate(translate, [x, y]);

          return function(t) {
            zoom.scale(interpolate_scale(t)).translate(interpolate_trans(t));

            zoomed();
          };
        });
      }
      graph.adjustZoom = adjustZoom;

      return graph;
    }

    return {
      rendering: rendering
    };
  };

  angular.module('kubernetesApp.components.graph.services.d3.rendering', [])
    .service('d3RenderingService', ['lodash', 'd3UtilitiesService', '$location', '$rootScope', 'inspectNodeService', d3RenderingService]);

})();