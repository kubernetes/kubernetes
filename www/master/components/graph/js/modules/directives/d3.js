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
 * This is a directive that uses d3 to generate an svg
 * element.
 =========================================================*/

angular.module('kubernetesApp.components.graph')
    .directive('d3Visualization', [
      'd3Service',
      'd3RenderingService',
      function(d3Service, d3RenderingService) {
        return {
          restrict: 'E',
          link: function(scope, element, attrs) {
            scope.$watch('viewModelService.viewModel.version', function(newValue, oldValue) {
              if (!window.d3) {
                d3Service.d3().then(d3Rendering);
              } else {
                d3Rendering();
              }
            });

            scope.$watch('selectionIdList', function(newValue, oldValue) {
              if (newValue !== undefined) {
                // The d3Rendering.nodeSelection() method expects a set of objects, each with an id property.
                var nodes = new Set();

                newValue.forEach(function(e) { nodes.add({id: e}); });

                d3Rendering.nodeSelection(nodes);
              }
            });

            var d3Rendering = d3RenderingService.rendering().controllerScope(scope).directiveElement(element[0]);
          }
        };
      }
    ]);
