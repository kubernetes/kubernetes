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
* Visualizer for force directed graph
=========================================================*/
(function() {
  "use strict";
  angular.module("kubernetesApp.components.graph", [
    "kubernetesApp.services",
    "kubernetesApp.components.graph.services",
    "kubernetesApp.components.graph.services.d3",
    "kubernetesApp.components.graph.services.d3.rendering",
    "yaru22.jsonHuman"
  ])
      .controller("GraphCtrl", [
        "$scope",
        "lodash",
        "viewModelService",
        "pollK8sDataService",
        "$location",
        "$window",
        "inspectNodeService",
        function($scope, _, viewModelService, pollK8sDataService, $location, $window, inspectNodeService) {
          $scope.showHide = function(id) {
            var element = document.getElementById(id);
            if (element) {
              element.style.display = (element.style.display === "none") ? "block" : "none";
            }
          };

          $scope.showElement = function(id) {
            var element = document.getElementById(id);
            if (element) {
              element.style.display = "block";
            }
          };

          $scope.hideElement = function(id) {
            var element = document.getElementById(id);
            if (element) {
              element.style.display = "none";
            }
          };

          $scope.pollK8sDataService = pollK8sDataService;

          $scope.getPlayIcon = function() {
            return pollK8sDataService.isPolling() ? "components/graph/img/Pause.svg" : "components/graph/img/Play.svg";
          };

          $scope.togglePlay = function() {
            if (pollK8sDataService.isPolling()) {
              pollK8sDataService.stop($scope);
            } else {
              pollK8sDataService.start($scope);
            }
          };

          // Update the view when the polling starts or stops.
          $scope.$watch("pollK8sDataService.isPolling()", function(newValue, oldValue) {
            if (newValue !== oldValue) {
              $scope.$apply();
            }
          });

          $scope.getSourceIcon = function() {
            return pollK8sDataService.k8sdatamodel.useSampleData ? "components/graph/img/SampleData.svg" :
                                                                   "components/graph/img/LiveData.svg";
          };

          $scope.toggleSource = function() {
            pollK8sDataService.k8sdatamodel.useSampleData = !pollK8sDataService.k8sdatamodel.useSampleData;
            pollK8sDataService.refresh($scope);
          };

          $scope.refresh = function() { pollK8sDataService.refresh($scope); };

          $scope.viewModelService = viewModelService;
          $scope.getTransformNames = function() { return _.sortBy(viewModelService.viewModel.transformNames); };

          $scope.selectedTransformName = viewModelService.defaultTransformName;

          // Sets the selected transform name
          $scope.setSelectedTransformName = function(transformName) {
            pollK8sDataService.stop($scope);
            $scope.selectedTransformName = transformName;
            $scope.updateModel();
          };

          $scope.updateModel = function() {
            viewModelService.generateViewModel(pollK8sDataService.k8sdatamodel.data, $scope.selectedTransformName);
          };

          // Update the view model when the data model changes.
          $scope.$watch("pollK8sDataService.k8sdatamodel.sequenceNumber", function(newValue, oldValue) {
            if (newValue !== oldValue) {
              $scope.updateModel();
            }
          });

          pollK8sDataService.refresh($scope);

          $scope.getLegendNodeTypes = function() {
            var result = [];
            var legend = viewModelService.getLegend();
            if (legend && legend.nodes) {
              result = _.keys(legend.nodes)
                .filter(function(type) { return legend.nodes[type].available; })
                .sort();
            }

            return result;
          };

          $scope.getLegendNodeDisplayName = function(type) {
            var result = type;
            var legend = viewModelService.getLegend();
            if (legend && legend.nodes && legend.nodes[type] && legend.nodes[type].displayName) {
              result = legend.nodes[type].displayName;
            }

            return result;
          };

          $scope.getLegendNodeFill = function(type) {
            var result = "white";
            var legend = viewModelService.getLegend();
            if (legend && legend.nodes && legend.nodes[type]) {
              if (legend.nodes[type].selected) {
                if (legend.nodes[type].style) {
                  result = legend.nodes[type].style.fill;
                }
              }
            }

            return result;
          };

          $scope.getLegendNodeStroke = function(type) {
            var result = "dimgray";
            var legend = viewModelService.getLegend();
            if (legend && legend.nodes && legend.nodes[type]) {
              if (legend.nodes[type].style.stroke) {
                result = legend.nodes[type].style.stroke;
              }
            }

            return result;
          };

          $scope.getLegendNodeStrokeWidth = function(type) {
            var result = "1";
            var legend = viewModelService.getLegend();
            if (legend && legend.nodes && legend.nodes[type]) {
              if (legend.nodes[type].style.strokeWidth) {
                result = legend.nodes[type].style.strokeWidth;
              }
            }

            return result;
          };

          $scope.getLegendNodeIcon = function(type) {
            var result = null;
            var legend = viewModelService.getLegend();
            if (legend && legend.nodes && legend.nodes[type]) {
              if (legend.nodes[type].style.icon) {
                result = legend.nodes[type].style.icon;
              }
            }

            return result;
          };

          $scope.getLegendLinkTypes = function() {
            var result = [];
            var legend = viewModelService.getLegend();
            if (legend && legend.links) {
              result = _.keys(legend.links).filter(function(type) { return legend.links[type].available; }).sort();
            }

            return result;
          };

          $scope.getLegendLinkStyle = function(type) {
            var result = {};
            var legend = viewModelService.getLegend();
            if (legend && legend.links) {
              result = legend.links[type].style;
            }

            return result;
          };

          $scope.getLegendLinkStyleStrokeWidth = function(type, defaultWidth) {
            var style = $scope.getLegendLinkStyle(type);
            return $window.Math.max(style.width, defaultWidth);
          };

          $scope.toggleLegend = function(type) {
            if (type) {
              var legend = viewModelService.getLegend();
              if (legend.nodes) {
                legend.nodes[type].selected = !legend.nodes[type].selected;
                $scope.updateModel();
              }
            }
          };

          var getSelection = function() {
            var selectedNode = undefined;
            var selectionIdList = viewModelService.getSelectionIdList();
            if (selectionIdList && selectionIdList.length > 0) {
              var selectedId = selectionIdList[0];
              selectedNode =
                  _.find(viewModelService.viewModel.data.nodes, function(node) { return node.id === selectedId; });
            }

            return selectedNode;
          };

          var stringifyNoQuotes = function(result) {
            if (typeof result !== "string") {
              if (result !== "undefined") {
                result = JSON.stringify(result, null, 2);
                result = result.replace(/\"([^(\")"]+)\":/g, "$1:");
              } else {
                result = "undefined";
              }
            }

            return result;
          };

          $scope.getSelectionDetails = function() {
            var results = {};
            var selectedNode = getSelection();
            if (selectedNode && selectedNode.tags) {
              _.forOwn(selectedNode.tags, function(value, property) {
                if (value) {
                  var result = stringifyNoQuotes(value);
                  if (result.length > 0) {
                    results[property] = result.trim();
                  }
                }
              });
            }

            return results;
          };

          $scope.inspectSelection = function() {
            var selectedNode = getSelection();
            if (selectedNode && selectedNode.metadata) {
              inspectNodeService.setDetailData(selectedNode);
              $location.path('/graph/inspect');
            }
          };

          $scope.$watch("viewModelService.getSelectionIdList()", function(newValue, oldValue) {
            if (newValue !== oldValue) {
              var selectionIdList = viewModelService.getSelectionIdList();
              if (!selectionIdList || selectionIdList.length < 1) {
                $scope.hideElement("details");
              } else {
                $scope.showElement("details");
              }
            }
          });

          $scope.getExpandIcon = function() {
            return viewModelService.getSettings().clustered ? "components/graph/img/Collapse.svg" :
                                                              "components/graph/img/Expand.svg";
          };

          $scope.toggleExpand = function() {
            var settings = viewModelService.getSettings();
            settings.clustered = !settings.clustered;
            $scope.updateModel();
          };

          $scope.getSelectIcon = function() {
            return viewModelService.getSelectionHops() ? "components/graph/img/SelectMany.svg" :
                                                         "components/graph/img/SelectOne.svg";
          };

          $scope.toggleSelect = function() {
            var selectionHops = viewModelService.getSelectionHops();
            if (!selectionHops) {
              viewModelService.setSelectionHops(1);
            } else {
              viewModelService.setSelectionHops(0);
            }
          };
        }
      ]);
}());
