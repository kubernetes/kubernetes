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

  // Compute the view model based on the data model and control parameters
  // and place the result in the current scope at $scope.viewModel.
  var viewModelService = function ViewModelService(_) {
    var defaultConfiguration = {
      "legend": undefined,
      "settings": {
        "clustered": false,
        "showEdgeLabels": false,
        "showNodeLabels": true
      },
      "selectionHops": 1,
      "selectionIdList": []
    };

    var defaultNode = {
      name: 'no data'
    };

    var defaultData = {
      "configuration": defaultConfiguration,
      "nodes": [],
      "links": []
    };

    // Load the default data.
    var loadDefaultData = function(defaultLegend) {
      if (defaultLegend && defaultLegend.nodes && defaultLegend.nodes.Cluster) {
        var legendEntry = defaultLegend.nodes.Cluster;
        if (legendEntry) {
          var legendStyle = legendEntry.style;
          if (legendStyle) {
            _.assign(defaultNode, legendStyle);
          }
        }
      }

      if (defaultData.nodes.length < 1) {
        defaultData.nodes = [defaultNode];
      }
    };

    // Load the default legend.
    (function() {
      $.getJSON("components/graph/assets/legend.json")
        .done(function(legend) {
          defaultData.configuration.legend = legend;
          loadDefaultData(legend);
        })
        .fail(function(jqxhr, settings, exception) {
          console.log('ERROR: Could not load default legend: ' + exception);
        });
    }());

    var viewModel = {
      "data": undefined,
      "default": undefined,
      "version": 0,
      "transformNames": []
    };

    var getViewModelData = function() {
      if (!viewModel.data || viewModel.data.nodes.length < 1) {
        viewModel.data = defaultData;
      }

      if (!viewModel.default || viewModel.default.nodes.length < 1) {
        viewModel.default = defaultData;
      }

      return viewModel.data;
    };

    var getViewModelConfiguration = function() {
      var data = getViewModelData();
      return data ? data.configuration : undefined;
    };

    var getLegend = function() {
      return (getViewModelConfiguration()) ?
        viewModel.data.configuration.legend :
        undefined;
    };

    var setLegend = function(legend) {
      if (getViewModelConfiguration()) {
        viewModel.data.configuration.legend = legend;
      }
    };

    var getSettings = function() {
      return (getViewModelConfiguration()) ?
        viewModel.data.configuration.settings :
        undefined;
    };

    var setSettings = function(settings) {
      if (getViewModelConfiguration()) {
        viewModel.data.configuration.settings = settings;
      }
    };

    var getSelectionHops = function() {
      return (getViewModelConfiguration()) ?
        viewModel.data.configuration.selectionHops :
        1;
    };

    var setSelectionHops = function(selectionHops) {
      if (getViewModelConfiguration()) {
        viewModel.data.configuration.selectionHops = selectionHops;
      }
    };

    var getSelectionIdList = function() {
      return (getViewModelConfiguration()) ?
        viewModel.data.configuration.selectionIdList : [];
    };

    var setSelectionIdList = function(selectionIdList) {
      if (getViewModelConfiguration()) {
        viewModel.data.configuration.selectionIdList = selectionIdList;
      }
    };

    var defaultTransformName = undefined;
    var transformsByName = {};

    // Load transforms.
    (function() {
      var stripSuffix = function(fileName) {
        var suffixIndex = fileName.indexOf(".");
        if (suffixIndex > 0) {
          fileName = fileName.substring(0, suffixIndex);
        }

        return fileName;
      };

      var getConstructor = function(constructorName) {
        return window[constructorName];
      };

      var bindTransform = function(constructorName, directoryEntry) {
        var constructor = getConstructor(constructorName);
        if (constructor) {
          var transform = constructor(_, directoryEntry.data);
          if (transform) {
            if (!defaultTransformName) {
              defaultTransformName = directoryEntry.name;
            }

            viewModel.transformNames.push(directoryEntry.name);
            transformsByName[directoryEntry.name] = transform;
            return;
          }
        }

        console.log('ERROR: Could not bind transform "' + directoryEntry.name + '".');
      };

      // Load a transform from a given directory entry.
      var loadTransform = function(directoryEntry) {
        if (directoryEntry && directoryEntry.name && directoryEntry.script) {
          var constructorName = stripSuffix(directoryEntry.script);
          if (!getConstructor(constructorName)) {
            // Load the script into the window scope.
            var scriptPath = "components/graph/assets/transforms/" + directoryEntry.script;
            $.getScript(scriptPath)
              .done(function() {
                // Defer to give the load opportunity to complete.
                _.defer(function() {
                  bindTransform(constructorName, directoryEntry);
                });
              })
              .fail(function(jqxhr, settings, exception) {
                console.log('ERROR: Could not load transform "' + directoryEntry.name + '": ' + exception);
              });
          } else {
            bindTransform(constructorName, directoryEntry);
          }
        }
      };

      // Load the transform directory
      $.getJSON("components/graph/assets/transforms.json")
        .done(function(transforms) {
          // Defer to give the load opportunity to complete.
          _.defer(function() {
            if (transforms.directory) {
              _.forEach(transforms.directory, function(directoryEntry) {
                loadTransform(directoryEntry);
              });
            }
          });
        })
        .fail(function(jqxhr, settings, exception) {
          console.log('ERROR: Could not load transform directory: ' + exception);
        });
    }());

    var setViewModel = function(data) {
      if (data && data.nodes && data.configuration && data.configuration.settings) {
        viewModel.data = data;
        viewModel.version++;
      }
    };

    // Generate the view model from a given data model using a given transform.
    var generateViewModel = function(fromData, transformName) {
      var initializeConfiguration = function(toData) {
        var initializeLegend = function(fromConfiguration, toConfiguration) {
          var toLegend = toConfiguration.legend;
          var fromLegend = fromConfiguration.legend;
          if (!toLegend) {
            toConfiguration.legend = JSON.parse(JSON.stringify(fromLegend));
          } else {
            if (!toLegend.nodes) {
              toLegend.nodes = JSON.parse(JSON.stringify(fromLegend.nodes));
            }

            if (!toLegend.links) {
              toLegend.links = JSON.parse(JSON.stringify(fromLegend.links));
            }
          }
        };

        var initializeSettings = function(fromConfiguration, toConfiguration) {
          if (!toConfiguration.settings) {
            toConfiguration.settings = JSON.parse(JSON.stringify(fromConfiguration.settings));
          }
        };

        var initializeSelection = function(fromConfiguration, toConfiguration) {
          if (!toConfiguration.selectionHops) {
            toConfiguration.selectionHops = fromConfiguration.selectionHops;
          }

          if (!toConfiguration.selectionIdList) {
            toConfiguration.selectionIdList = fromConfiguration.selectionIdList;
          }
        };

        var toConfiguration = toData.configuration;
        var fromConfiguration = viewModel.data.configuration;

        if (!toConfiguration) {
          toData.configuration = JSON.parse(JSON.stringify(fromConfiguration));
        } else {
          initializeLegend(fromConfiguration, toConfiguration);
          initializeSettings(fromConfiguration, toConfiguration);
          initializeSelection(fromConfiguration, toConfiguration);
        }
      };

      var processNodes = function(toData) {
        var typeToCluster = {};
        var idToIndex = {};

        var setIndex = function(toNode, idToIndex) {
          if (!idToIndex[toNode.id]) {
            idToIndex[toNode.id] = _.keys(idToIndex).length;
          }
        };

        var setCluster = function(toNode, typeToCluster) {
          if (toNode.type) {
            toNode.cluster = typeToCluster[toNode.type];
            if (toNode.cluster === undefined) {
              toNode.cluster = _.keys(typeToCluster).length;
              typeToCluster[toNode.type] = toNode.cluster;
            }
          } else {
            toNode.cluster = 0;
          }
        };

        var setStyle = function(toItem, entries) {
          if (toItem.type && entries[toItem.type]) {
            _.assign(toItem, entries[toItem.type].style);
            entries[toItem.type].available = true;
          } else {
            toItem.type = undefined;
          }
        };

        var processLinks = function(toData, legend, filtered) {
          var getIndex = function(toLink, idToIndex) {
            if (toLink.source && toLink.target) {
              toLink.source = idToIndex[toLink.source];
              toLink.target = idToIndex[toLink.target];
            }
          };

          var chain = _.chain(toData.links)
            .forEach(function(toLink) {
              setStyle(toLink, legend.links);
              if (toLink.type) {
                getIndex(toLink, idToIndex);
              }
            });

          chain = chain.filter("type");
          if (filtered) {
            chain = chain.filter(function(toLink) {
              return (toLink.source !== undefined) && (toLink.target !== undefined);
            });
          }

          toData.links = chain.value();
        };

        var configuration = toData.configuration;
        var legend = configuration.legend;
        var settings = configuration.settings;

        _.forOwn(legend.nodes, function(nodeEntry) {
          nodeEntry.available = false;
        });

        var chain = _.chain(toData.nodes).forEach(function(toNode) {
          setStyle(toNode, legend.nodes);
        });

        var filtered = _.any(legend.nodes, function(nodeEntry) {
          return !nodeEntry.selected;
        });

        chain = chain.filter("type");
        if (filtered) {
          chain = chain.filter(function(toNode) {
            return legend.nodes[toNode.type] ? legend.nodes[toNode.type].selected : false;
          });
        }

        if (settings && settings.clustered) {
          chain = chain.forEach(function(toNode) {
            setCluster(toNode, typeToCluster);
          });
        }

        toData.nodes = chain.forEach(function(toNode) {
          setIndex(toNode, idToIndex);
        }).value();

        if (toData.links) {
          processLinks(toData, legend, filtered);
        }
      };

      if (fromData && transformName) {
        var transform = transformsByName[transformName];
        if (transform) {
          var configuration = JSON.parse(JSON.stringify(viewModel.data.configuration));
          var toData = transform(fromData, configuration);
          if (toData.nodes) {
            initializeConfiguration(toData);
            processNodes(toData);
          } else {
            toData.configuration = configuration;
            toData.nodes = defaultData.nodes;
            toData.links = defaultData.links;
          }

          setViewModel(toData);
        } else {
          console.log('ERROR: Could not find transform "' + transformName + '".');
        }
      }
    };

    this.$get = function() {
      return {
        "viewModel": viewModel,
        "getLegend": getLegend,
        "setLegend": setLegend,
        "getSettings": getSettings,
        "setSettings": setSettings,
        "getSelectionIdList": getSelectionIdList,
        "setSelectionIdList": setSelectionIdList,
        "getSelectionHops": getSelectionHops,
        "setSelectionHops": setSelectionHops,
        "defaultTransformName": defaultTransformName,
        "generateViewModel": generateViewModel,
        "setViewModel": setViewModel
      };
    };
  };

  angular.module("kubernetesApp.components.graph").provider("viewModelService", ["lodash", viewModelService]);
}());