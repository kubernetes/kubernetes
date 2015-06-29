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
function templateTransform(_, template) {
  var stringifyNoQuotes = function(result) {
    if (typeof result !== "string") {
      if (result !== "undefined") {
        result = JSON.stringify(result);
        result = result.replace(/\"([^(\")"]+)\":/g, "$1:");
      } else {
        result = "undefined";
      }
    }

    return result;
  };

  var evalExpression = function(item, expression) {
    var result = undefined;
    var expr = expression.eval;
    if (typeof expr === "string") {
      var args = [];
      if (_.isArray(expression.args)) {
        _.forEach(expression.args, function(results) {
          if (typeof results === "string") {
            if (results.charAt(0) === "$") {
              results = JSONPath(null, item, results);
            }
            if (results && results.length > 0) {
              results = _.map(results, function(result) { return stringifyNoQuotes(result); });
              args.push(results);
            }
          }
        });

        expr = vsprintf(expr, args);
      }

      result = eval(expr);
    }

    return result;
  };

  var filterItem = function(filters) {
    return function(fromItem) {
      return _.every(filters, function(filter) { return evalExpression(fromItem, filter); });
    };
  };

  var mapItem = function(fromItem, toItem, maps) {
    var random = function() {
      var str = JSON.stringify(Math.random());
      var idx = str.indexOf(".") + 1;
      if (idx > 0 && idx < str.length) {
        str = str.substring(idx);
      }

      return str;
    };

    var mapProperties = function(fromItem, toItem, properties) {
      var evalMapping = function(item, mapping) {
        var result = undefined;
        if (mapping) {
          if (typeof mapping === "string") {
            if (mapping.charAt(0) === "$") {
              results = JSONPath(null, item, mapping);
              if (results && results.length > 0) {
                if (results.length === 1) {
                  result = results[0];
                }
              }
            } else {
              result = mapping;
            }
          } else if (typeof mapping === "object") {
            if (mapping.expression) {
              result = evalExpression(item, mapping);
            } else {
              result = mapProperties(item, {}, mapping);
            }
          } else if (_.isArray(mapping)) {
            result = _.map(mapping, function(member) { return evalMapping(item, member); });
          }
        }

        return result;
      };

      if (properties) {
        _.forOwn(properties, function(mapping, property) {
          mapping = evalMapping(fromItem, mapping);
          if (mapping) {
            property = evalMapping(fromItem, property);
            if (property) {
              property = stringifyNoQuotes(property);
              toItem[property] = mapping;
            }
          }
        });
      }

      return toItem;
    };

    toItem.id = fromItem.id || random();
    if (maps) {
      // TODO: Apply maps progressively not sequentially.
      _.forEach(maps, function(map) {
        if (!map.filter || evalExpression(fromItem, map.filter)) {
          mapProperties(fromItem, toItem, map.properties);
        }
      });
    }
  };

  var mapNodes = function(fromNodes, toData) {
    var mapNode = function(fromNode) {
      var toNode = {};
      mapItem(fromNode, toNode, template.nodeMaps);
      return toNode;
    };

    var sortNode = function(fromNode) { return fromNode.id; };

    var chain = _.chain(fromNodes);
    if (template.nodeFilters) {
      chain = chain.filter(filterItem(template.nodeFilters));
    }

    toData.nodes = chain.map(mapNode).sortBy(sortNode).value();
  };

  var mapLinks = function(fromLinks, toData) {
    var mapLink = function(fromLink) {
      var toLink = {};
      mapItem(fromLink, toLink, template.edgeMaps);
      return toLink;
    };

    var sortLink = function(fromLink) { return fromLink.source + ":" + fromLink.target; };

    var chain = _.chain(fromLinks);
    if (template.edgeFilters) {
      chain = chain.filter(filterItem(template.edgeFilters));
    }

    toData.links = chain.map(mapLink).sortBy(sortLink).value();
  };

  return function(fromData, configuration) {
    var toData = {};
    toData.configuration = configuration;
    if (template.configuration) {
      if (template.configuration.legend) {
        toData.configuration.legend = template.configuration.legend;
      }

      if (template.configuration.settings) {
        toData.configuration.settings = template.configuration.settings;
      }

      if (template.configuration.selectionHops) {
        toData.configuration.selectionHops = template.configuration.selectionHops;
      }

      if (template.configuration.selectionIdList) {
        toData.configuration.selectionIdList = template.configuration.selectionIdList;
      }
    }

    if (fromData) {
      if (fromData.resources) {
        mapNodes(fromData.resources, toData);
        if (fromData.relations) {
          mapLinks(fromData.relations, toData);
        }
      }
    }

    return toData;
  };
}
