(function() {
  'use strict';

  angular.module('kubernetesApp.components.dashboard')
      .directive('d3MinionBarGauge', [
        'd3DashboardService',
        function(d3DashboardService) {

          return {
            restrict: 'E',
            scope: {
              data: '=',
              thickness: '@',
              graphWidth: '@',
              graphHeight: '@'

            },
            link: function(scope, element, attrs) {

              var draw = function(d3) {
                var svg = d3.select("svg.chart");
                var legendSvg = d3.select("svg.legend");
                window.onresize = function() { return scope.$apply(); };

                scope.$watch(function() { return angular.element(window)[0].innerWidth; },
                             function() { return scope.render(scope.data); });

                scope.$watch('data', function(newVals, oldVals) {
                  return initOrUpdate(newVals, oldVals);

                }, true);

                function initOrUpdate(newVals, oldVals) {
                  if (oldVals === null || oldVals === undefined) {
                    return scope.render(newVals);
                  } else {
                    return update(oldVals, newVals);
                  }
                }

                var textOffset = 10;
                var el = null;
                var radius = 100;
                var oldData = [];

                function init(options) {
                  var clone = options.data;
                  var preparedData = setData(clone);
                  setup(preparedData, options.width, options.height);
                }

                function setup(data, w, h) {
                  svg = d3.select(element[0]).append("svg").attr("width", "100%");

                  legendSvg = d3.select(element[0]).append("svg").attr("width", "100%");

                  var chart = svg.attr("class", "chart")
                                  .attr("width", w)
                                  .attr("height", h - 25)
                                  .append("svg:g")
                                  .attr("class", "concentricchart")
                                  .attr("transform", "translate(" + ((w / 2)) + "," + h / 4 + ")");

                  var legend = legendSvg.attr("class", "legend").attr("width", w);

                  radius = Math.min(w, h) / 2;

                  var hostName = legendSvg.append("text")
                                     .attr("class", "hostName")
                                     .attr("transform", "translate(" + ((w - 120) / 2) + "," + 15 + ")");

                  var label_legend_area = legendSvg.append("svg:g")
                                              .attr("class", "label_legend_area")
                                              .attr("transform", "translate(" + ((w - 185) / 2) + "," + 35 + ")");

                  var legend_group = label_legend_area.append("svg:g").attr("class", "legend_group");

                  var label_group = label_legend_area.append("svg:g")
                                        .attr("class", "label_group")
                                        .attr("transform", "translate(" + 25 + "," + 11 + ")");

                  var stats_group = label_legend_area.append("svg:g")
                                        .attr("class", "stats_group")
                                        .attr("transform", "translate(" + 85 + "," + 11 + ")");

                  var path_group = chart.append("svg:g")
                                       .attr("class", "path_group")
                                       .attr("transform", "translate(0," + (h / 4) + ")");
                  var value_group = chart.append("svg:g")
                                        .attr("class", "value_group")
                                        .attr("transform", "translate(" + -(w * 0.205) + "," + -(h * 0.10) + ")");
                  generateArcs(chart, data);
                }

                function update(_oldData, _newData) {
                  if (_newData === undefined || _newData === null) {
                    return;
                  }

                  var clone = jQuery.extend(true, {}, _newData);
                  var cloneOld = jQuery.extend(true, {}, _oldData);
                  var preparedData = setData(clone);
                  oldData = setData(cloneOld);
                  animate(preparedData);
                }

                function animate(data) { generateArcs(null, data); }

                function setData(data) {
                  var diameter = 2 * Math.PI * radius;
                  var localData = [];

                  $.each(data[0].segments, function(ri, value) {

                    function calcAngles(v) {
                      var segmentValueSum = 200;
                      if (v > segmentValueSum) {
                        v = segmentValueSum;
                      }

                      var segmentValue = v;
                      var fraction = segmentValue / segmentValueSum;
                      var arcBatchLength = fraction * 4 * Math.PI;
                      var arcPartition = arcBatchLength;
                      var startAngle = Math.PI * 2;
                      var endAngle = startAngle + arcPartition;

                      return {
                        startAngle: startAngle,
                        endAngle: endAngle
                      };
                    }

                    var valueData = calcAngles(value.value);
                    data[0].segments[ri].startAngle = valueData.startAngle;
                    data[0].segments[ri].endAngle = valueData.endAngle;

                    var maxData = value.maxData;
                    var maxTickData = calcAngles(maxData.maxValue + 0.2);
                    data[0].segments[ri].maxTickStartAngle = maxTickData.startAngle;
                    data[0].segments[ri].maxTickEndAngle = maxTickData.endAngle;

                    var maxArcData = calcAngles(maxData.maxValue);
                    data[0].segments[ri].maxArcStartAngle = maxArcData.startAngle;
                    data[0].segments[ri].maxArcEndAngle = maxArcData.endAngle;

                    data[0].segments[ri].index = ri;
                  });
                  localData.push(data[0].segments);
                  return localData[0];
                }

                function generateArcs(_svg, data) {
                  var chart = svg;
                  var transitionTime = 750;
                  $.each(data, function(index, value) {
                    if (oldData[index] !== undefined) {
                      data[index].previousEndAngle = oldData[index].endAngle;
                    } else {
                      data[index].previousEndAngle = 0;
                    }
                  });
                  var thickness = parseInt(scope.thickness, 10);
                  var ir = (parseInt(scope.graphWidth, 10) / 3);
                  var path_group = svg.select('.path_group');
                  var arc_group = path_group.selectAll(".arc_group").data(data);
                  var arcEnter = arc_group.enter().append("g").attr("class", "arc_group");

                  arcEnter.append("path").attr("class", "bg-circle").attr("d", getBackgroundArc(thickness, ir));

                  arcEnter.append("path")
                      .attr("class", function(d, i) { return 'max_tick_arc ' + d.maxData.maxTickClassNames; });

                  arcEnter.append("path")
                      .attr("class", function(d, i) { return 'max_bg_arc ' + d.maxData.maxClassNames; });

                  arcEnter.append("path").attr("class", function(d, i) { return 'value_arc ' + d.classNames; });

                  var max_tick_arc = arc_group.select(".max_tick_arc");

                  max_tick_arc.transition()
                      .attr("class", function(d, i) { return 'max_tick_arc ' + d.maxData.maxTickClassNames; })
                      .attr("d", function(d) {
                        var arc = maxArc(thickness, ir);
                        arc.startAngle(d.maxTickStartAngle);
                        arc.endAngle(d.maxTickEndAngle);
                        return arc(d);
                      });

                  var max_bg_arc = arc_group.select(".max_bg_arc");

                  max_bg_arc.transition()
                      .attr("class", function(d, i) { return 'max_bg_arc ' + d.maxData.maxClassNames; })
                      .attr("d", function(d) {
                        var arc = maxArc(thickness, ir);
                        arc.startAngle(d.maxArcStartAngle);
                        arc.endAngle(d.maxArcEndAngle);
                        return arc(d);
                      });

                  var value_arc = arc_group.select(".value_arc");

                  value_arc.transition().ease("exp").attr("class", function(d, i) {
                    return 'value_arc ' + d.classNames;
                  }).duration(transitionTime).attrTween("d", function(d) { return arcTween(d, thickness, ir); });

                  arc_group.exit()
                      .select(".value_arc")
                      .transition()
                      .ease("exp")
                      .duration(transitionTime)
                      .attrTween("d", function(d) { return arcTween(d, thickness, ir); })
                      .remove();

                  drawLabels(chart, data, ir, thickness);
                  buildLegend(chart, data);
                }

                function arcTween(b, thickness, ir) {
                  var prev = JSON.parse(JSON.stringify(b));
                  prev.endAngle = b.previousEndAngle;
                  var i = d3.interpolate(prev, b);
                  return function(t) { return getArc(thickness, ir)(i(t)); };
                }

                function maxArc(thickness, ir) {
                  var arc = d3.svg.arc().innerRadius(function(d) {
                    return getRadiusRing(ir, d.index);
                  }).outerRadius(function(d) { return getRadiusRing(ir + thickness, d.index); });
                  return arc;
                }

                function drawLabels(chart, data, ir, thickness) {
                  svg.select('.value_group').selectAll("*").remove();
                  var counts = data.length;
                  var value_group = chart.select('.value_group');
                  var valueLabels = value_group.selectAll("text.value").data(data);
                  valueLabels.enter()
                      .append("svg:text")
                      .attr("class", "value")
                      .attr(
                           "transform", function(d) { return "translate(" + (getRadiusRing(ir, counts - 1)) + ", 0)"; })
                      .attr("dx", function(d, i) { return 0; })
                      .attr("dy", function(d, i) { return (thickness + 3) * i; })
                      .attr("text-anchor", function(d) { return "start"; })
                      .text(function(d) { return d.value; });
                  valueLabels.transition().duration(300).attrTween(
                      "d", function(d) { return arcTween(d, thickness, ir); });
                  valueLabels.exit().remove();
                }

                function buildLegend(chart, data) {
                  var svg = legendSvg;
                  svg.select('.label_group').selectAll("*").remove();
                  svg.select('.legend_group').selectAll("*").remove();
                  svg.select('.stats_group').selectAll("*").remove();

                  var host_name = svg.select('.hostName');
                  var label_group = svg.select('.label_group');
                  var stats_group = svg.select('.stats_group');

                  host_name.text(data[0].hostName);

                  host_name = svg.selectAll("text.hostName").data(data);

                  host_name.attr("text-anchor", function(d) { return "start"; })
                      .text(function(d) { return d.hostName; });
                  host_name.exit().remove();

                  var labels = label_group.selectAll("text.labels").data(data);
                  labels.enter()
                      .append("svg:text")
                      .attr("class", "labels")
                      .attr("dy", function(d, i) { return 19 * i; })
                      .attr("text-anchor", function(d) { return "start"; })
                      .text(function(d) { return d.label; });
                  labels.exit().remove();

                  var stats = stats_group.selectAll("text.stats").data(data);
                  stats.enter()
                      .append("svg:text")
                      .attr("class", "stats")
                      .attr("dy", function(d, i) { return 19 * i; })
                      .attr("text-anchor", function(d) { return "start"; })
                      .text(function(d) { return d.stats; });
                  stats.exit().remove();

                  var legend_group = svg.select('.legend_group');
                  var legend = legend_group.selectAll("rect").data(data);
                  legend.enter()
                      .append("svg:rect")
                      .attr("x", 2)
                      .attr("y", function(d, i) { return 19 * i; })
                      .attr("width", 13)
                      .attr("height", 13)
                      .attr("class", function(d, i) { return "rect " + d.classNames; });

                  legend.exit().remove();
                }

                function getRadiusRing(ir, i) { return ir - (i * 20); }

                function getArc(thickness, ir) {
                  var arc = d3.svg.arc()
                                .innerRadius(function(d) { return getRadiusRing(ir, d.index); })
                                .outerRadius(function(d) { return getRadiusRing(ir + thickness, d.index); })
                                .startAngle(function(d, i) { return d.startAngle; })
                                .endAngle(function(d, i) { return d.endAngle; });
                  return arc;
                }

                function getBackgroundArc(thickness, ir) {
                  var arc = d3.svg.arc()
                                .innerRadius(function(d) { return getRadiusRing(ir, d.index); })
                                .outerRadius(function(d) { return getRadiusRing(ir + thickness, d.index); })
                                .startAngle(0)
                                .endAngle(function() { return 2 * Math.PI; });
                  return arc;
                }

                scope.render = function(data) {
                  if (data === undefined || data === null) {
                    return;
                  }

                  svg.selectAll("*").remove();

                  var graph = $(element[0]);
                  var w = scope.graphWidth;
                  var h = scope.graphHeight;

                  var options = {
                    data: data,
                    width: w,
                    height: h
                  };

                  init(options);
                };
              };
              d3DashboardService.d3().then(draw);
            }
          };
        }
      ]);
}());
