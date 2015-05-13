// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package static

const containersJs = gchartsJs + `
function humanize(num, size, units) {
    var unit;
    for (unit = units.pop(); units.length && num >= size; unit = units.pop()) {
        num /= size;
    }
    return [num, unit];
}

// Following the IEC naming convention
function humanizeIEC(num) {
        var ret = humanize(num, 1024, ["TiB", "GiB", "MiB", "KiB", "Bytes"]);
	return ret[0].toFixed(2) + " " + ret[1];
}

// Following the Metric naming convention
function humanizeMetric(num) {
        var ret = humanize(num, 1000, ["TB", "GB", "MB", "KB", "Bytes"]);
	return ret[0].toFixed(2) + " " + ret[1];
}

// Draw a table.
function drawTable(seriesTitles, titleTypes, data, elementId) {
	var dataTable = new google.visualization.DataTable();
	for (var i = 0; i < seriesTitles.length; i++) {
		dataTable.addColumn(titleTypes[i], seriesTitles[i]);
	}
	dataTable.addRows(data);
	if (!(elementId in window.charts)) {
		window.charts[elementId] = new google.visualization.Table(document.getElementById(elementId));
	}

	var opts = {
		alternatingRowStyle: true,
		page: 'enable',
		pageSize: 25,
	};
	window.charts[elementId].draw(dataTable, opts);
}

// Draw a line chart.
function drawLineChart(seriesTitles, data, elementId, unit) {
	var min = Infinity;
	var max = -Infinity;
	for (var i = 0; i < data.length; i++) {
		// Convert the first column to a Date.
		if (data[i] != null) {
			data[i][0] = new Date(data[i][0]);
		}

		// Find min, max.
		for (var j = 1; j < data[i].length; j++) {
			var val = data[i][j];
			if (val < min) {
				min = val;
			}
			if (val > max) {
				max = val;
			}
		}
	}

	// We don't want to show any values less than 0 so cap the min value at that.
	// At the same time, show 10% of the graph below the min value if we can.
	var minWindow = min - (max - min) / 10;
	if (minWindow < 0) {
		minWindow = 0;
	}

	// Add the definition of each column and the necessary data.
	var dataTable = new google.visualization.DataTable();
	dataTable.addColumn('datetime', seriesTitles[0]);
	for (var i = 1; i < seriesTitles.length; i++) {
		dataTable.addColumn('number', seriesTitles[i]);
	}
	dataTable.addRows(data);

	// Create and draw the visualization.
	if (!(elementId in window.charts)) {
		window.charts[elementId] = new google.visualization.LineChart(document.getElementById(elementId));
	}

	// TODO(vmarmol): Look into changing the view window to get a smoother animation.
	var opts = {
		curveType: 'function',
		height: 300,
		legend:{position:"none"},
		focusTarget: "category",
		vAxis: {
			title: unit,
			viewWindow: {
				min: minWindow,
			},
		},
		legend: {
			position: 'bottom',
		},
	};
	// If the whole data series has the same value, try to center it in the chart.
	if ( min == max) {
		opts.vAxis.viewWindow.max = 1.1 * max
		opts.vAxis.viewWindow.min = 0.9 * max
	}

	window.charts[elementId].draw(dataTable, opts);
}

// Gets the length of the interval in nanoseconds.
function getInterval(current, previous) {
	var cur = new Date(current);
	var prev = new Date(previous);

	// ms -> ns.
	return (cur.getTime() - prev.getTime()) * 1000000;
}

// Checks if the specified stats include the specified resource.
function hasResource(stats, resource) {
	return stats.stats.length > 0 && stats.stats[0][resource];
}

// Draw a set of gauges. Data is comprised of an array of arrays with two elements:
// a string label and a numeric value for the gauge.
function drawGauges(elementId, gauges) {
	gauges.unshift(['Label', 'Value']);

	// Create and populate the data table.
	var data = google.visualization.arrayToDataTable(gauges);

	// Create and draw the visualization.
	var options = {
		height: 100,
		redFrom: 90, redTo: 100,
		yellowFrom:75, yellowTo: 90,
		minorTicks: 5,
		animation: {
			duration: 900,
			easing: 'linear'
		}
	};
	var chart = new google.visualization.Gauge(document.getElementById(elementId));
	chart.draw(data, options);
}

// Get the machine info.
function getMachineInfo(rootDir, callback) {
	$.getJSON(rootDir + "api/v1.0/machine", function(data) {
		callback(data);
	});
}

// Get ps info.
function getProcessInfo(rootDir, containerName, callback) {
	$.getJSON(rootDir + "api/v2.0/ps" + containerName, function(data) {
		callback(data);
	});
}

// Get the container stats for the specified container.
function getStats(rootDir, containerName, callback) {
	// Request 60s of container history and no samples.
	var request = JSON.stringify({
                // Update main.statsRequestedByUI while updating "num_stats" here.
		"num_stats": 60,
		"num_samples": 0
	});
	$.post(rootDir + "api/v1.0/containers" + containerName, request, function(data) {
		callback(data);
	}, "json");
}

// Draw the graph for CPU usage.
function drawCpuTotalUsage(elementId, machineInfo, stats) {
	if (stats.spec.has_cpu && !hasResource(stats, "cpu")) {
		return;
	}

	var titles = ["Time", "Total"];
	var data = [];
	for (var i = 1; i < stats.stats.length; i++) {
		var cur = stats.stats[i];
		var prev = stats.stats[i - 1];
		var intervalInNs = getInterval(cur.timestamp, prev.timestamp);

		var elements = [];
		elements.push(cur.timestamp);
		elements.push((cur.cpu.usage.total - prev.cpu.usage.total) / intervalInNs);
		data.push(elements);
	}
	drawLineChart(titles, data, elementId, "Cores");
}

// Draw the graph for CPU load.
function drawCpuLoad(elementId, machineInfo, stats) {

	var titles = ["Time", "Average"];
	var data = [];
	for (var i = 1; i < stats.stats.length; i++) {
		var cur = stats.stats[i];

		var elements = [];
		elements.push(cur.timestamp);
		elements.push(cur.cpu.load_average/1000);
		data.push(elements);
	}
	drawLineChart(titles, data, elementId, "Runnable threads");
}


// Draw the graph for per-core CPU usage.
function drawCpuPerCoreUsage(elementId, machineInfo, stats) {
	if (stats.spec.has_cpu && !hasResource(stats, "cpu")) {
		return;
	}

	// Add a title for each core.
	var titles = ["Time"];
	for (var i = 0; i < machineInfo.num_cores; i++) {
		titles.push("Core " + i);
	}
	var data = [];
	for (var i = 1; i < stats.stats.length; i++) {
		var cur = stats.stats[i];
		var prev = stats.stats[i - 1];
		var intervalInNs = getInterval(cur.timestamp, prev.timestamp);

		var elements = [];
		elements.push(cur.timestamp);
		for (var j = 0; j < machineInfo.num_cores; j++) {
			elements.push((cur.cpu.usage.per_cpu_usage[j] - prev.cpu.usage.per_cpu_usage[j]) / intervalInNs);
		}
		data.push(elements);
	}
	drawLineChart(titles, data, elementId, "Cores");
}

// Draw the graph for CPU usage breakdown.
function drawCpuUsageBreakdown(elementId, machineInfo, containerInfo) {
	if (containerInfo.spec.has_cpu && !hasResource(containerInfo, "cpu")) {
		return;
	}

	var titles = ["Time", "User", "Kernel"];
	var data = [];
	for (var i = 1; i < containerInfo.stats.length; i++) {
		var cur = containerInfo.stats[i];
		var prev = containerInfo.stats[i - 1];
		var intervalInNs = getInterval(cur.timestamp, prev.timestamp);

		var elements = [];
		elements.push(cur.timestamp);
		elements.push((cur.cpu.usage.user - prev.cpu.usage.user) / intervalInNs);
		elements.push((cur.cpu.usage.system - prev.cpu.usage.system) / intervalInNs);
		data.push(elements);
	}
	drawLineChart(titles, data, elementId, "Cores");
}

// Draw the gauges for overall resource usage.
function drawOverallUsage(elementId, machineInfo, containerInfo) {
	var cur = containerInfo.stats[containerInfo.stats.length - 1];
	var gauges = [];

	var cpuUsage = 0;
	if (containerInfo.spec.has_cpu && containerInfo.stats.length >= 2) {
		var prev = containerInfo.stats[containerInfo.stats.length - 2];
		var rawUsage = cur.cpu.usage.total - prev.cpu.usage.total;
		var intervalInNs = getInterval(cur.timestamp, prev.timestamp);

		// Convert to millicores and take the percentage
		cpuUsage = Math.round(((rawUsage / intervalInNs) / machineInfo.num_cores) * 100);
		if (cpuUsage > 100) {
			cpuUsage = 100;
		}
		gauges.push(['CPU', cpuUsage]);
	}

	var memoryUsage = 0;
	if (containerInfo.spec.has_memory) {
		// Saturate to the machine size.
		var limit = containerInfo.spec.memory.limit;
		if (limit > machineInfo.memory_capacity) {
			limit = machineInfo.memory_capacity;
		}

		memoryUsage = Math.round((cur.memory.usage / limit) * 100);
		gauges.push(['Memory', memoryUsage]);
	}

	var numGauges = gauges.length;
	if (cur.filesystem) {
		for (var i = 0; i < cur.filesystem.length; i++) {
			var data = cur.filesystem[i];
			var totalUsage = Math.floor((data.usage * 100.0) / data.capacity);
			var els = window.cadvisor.fsUsage.elements[data.device];

			// Update the gauges in the right order.
			gauges[numGauges + els.index] = ['FS #' + (els.index + 1), totalUsage];
		}

		// Limit the number of filesystem gauges displayed to 5.
		// 'Filesystem details' section still shows information for all filesystems.
		var max_gauges = numGauges + 5;
		if (gauges.length > max_gauges) {
			gauges = gauges.slice(0, max_gauges);
		}
	}

	drawGauges(elementId, gauges);
}

var oneMegabyte = 1024 * 1024;
var oneGigabyte = 1024 * oneMegabyte;

function drawMemoryUsage(elementId, machineInfo, containerInfo) {
	if (containerInfo.spec.has_memory && !hasResource(containerInfo, "memory")) {
		return;
	}

	var titles = ["Time", "Total", "Hot"];
	var data = [];
	for (var i = 0; i < containerInfo.stats.length; i++) {
		var cur = containerInfo.stats[i];

		var elements = [];
		elements.push(cur.timestamp);
		elements.push(cur.memory.usage / oneMegabyte);
		elements.push(cur.memory.working_set / oneMegabyte);
		data.push(elements);
	}

	// Get the memory limit, saturate to the machine size.
	var memory_limit = machineInfo.memory_capacity;
	if (containerInfo.spec.memory.limit && (containerInfo.spec.memory.limit < memory_limit)) {
		memory_limit = containerInfo.spec.memory.limit;
	}

	// Updating the progress bar.
	var cur = containerInfo.stats[containerInfo.stats.length-1];
        var hotMemory = Math.floor((cur.memory.working_set * 100.0) / memory_limit);
        var totalMemory = Math.floor((cur.memory.usage * 100.0) / memory_limit);
	var coldMemory = totalMemory - hotMemory;
	$("#progress-hot-memory").width(hotMemory + "%");
        $("#progress-cold-memory").width(coldMemory + "%");
	$("#memory-text").text(humanizeIEC(cur.memory.usage) + " / " + humanizeIEC(memory_limit) +  " ("+ totalMemory +"%)");

	drawLineChart(titles, data, elementId, "Megabytes");
}

// Draw the graph for network tx/rx bytes.
function drawNetworkBytes(elementId, machineInfo, stats) {
	if (stats.spec.has_network && !hasResource(stats, "network")) {
		return;
	}

	var titles = ["Time", "Tx bytes", "Rx bytes"];
	var data = [];
	for (var i = 1; i < stats.stats.length; i++) {
		var cur = stats.stats[i];
		var prev = stats.stats[i - 1];
		var intervalInSec = getInterval(cur.timestamp, prev.timestamp) / 1000000000;

		var elements = [];
		elements.push(cur.timestamp);
		elements.push((cur.network.tx_bytes - prev.network.tx_bytes) / intervalInSec);
		elements.push((cur.network.rx_bytes - prev.network.rx_bytes) / intervalInSec);
		data.push(elements);
	}
	drawLineChart(titles, data, elementId, "Bytes per second");
}

// Draw the graph for network errors
function drawNetworkErrors(elementId, machineInfo, stats) {
	if (stats.spec.has_network && !hasResource(stats, "network")) {
		return;
	}

	var titles = ["Time", "Tx", "Rx"];
	var data = [];
	for (var i = 1; i < stats.stats.length; i++) {
		var cur = stats.stats[i];
		var prev = stats.stats[i - 1];
		var intervalInSec = getInterval(cur.timestamp, prev.timestamp) / 1000000000;

		var elements = [];
		elements.push(cur.timestamp);
		elements.push((cur.network.tx_errors - prev.network.tx_errors) / intervalInSec);
		elements.push((cur.network.rx_errors - prev.network.rx_errors) / intervalInSec);
		data.push(elements);
	}
	drawLineChart(titles, data, elementId, "Errors per second");
}

// Update the filesystem usage values.
function drawFileSystemUsage(machineInfo, stats) {
	var cur = stats.stats[stats.stats.length - 1];
	if (!cur.filesystem) {
		return;
	}

	var el = $("<div>");
	for (var i = 0; i < cur.filesystem.length; i++) {
		var data = cur.filesystem[i];
		var totalUsage = Math.floor((data.usage * 100.0) / data.capacity);

		// Update DOM elements.
		var els = window.cadvisor.fsUsage.elements[data.device];
		els.progressElement.width(totalUsage + "%");
		els.textElement.text(humanizeMetric(data.usage) + " / " + humanizeMetric(data.capacity)+ " (" + totalUsage + "%)");
	}
}

function drawProcesses(processInfo) {
	var titles = ["User", "PID", "PPID", "Start Time", "CPU %", "RSS", "Virtual Size", "Status", "Running Time", "Command"];
	var titleTypes = ['string', 'number', 'number', 'string', 'string', 'string', 'string', 'string', 'string', 'string'];
	var data = []
	for (var i = 1; i < processInfo.length; i++) {
		var elements = [];
		elements.push(processInfo[i].user);
		elements.push(processInfo[i].pid);
		elements.push(processInfo[i].parent_pid);
		elements.push(processInfo[i].start_time);
		elements.push(processInfo[i].percent_cpu);
		elements.push(processInfo[i].rss);
		elements.push(processInfo[i].virtual_size);
		elements.push(processInfo[i].status);
		elements.push(processInfo[i].running_time);
		elements.push(processInfo[i].cmd);
		data.push(elements);
	}
	drawTable(titles, titleTypes, data, "processes-top");
}

// Draw the filesystem usage nodes.
function startFileSystemUsage(elementId, machineInfo, stats) {
	window.cadvisor.fsUsage = {};

	// A map of device name to DOM elements.
	window.cadvisor.fsUsage.elements = {};

	var cur = stats.stats[stats.stats.length - 1];
	var el = $("<div>");
	if (!cur.filesystem) {
		return;
	}
	for (var i = 0; i < cur.filesystem.length; i++) {
		var data = cur.filesystem[i];
		el.append($("<div>")
			.addClass("row col-sm-12")
			.append($("<h4>")
				.text("FS #" + (i + 1) + ": " + data.device)));

		var progressElement = $("<div>").addClass("progress-bar progress-bar-danger");
		el.append($("<div>")
			.addClass("col-sm-9")
			.append($("<div>")
				.addClass("progress")
				.append(progressElement)));

		var textElement = $("<div>").addClass("col-sm-3");
		el.append(textElement);

		window.cadvisor.fsUsage.elements[data.device] = {
			'progressElement': progressElement,
			'textElement': textElement,
			'index': i,
		};
	}
	$("#" + elementId).empty().append(el);

	drawFileSystemUsage(machineInfo, stats);
}

// Expects an array of closures to call. After each execution the JS runtime is given control back before continuing.
// This function returns asynchronously
function stepExecute(steps) {
	// No steps, stop.
	if (steps.length == 0) {
		return;
	}

	// Get a step and execute it.
	var step = steps.shift();
	step();

	// Schedule the next step.
	setTimeout(function() {
		stepExecute(steps);
	}, 0);
}

// Draw all the charts on the page.
function drawCharts(machineInfo, containerInfo) {
	var steps = [];

	if (containerInfo.spec.has_cpu || containerInfo.spec.has_memory) {
		steps.push(function() {
			drawOverallUsage("usage-gauge", machineInfo, containerInfo)
		});
	}

	// CPU.
	if (containerInfo.spec.has_cpu) {
		steps.push(function() {
			drawCpuTotalUsage("cpu-total-usage-chart", machineInfo, containerInfo);
		});
		// TODO(rjnagal): Re-enable CPU Load after understanding resource usage. 
		// steps.push(function() {
		// 	drawCpuLoad("cpu-load-chart", machineInfo, containerInfo);
		// });
		steps.push(function() {
			drawCpuPerCoreUsage("cpu-per-core-usage-chart", machineInfo, containerInfo);
		});
		steps.push(function() {
			drawCpuUsageBreakdown("cpu-usage-breakdown-chart", machineInfo, containerInfo);
		});
	}

	// Memory.
	if (containerInfo.spec.has_memory) {
		steps.push(function() {
			drawMemoryUsage("memory-usage-chart", machineInfo, containerInfo);
		});
	}

	// Network.
	if (containerInfo.spec.has_network) {
		steps.push(function() {
			drawNetworkBytes("network-bytes-chart", machineInfo, containerInfo);
		});
		steps.push(function() {
			drawNetworkErrors("network-errors-chart", machineInfo, containerInfo);
		});
	}

	// Filesystem.
	if (containerInfo.spec.has_filesystem) {
		steps.push(function() {
                        drawFileSystemUsage(machineInfo, containerInfo);
                });
	}

	stepExecute(steps);
}

// Executed when the page finishes loading.
function startPage(containerName, hasCpu, hasMemory, rootDir) {
	// Don't fetch data if we don't have any resource.
	if (!hasCpu && !hasMemory) {
		return;
	}

	window.charts = {};
	window.cadvisor = {};
	window.cadvisor.firstRun = true;

	// Draw process information at start and refresh every 60s.
	getProcessInfo(rootDir, containerName, function(processInfo) {
		drawProcesses(processInfo)
	});
	setInterval(function() {
		getProcessInfo(rootDir, containerName, function(processInfo) {
			drawProcesses(processInfo)
		});
	}, 60000);

	// Get machine info, then get the stats every 1s.
	getMachineInfo(rootDir, function(machineInfo) {
		setInterval(function() {
			getStats(rootDir, containerName, function(containerInfo){
				if (window.cadvisor.firstRun && containerInfo.spec.has_filesystem) {
					window.cadvisor.firstRun = false;
					startFileSystemUsage("filesystem-usage", machineInfo, containerInfo);
				}

				drawCharts(machineInfo, containerInfo);
			});
		}, 1000);
	});
}
`
