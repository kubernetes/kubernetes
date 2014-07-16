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

const containersJs = `
google.load("visualization", "1", {packages: ["corechart", "gauge"]});

// Draw a line chart.
function drawLineChart(seriesTitles, data, elementId, unit) {
	// Convert the first column to a Date.
	for (var i = 0; i < data.length; i++) {
		if (data[i] != null) {
			data[i][0] = new Date(data[i][0]);
		}
	}

	// Add the definition of each column and the necessary data.
	var dataTable = new google.visualization.DataTable();
	dataTable.addColumn('datetime', seriesTitles[0]);
	for (var i = 1; i < seriesTitles.length; i++) {
		dataTable.addColumn('number', seriesTitles[i]);
	}
	dataTable.addRows(data);

	// Create and draw the visualization.
	var ac = null;
	var opts = null;
	// TODO(vmarmol): Remove this hack, it is to support the old charts and the new charts during the transition.
	if (window.charts) {
		if (!(elementId in window.charts)) {
			ac = new google.visualization.LineChart(document.getElementById(elementId));
			window.charts[elementId] = ac;
		}
		ac = window.charts[elementId];
		opts = window.chartOptions;
	} else {
		ac = new google.visualization.LineChart(document.getElementById(elementId));
		opts = {};
	}
	opts.vAxis = {title: unit};
	opts.legend = {position: 'bottom'};
	ac.draw(dataTable, window.chartOptions);
}

// Draw a gauge.
function drawGauge(elementId, cpuUsage, memoryUsage) {
	var gauges = [['Label', 'Value']];
	if (cpuUsage >= 0) {
		gauges.push(['CPU', cpuUsage]);
	}
	if (memoryUsage >= 0) {
		gauges.push(['Memory', memoryUsage]);
	}
	// Create and populate the data table.
	var data = google.visualization.arrayToDataTable(gauges);
	
	// Create and draw the visualization.
	var options = {
		width: 400, height: 120,
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
function getMachineInfo(callback) {
	$.getJSON("/api/v1.0/machine", function(data) {
		callback(data);
	});
}

// Get the container stats for the specified container.
function getStats(containerName, callback) {
	$.getJSON("/api/v1.0/containers" + containerName, function(data) {
		callback(data);
	});
}

// Draw the graph for CPU usage.
function drawCpuTotalUsage(elementId, machineInfo, stats) {
	var titles = ["Time", "Total"];
	var data = [];
	for (var i = 1; i < stats.stats.length; i++) {
		var cur = stats.stats[i];
		var prev = stats.stats[i - 1];

		// TODO(vmarmol): This assumes we sample every second, use the timestamps.
		var elements = [];
		elements.push(cur.timestamp);
		elements.push((cur.cpu.usage.total - prev.cpu.usage.total) / 1000000000);
		data.push(elements);
	}
	drawLineChart(titles, data, elementId, "Cores");
}

// Draw the graph for per-core CPU usage.
function drawCpuPerCoreUsage(elementId, machineInfo, stats) {
	// Add a title for each core.
	var titles = ["Time"];
	for (var i = 0; i < machineInfo.num_cores; i++) {
		titles.push("Core " + i);
	}
	var data = [];
	for (var i = 1; i < stats.stats.length; i++) {
		var cur = stats.stats[i];
		var prev = stats.stats[i - 1];

		var elements = [];
		elements.push(cur.timestamp);
		for (var j = 0; j < machineInfo.num_cores; j++) {
			// TODO(vmarmol): This assumes we sample every second, use the timestamps.
			elements.push((cur.cpu.usage.per_cpu_usage[j] - prev.cpu.usage.per_cpu_usage[j]) / 1000000000);
		}
		data.push(elements);
	}
	drawLineChart(titles, data, elementId, "Cores");
}

// Draw the graph for CPU usage breakdown.
function drawCpuUsageBreakdown(elementId, containerInfo) {
	var titles = ["Time", "User", "Kernel"];
	var data = [];
	for (var i = 1; i < containerInfo.stats.length; i++) {
		var cur = containerInfo.stats[i];
		var prev = containerInfo.stats[i - 1];

		// TODO(vmarmol): This assumes we sample every second, use the timestamps.
		var elements = [];
		elements.push(cur.timestamp);
		elements.push((cur.cpu.usage.user - prev.cpu.usage.user) / 1000000000);
		elements.push((cur.cpu.usage.system - prev.cpu.usage.system) / 1000000000);
		data.push(elements);
	}
	drawLineChart(titles, data, elementId, "Cores");
}

// Draw the gauges for overall resource usage.
function drawOverallUsage(elementId, machineInfo, containerInfo) {
	var cur = containerInfo.stats[containerInfo.stats.length - 1];

	var cpuUsage = 0;
	if (containerInfo.spec.cpu && containerInfo.stats.length >= 2) {
		var prev = containerInfo.stats[containerInfo.stats.length - 2];
		var rawUsage = cur.cpu.usage.total - prev.cpu.usage.total;

		// Convert to millicores and take the percentage
		cpuUsage = Math.round(((rawUsage / 1000000) / containerInfo.spec.cpu.limit) * 100);
		if (cpuUsage > 100) {
			cpuUsage = 100;
		}
	}

	var memoryUsage = 0;
	if (containerInfo.spec.memory) {
		// Saturate to the machine size.
		var limit = containerInfo.spec.memory.limit;
		if (limit > machineInfo.memory_capacity) {
			limit = machineInfo.memory_capacity;
		}

		memoryUsage = Math.round((cur.memory.usage / limit) * 100);
	}

	drawGauge(elementId, cpuUsage, memoryUsage);
}

var oneMegabyte = 1024 * 1024;

function drawMemoryUsage(elementId, containerInfo) {
	var titles = ["Time", "Total"];
	var data = [];
	for (var i = 0; i < containerInfo.stats.length; i++) {
		var cur = containerInfo.stats[i];

		// TODO(vmarmol): This assumes we sample every second, use the timestamps.
		var elements = [];
		elements.push(cur.timestamp);
		elements.push(cur.memory.usage / oneMegabyte);
		data.push(elements);
	}
	drawLineChart(titles, data, elementId, "Megabytes");
}

function drawMemoryPageFaults(elementId, containerInfo) {
	var titles = ["Time", "Faults", "Major Faults"];
	var data = [];
	for (var i = 1; i < containerInfo.stats.length; i++) {
		var cur = containerInfo.stats[i];
		var prev = containerInfo.stats[i - 1];

		// TODO(vmarmol): This assumes we sample every second, use the timestamps.
		var elements = [];
		elements.push(cur.timestamp);
		elements.push(cur.memory.hierarchical_data.pgfault - prev.memory.hierarchical_data.pgfault);
		// TODO(vmarmol): Fix to expose this data.
		//elements.push(cur.memory.hierarchical_data.pgmajfault - prev.memory.hierarchical_data.pgmajfault);
		elements.push(0);
		data.push(elements);
	}
	drawLineChart(titles, data, elementId, "Faults");
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

	steps.push(function() {
		drawOverallUsage("usage-gauge", machineInfo, containerInfo)
	});

	// CPU.
	steps.push(function() {
		drawCpuTotalUsage("cpu-total-usage-chart", machineInfo, containerInfo);
	});
	steps.push(function() {
		drawCpuPerCoreUsage("cpu-per-core-usage-chart", machineInfo, containerInfo);
	});
	steps.push(function() {
		drawCpuUsageBreakdown("cpu-usage-breakdown-chart", containerInfo);
	});

	// Memory.
	steps.push(function() {
		drawMemoryUsage("memory-usage-chart", containerInfo);
	});
	steps.push(function() {
		drawMemoryPageFaults("memory-page-faults-chart", containerInfo);
	});

	stepExecute(steps);
}

// Executed when the page finishes loading.
function startPage(containerName, hasCpu, hasMemory) {
	// Don't fetch data if we don't have any resource.
	if (!hasCpu && !hasMemory) {
		return;
	}

	// TODO(vmarmol): Look into changing the view window to get a smoother animation.
	window.chartOptions = {
		curveType: 'function',
		height: 300,
		legend:{position:"none"},
		focusTarget: "category",
	};
	window.charts = {};

	// Get machine info, then get the stats every 1s.
	getMachineInfo(function(machineInfo) {
		setInterval(function() {
			getStats(containerName, function(stats){
				drawCharts(machineInfo, stats);
			});
		}, 1000);
	});
}
`
