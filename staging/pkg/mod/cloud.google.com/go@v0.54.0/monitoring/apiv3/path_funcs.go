// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package monitoring

// GroupProjectPath returns the path for the project resource.
//
// Deprecated: Use
//   fmt.Sprintf("projects/%s", project)
// instead.
func GroupProjectPath(project string) string {
	return "" +
		"projects/" +
		project +
		""
}

// GroupGroupPath returns the path for the group resource.
//
// Deprecated: Use
//   fmt.Sprintf("projects/%s/groups/%s", project, group)
// instead.
func GroupGroupPath(project, group string) string {
	return "" +
		"projects/" +
		project +
		"/groups/" +
		group +
		""
}

// MetricProjectPath returns the path for the project resource.
//
// Deprecated: Use
//   fmt.Sprintf("projects/%s", project)
// instead.
func MetricProjectPath(project string) string {
	return "" +
		"projects/" +
		project +
		""
}

// MetricMetricDescriptorPath returns the path for the metric descriptor resource.
//
// Deprecated: Use
//   fmt.Sprintf("projects/%s/metricDescriptors/%s", project, metricDescriptor)
// instead.
func MetricMetricDescriptorPath(project, metricDescriptor string) string {
	return "" +
		"projects/" +
		project +
		"/metricDescriptors/" +
		metricDescriptor +
		""
}

// MetricMonitoredResourceDescriptorPath returns the path for the monitored resource descriptor resource.
//
// Deprecated: Use
//   fmt.Sprintf("projects/%s/monitoredResourceDescriptors/%s", project, monitoredResourceDescriptor)
// instead.
func MetricMonitoredResourceDescriptorPath(project, monitoredResourceDescriptor string) string {
	return "" +
		"projects/" +
		project +
		"/monitoredResourceDescriptors/" +
		monitoredResourceDescriptor +
		""
}

// UptimeCheckProjectPath returns the path for the project resource.
//
// Deprecated: Use
//   fmt.Sprintf("projects/%s", project)
// instead.
func UptimeCheckProjectPath(project string) string {
	return "" +
		"projects/" +
		project +
		""
}

// UptimeCheckUptimeCheckConfigPath returns the path for the uptime check config resource.
//
// Deprecated: Use
//   fmt.Sprintf("projects/%s/uptimeCheckConfigs/%s", project, uptimeCheckConfig)
// instead.
func UptimeCheckUptimeCheckConfigPath(project, uptimeCheckConfig string) string {
	return "" +
		"projects/" +
		project +
		"/uptimeCheckConfigs/" +
		uptimeCheckConfig +
		""
}
