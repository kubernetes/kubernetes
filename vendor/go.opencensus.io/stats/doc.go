// Copyright 2017, OpenCensus Authors
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
//

/*
Package stats contains support for OpenCensus stats recording.

OpenCensus allows users to create typed measures, record measurements,
aggregate the collected data, and export the aggregated data.

Measures

A measure represents a type of metric to be tracked and recorded.
For example, latency, request Mb/s, and response Mb/s are measures
to collect from a server.

Each measure needs to be registered before being used. Measure
constructors such as Int64 and Float64 automatically
register the measure by the given name. Each registered measure needs
to be unique by name. Measures also have a description and a unit.

Libraries can define and export measures for their end users to
create views and collect instrumentation data.

Recording measurements

Measurement is a data point to be collected for a measure. For example,
for a latency (ms) measure, 100 is a measurement that represents a 100ms
latency event. Users collect data points on the existing measures with
the current context. Tags from the current context are recorded with the
measurements if they are any.

Recorded measurements are dropped immediately if user is not aggregating
them via views. Users don't necessarily need to conditionally enable/disable
recording to reduce cost. Recording of measurements is cheap.

Libraries can always record measurements, and end-users can later decide
on which measurements they want to collect by registering views. This allows
libraries to turn on the instrumentation by default.
*/
package stats

// TODO(acetechnologist): Add a link to the language independent OpenCensus
// spec when it is available.
