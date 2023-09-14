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

# Measures

A measure represents a type of data point to be tracked and recorded.
For example, latency, request Mb/s, and response Mb/s are measures
to collect from a server.

Measure constructors such as Int64 and Float64 automatically
register the measure by the given name. Each registered measure needs
to be unique by name. Measures also have a description and a unit.

Libraries can define and export measures. Application authors can then
create views and collect and break down measures by the tags they are
interested in.

# Recording measurements

Measurement is a data point to be collected for a measure. For example,
for a latency (ms) measure, 100 is a measurement that represents a 100ms
latency event. Measurements are created from measures with
the current context. Tags from the current context are recorded with the
measurements if they are any.

Recorded measurements are dropped immediately if no views are registered for them.
There is usually no need to conditionally enable and disable
recording to reduce cost. Recording of measurements is cheap.

Libraries can always record measurements, and applications can later decide
on which measurements they want to collect by registering views. This allows
libraries to turn on the instrumentation by default.

# Exemplars

For a given recorded measurement, the associated exemplar is a diagnostic map
that gives more information about the measurement.

When aggregated using a Distribution aggregation, an exemplar is kept for each
bucket in the Distribution. This allows you to easily find an example of a
measurement that fell into each bucket.

For example, if you also use the OpenCensus trace package and you
record a measurement with a context that contains a sampled trace span,
then the trace span will be added to the exemplar associated with the measurement.

When exported to a supporting back end, you should be able to easily navigate
to example traces that fell into each bucket in the Distribution.
*/
package stats // import "go.opencensus.io/stats"
