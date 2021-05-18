// Copyright The OpenTelemetry Authors
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

/*
Package metric implements the OpenTelemetry metric API.

This package is currently in a pre-GA phase. Backwards incompatible changes
may be introduced in subsequent minor version releases as we work to track the
evolving OpenTelemetry specification and user feedback.

The Accumulator type supports configurable metrics export behavior through a
collection of export interfaces that support various export strategies,
described below.

The OpenTelemetry metric API consists of methods for constructing synchronous
and asynchronous instruments.  There are two constructors per instrument for
the two kinds of number (int64, float64).

Synchronous instruments are managed by a sync.Map containing a *record
with the current state for each synchronous instrument.  A bound
instrument encapsulates a direct pointer to the record, allowing
bound metric events to bypass a sync.Map lookup.  A lock-free
algorithm is used to protect against races when adding and removing
items from the sync.Map.

Asynchronous instruments are managed by an internal
AsyncInstrumentState, which coordinates calling batch and single
instrument callbacks.

Internal Structure

Each observer also has its own kind of record stored in the SDK. This
record contains a set of recorders for every specific label set used in the
callback.

A sync.Map maintains the mapping of current instruments and label sets to
internal records.  To create a new bound instrument, the SDK consults the Map to
locate an existing record, otherwise it constructs a new record.  The SDK
maintains a count of the number of references to each record, ensuring
that records are not reclaimed from the Map while they are still active
from the user's perspective.

Metric collection is performed via a single-threaded call to Collect that
sweeps through all records in the SDK, checkpointing their state.  When a
record is discovered that has no references and has not been updated since
the prior collection pass, it is removed from the Map.

Both synchronous and asynchronous instruments have an associated
aggregator, which maintains the current state resulting from all metric
events since its last checkpoint.  Aggregators may be lock-free or they may
use locking, but they should expect to be called concurrently.  Aggregators
must be capable of merging with another aggregator of the same type.

Export Pipeline

While the SDK serves to maintain a current set of records and
coordinate collection, the behavior of a metrics export pipeline is
configured through the export types in
go.opentelemetry.io/otel/sdk/export/metric.  It is important to keep
in mind the context these interfaces are called from.  There are two
contexts, instrumentation context, where a user-level goroutine that
enters the SDK resulting in a new record, and collection context,
where a system-level thread performs a collection pass through the
SDK.

Descriptor is a struct that describes the metric instrument to the
export pipeline, containing the name, units, description, metric kind,
number kind (int64 or float64).  A Descriptor accompanies metric data
as it passes through the export pipeline.

The AggregatorSelector interface supports choosing the method of
aggregation to apply to a particular instrument, by delegating the
construction of an Aggregator to this interface.  Given the Descriptor,
the AggregatorFor method returns an implementation of Aggregator.  If this
interface returns nil, the metric will be disabled.  The aggregator should
be matched to the capabilities of the exporter.  Selecting the aggregator
for Adding instruments is relatively straightforward, but many options
are available for aggregating distributions from Grouping instruments.

Aggregator is an interface which implements a concrete strategy for
aggregating metric updates.  Several Aggregator implementations are
provided by the SDK.  Aggregators may be lock-free or use locking,
depending on their structure and semantics.  Aggregators implement an
Update method, called in instrumentation context, to receive a single
metric event.  Aggregators implement a Checkpoint method, called in
collection context, to save a checkpoint of the current state.
Aggregators implement a Merge method, also called in collection
context, that combines state from two aggregators into one.  Each SDK
record has an associated aggregator.

Processor is an interface which sits between the SDK and an exporter.
The Processor embeds an AggregatorSelector, used by the SDK to assign
new Aggregators.  The Processor supports a Process() API for submitting
checkpointed aggregators to the processor, and a CheckpointSet() API
for producing a complete checkpoint for the exporter.  Two default
Processor implementations are provided, the "defaultkeys" Processor groups
aggregate metrics by their recommended Descriptor.Keys(), the
"simple" Processor aggregates metrics at full dimensionality.

LabelEncoder is an optional optimization that allows an exporter to
provide the serialization logic for labels.  This allows avoiding
duplicate serialization of labels, once as a unique key in the SDK (or
Processor) and once in the exporter.

CheckpointSet is an interface between the Processor and the Exporter.
After completing a collection pass, the Processor.CheckpointSet() method
returns a CheckpointSet, which the Exporter uses to iterate over all
the updated metrics.

Record is a struct containing the state of an individual exported
metric.  This is the result of one collection interface for one
instrument and one label set.

Labels is a struct containing an ordered set of labels, the
corresponding unique encoding, and the encoder that produced it.

Exporter is the final stage of an export pipeline.  It is called with
a CheckpointSet capable of enumerating all the updated metrics.

Controller is not an export interface per se, but it orchestrates the
export pipeline.  For example, a "push" controller will establish a
periodic timer to regularly collect and export metrics.  A "pull"
controller will await a pull request before initiating metric
collection.  Either way, the job of the controller is to call the SDK
Collect() method, then read the checkpoint, then invoke the exporter.
Controllers are expected to implement the public metric.MeterProvider
API, meaning they can be installed as the global Meter provider.

*/
package metric // import "go.opentelemetry.io/otel/sdk/metric"
