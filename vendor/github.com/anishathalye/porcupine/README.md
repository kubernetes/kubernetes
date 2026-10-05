# Porcupine [![Build Status](https://github.com/anishathalye/porcupine/actions/workflows/ci.yml/badge.svg)](https://github.com/anishathalye/porcupine/actions/workflows/ci.yml) [![Go Reference](https://pkg.go.dev/badge/github.com/anishathalye/porcupine)](https://pkg.go.dev/github.com/anishathalye/porcupine)

Porcupine is a fast linearizability checker [used](#users) in both academia and
industry for testing the correctness of distributed systems. It takes a
sequential specification as executable Go code, along with a concurrent
history, and it determines whether the history is linearizable with respect to
the sequential specification. Porcupine also implements a visualizer for
histories and linearization points.

<p align="center">
<a href="https://anishathalye.github.io/porcupine/ex-2.html">
<img src="https://raw.githubusercontent.com/anishathalye/assets/master/porcupine/top-demo.png" width="500" alt="Linearizability visualization demo">
</a>
<br>
(click for interactive version)
</p>

Porcupine implements the algorithm described in [Faster linearizability
checking via P-compositionality][faster-linearizability-checking], an
optimization of the algorithm described in [Testing for
Linearizability][linearizability-testing].

Porcupine is faster and can handle more histories than [Knossos][knossos]'s
linearizability checker. Testing on the data in `test_data/jepsen/`, Porcupine
is generally **1,000x**-**10,000x** faster and has a much smaller memory
footprint. On histories where it can take advantage of P-compositionality,
Porcupine can be millions of times faster.

## Usage

Porcupine takes an executable model of a system along with a history, and it
runs a decision procedure to determine if the history is linearizable with
respect to the model. Porcupine supports specifying history in two ways, either
as a list of operations with given call and return times, or as a list of
call/return events in time order. Porcupine can also visualize histories, along
with partial linearizations, which may aid in debugging.

See the [documentation] for how to write a [model][porcupine-doc-model] and
[specify histories][porcupine-doc-history]. You can also check out some
[example implementations][porcupine-tests] of models from the tests.

Once you've written a model and have a history, you can use the
[`CheckOperations`][CheckOperations] and [`CheckEvents`][CheckEvents] functions
to determine if your history is linearizable. If you want to visualize a
history, along with partial linearizations, you can use the
[`Visualize`][Visualize] function.

[documentation]: https://pkg.go.dev/github.com/anishathalye/porcupine
[porcupine-doc-model]: https://pkg.go.dev/github.com/anishathalye/porcupine#Model
[porcupine-doc-history]: https://pkg.go.dev/github.com/anishathalye/porcupine#Operation
[CheckOperations]: https://pkg.go.dev/github.com/anishathalye/porcupine#CheckOperations
[CheckEvents]: https://pkg.go.dev/github.com/anishathalye/porcupine#CheckEvents
[Visualize]: https://pkg.go.dev/github.com/anishathalye/porcupine#Visualize
[porcupine-tests]: https://github.com/anishathalye/porcupine/blob/master/porcupine_test.go

### Testing linearizability

Suppose we're testing linearizability for operations on a read/write register
that's initialized to `0`. We write a sequential specification for the register
like this:

```go
type registerInput struct {
    op    bool // false = put, true = get
    value int
}

// a sequential specification of a register
registerModel := porcupine.Model{
    Init: func() interface{} {
        return 0
    },
    // step function: takes a state, input, and output, and returns whether it
    // was a legal operation, along with a new state
    Step: func(state, input, output interface{}) (bool, interface{}) {
        regInput := input.(registerInput)
        if regInput.op == false {
            return true, regInput.value // always ok to execute a put
        } else {
            readCorrectValue := output == state
            return readCorrectValue, state // state is unchanged
        }
    },
    // computes a hash for a state (optional field, improves performance)
    Hash: func(state interface{}) uint64 {
        return uint64(state.(int))
    },
}
```

Suppose we have the following concurrent history from a set of 3 clients. In a
row, the first `|` is when the operation was invoked, and the second `|` is
when the operation returned.

```
C0:  |-------- put('100') --------|
C1:     |--- get() -> '100' ---|
C2:        |- get() -> '0' -|
```

We encode this history as follows:

```go
events := []porcupine.Event{
    // C0: put('100')
    {Kind: porcupine.CallEvent, Value: registerInput{false, 100}, Id: 0, ClientId: 0},
    // C1: get()
    {Kind: porcupine.CallEvent, Value: registerInput{true, 0}, Id: 1, ClientId: 1},
    // C2: get()
    {Kind: porcupine.CallEvent, Value: registerInput{true, 0}, Id: 2, ClientId: 2},
    // C2: Completed get() -> '0'
    {Kind: porcupine.ReturnEvent, Value: 0, Id: 2, ClientId: 2},
    // C1: Completed get() -> '100'
    {Kind: porcupine.ReturnEvent, Value: 100, Id: 1, ClientId: 1},
    // C0: Completed put('100')
    {Kind: porcupine.ReturnEvent, Value: 0, Id: 0, ClientId: 0},
}
```

We can have Porcupine check the linearizability of the history as follows:

```go
ok := porcupine.CheckEvents(registerModel, events)
// returns true
```

Porcupine can visualize the linearization points as well:

<p align="center">
<a href="https://anishathalye.github.io/porcupine/ex-1.html">
<img src="https://raw.githubusercontent.com/anishathalye/assets/master/porcupine/ex-1.png" width="500" alt="Example 1">
</a>
</p>

Now, suppose we have another history:

```
C0:  |---------------- put('200') ----------------|
C1:    |- get() -> '200' -|
C2:                           |- get() -> '0' -|
```

We can check the history with Porcupine and see that it's not linearizable:

```go
events := []porcupine.Event{
    // C0: put('200')
    {Kind: porcupine.CallEvent, Value: registerInput{false, 200}, Id: 0, ClientId: 0},
    // C1: get()
    {Kind: porcupine.CallEvent, Value: registerInput{true, 0}, Id: 1, ClientId: 1},
    // C1: Completed get() -> '200'
    {Kind: porcupine.ReturnEvent, Value: 200, Id: 1, ClientId: 1},
    // C2: get()
    {Kind: porcupine.CallEvent, Value: registerInput{true, 0}, Id: 2, ClientId: 2},
    // C2: Completed get() -> '0'
    {Kind: porcupine.ReturnEvent, Value: 0, Id: 2, ClientId: 2},
    // C0: Completed put('200')
    {Kind: porcupine.ReturnEvent, Value: 0, Id: 0, ClientId: 0},
}

ok := porcupine.CheckEvents(registerModel, events)
// returns false
```

<p align="center">
<a href="https://anishathalye.github.io/porcupine/ex-2.html">
<img src="https://raw.githubusercontent.com/anishathalye/assets/master/porcupine/ex-2.png" width="500" alt="Example 2">
</a>
</p>

See [`porcupine_test.go`](porcupine_test.go) for more examples on how to write
models and histories.

### Visualizing histories

Porcupine provides functionality to visualize histories, along with the
linearization (or partial linearizations and illegal linearization points, in
the case of a non-linearizable history). The result is an HTML page that draws
an interactive visualization using JavaScript. The output looks like this:

<p align="center">
<a href="https://anishathalye.github.io/porcupine/demo-annotations.html">
<img src="https://raw.githubusercontent.com/anishathalye/assets/master/porcupine/demo-annotations.png" width="1021" alt="Visualization demo">
</a>
</p>

You can see the full interactive version
[here](https://anishathalye.github.io/porcupine/demo-annotations.html).

The visualization is by partition: all partitions are essentially independent,
so with the key-value store example above, operations related to each unique
key are in a separate partition.

Statically, the visualization shows all history elements, along with
linearization points for each partition. If a partition has no full
linearization, the visualization shows the longest partial linearization. It
also shows, for each history element, the longest partial linearization
containing that event, even if it's not the longest overall partial
linearization; these are greyed out by default. It also shows illegal
linearization points, history elements that were checked to see if they could
occur next but which were illegal to linearize at that point according to the
model.

When a history element is hovered over, the visualization highlights the most
relevant partial linearization. When it exists, the longest partial
linearization containing the event is shown. Otherwise, when it exists, the
visualization shows the longest partial linearization that ends with an illegal
LP that is the event being hovered over. Hovering over an event also shows a
tooltip showing extra information, such as the previous and current states of
the state machine, as well as the time the operation was invoked and when it
returned. This information is derived from the currently selected linearization.

Clicking on a history element selects it, which highlights the event with a
bold border. This has the effect of making the selection of a partial
linearization "sticky", so it's possible to move around the history without
de-selecting it. Clicking on another history element will select that one
instead, and clicking on the background will deselect.

All that's needed to visualize histories is the
[`CheckOperationsVerbose`][CheckOperationsVerbose] /
[`CheckEventsVerbose`][CheckEventsVerbose] functions, which return extra
information that's used by the visualization, and the [`Visualize`][Visualize]
function, which produces the visualization. For the visualization to be good,
it's useful to fill out the `DescribeOperation` and `DescribeState` fields of
the model. See [`visualization_test.go`](visualization_test.go) for an
end-to-end example of how to visualize a history using Porcupine.

You can also add custom annotations to visualizations, as shown in the example
above. This can be helpful for attaching debugging information, for example,
from servers or the test framework. You can do this using the
[`AddAnnotations`][AddAnnotations] method.

[CheckOperationsVerbose]: https://pkg.go.dev/github.com/anishathalye/porcupine#CheckOperationsVerbose
[CheckEventsVerbose]: https://pkg.go.dev/github.com/anishathalye/porcupine#CheckEventsVerbose
[AddAnnotations]: https://pkg.go.dev/github.com/anishathalye/porcupine#LinearizationInfo.AddAnnotations

## Notes

- If Porcupine runs really slowly on your model/history, it may be inevitable,
  due to state space explosion. See [this
  issue](https://github.com/anishathalye/porcupine/issues/6) for a discussion
  of this challenge in the context of a particular model and history.
- When recording timestamps for operations, especially on ARM and other
  weakly-ordered architectures, you may need to use memory barriers or atomic
  operations to ensure accurate measurements and avoid spurious linearizability
  violations. See [this
  issue](https://github.com/anishathalye/porcupine/issues/40) for details.

## Users

Porcupine is used in both academia and industry. It can be helpful to look at
existing uses of Porcupine to learn how you can design a robust testing
framework with linearizability checking at its core.

- [etcd](https://github.com/etcd-io/etcd). etcd uses Porcupine's linearizability checker and visualizer in its [robustness tests](https://github.com/etcd-io/etcd/tree/main/tests/robustness) [[KubeCon 2023eu talk](https://www.youtube.com/watch?v=IIMs0EjQZHg)].
- [PingCAP TiPocket](https://github.com/pingcap/tipocket). PingCAP uses Porcupine to test the [TiDB](https://www.pingcap.com/tidb/) distributed database.
- [Amazon MemoryDB (SIGMOD'24)](https://www.amazon.science/publications/amazon-memorydb-a-fast-and-durable-memory-first-cloud-database). Amazon uses Porcupine to test MemoryDB consistency.
- [S2 Stream Store](https://s2.dev/). S2 [uses](https://s2.dev/blog/linearizability) Porcupine's linearizability checker and visualizer in its [linearizability tests](https://github.com/s2-streamstore/s2-verification).
- [LEGOStore (VLDB'22)](https://www.vldb.org/pvldb/vol15/p2201-zare.pdf). Porcupine is used to test the linearizability of LEGOStore.
- [Resonate](https://github.com/resonatehq/durable-promise-test-harness). Resonate's test harness uses Porcupine as its linearizability checker.
- [MIT 6.5840 (Distributed Systems) key-value store](https://pdos.csail.mit.edu/6.824/labs/lab-kvraft.html). Porcupine was originally developed for use in MIT's distributed systems class to test the linearizability of the Raft-based distributed key-value store.
- [IIT Delhi COL733 (Cloud computing technology fundamentals) object store](https://github.com/codenet/col733-cloud/tree/main/labs/lab3). The testing harness uses Porcupine to test the linearizability of the strongly-consistent variant of [CRAQ](https://www.usenix.org/legacy/event/usenix09/tech/full_papers/terrace/terrace.pdf).

Does your system use Porcupine? Send a PR to add it to this list!

## Citation

```bibtex
@misc{athalye2017porcupine,
  author = {Anish Athalye},
  title = {Porcupine: A fast linearizability checker in {Go}},
  year = {2017},
  howpublished = {\url{https://github.com/anishathalye/porcupine}}
}
```

## License

Copyright (c) Anish Athalye. Released under the MIT License. See
[LICENSE.md][license] for details.

[faster-linearizability-checking]: https://arxiv.org/pdf/1504.00204.pdf
[linearizability-testing]: http://www.cs.ox.ac.uk/people/gavin.lowe/LinearizabiltyTesting/paper.pdf
[knossos]: https://github.com/jepsen-io/knossos
[license]: LICENSE.md
