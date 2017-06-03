# smat – State Machine Assisted Testing

The concept is simple, describe valid uses of your library as states and actions.  States describe which actions are possible, and with what probability they should occur.  Actions mutate the context and transition to another state.

By doing this, two things are possible:

1.  Use [go-fuzz](https://github.com/dvyukov/go-fuzz) to find/test interesting sequences of operations on your library.

2.  Automate longevity testing of your application by performing long sequences of valid operations.

**NOTE**: both of these can also incorporate validation logic (not just failure detection by building validation into the state machine)

## Status

The API is still not stable.  This is brand new and we'll probably change things we don't like...

[![Build Status](https://travis-ci.org/mschoch/smat.svg?branch=master)](https://travis-ci.org/mschoch/smat)
[![Coverage Status](https://coveralls.io/repos/github/mschoch/smat/badge.svg?branch=master)](https://coveralls.io/github/mschoch/smat?branch=master)
[![GoDoc](https://godoc.org/github.com/mschoch/smat?status.svg)](https://godoc.org/github.com/mschoch/smat)
[![codebeat badge](https://codebeat.co/badges/c3ff6180-a241-4128-97f0-fa6bf6f48752)](https://codebeat.co/projects/github-com-mschoch-smat)
[![Go Report Card](https://goreportcard.com/badge/github.com/mschoch/smat)](https://goreportcard.com/report/github.com/mschoch/smat)

## License

Apache 2.0

## How do I use it?

### smat.Context

Choose a structure to keep track of any state.  You pass in an instance of this when you start, and it will be passed to every action when it executes.  The actions may mutate this context.

For example, consider a database library, once you open a database handle, you need to use it inside of the other actions.  So you might use a structure like:

```
type context struct {
  db *DB
}
```

### smat.State

A state represents a state that your application/library can be in, and the probabilities thats certain actions should be taken.

For example, consider a database library, in a state where the database is open, there many things you can do.  Let's consider just two right now, you can set a value, or you can delete a value.

```
func dbOpen(next byte) smat.ActionID {
	return smat.PercentExecute(next,
		smat.PercentAction{50, setValue},
		smat.PercentAction{50, deleteValue},
	)
}
```

This says that in the open state, there are two valid actions, 50% of the time you should set a value and 50% of the time you should delete a value.  **NOTE**: these percentages are just for characterizing the test workload.

### smat.Action

Actions are functions that do some work, optionally mutate the context, and indicate the next state to transition to.  Below we see an example action to set value in a database.

```
func setValueFunc(ctx smat.Context) (next smat.State, err error) {
  // type assert to our custom context type
	context := ctx.(*context)
  // perform the operation
  err = context.db.Set("k", "v")
  if err != nil {
    return nil, err
  }
  // return the new state
  return dbOpen, nil
}
```

### smat.ActionID and smat.ActionMap

Actions are just functions, and since we can't compare functions in Go, we need to introduce an external identifier for them.  This allows us to build a bi-directional mapping which we'll take advantage of later.

```
const (
  setup smat.ActionID = iota
  teardown
  setValue
  deleteValue
)

var actionMap = smat.ActionMap{
  setup:       setupFunc,
  teardown:    teardownFunc,
	setValue:    setValueFunc,
	deleteValue: deleteValueFunc,
}
```

### smat.ActionSeq

A common way that many users think about a library is as a sequence of actions to be performed.  Using the ActionID's that we've already seen we can build up sequences of operations.

```
  actionSeq := smat.ActionSeq{
		open,
		setValue,
		setValue,
		setValue,
	}
```

Notice that we build these actions using the constants we defined above, and because of this we can have a bi-directional mapping between a stream of bytes (driving the state machine) and a sequence of actions to be performed.

## Fuzzing

We've built a lot of pieces, lets wire it up to go-fuzz.

```
func Fuzz(data []byte) int {
	return smat.Fuzz(&context{}, setup, teardown, actionMap, data)
}
```

* The first argument is an instance of context structure.
* The second argument is the ActionID of our setup function.  The setup function does not consume any of the input stream and is used to initialize the context and determine the start state.
* The third argument is the teardown function.  This will be called unconditionally to clean up any resources associated with the test.
* The fourth argument is the actionMap which maps all ActionIDs to Actions.
* The fifth argument is the data passed in from the go-fuzz application.

### Generating Initial go-fuzz Corpus

Earlier we mentioned the bi-directional mapping between Actions and the byte stream driving the state machine.  We can now leverage this to build the inital go-fuzz corpus.

Using the `ActinSeq`s we learned about earlier we can build up a list of them as:

    var actionSeqs = []smat.ActionSeq{...}

Then, we can write them out to disk using:

```
for i, actionSeq := range actionSeqs {
  byteSequence, err := actionSeq.ByteEncoding(&context{}, setup, teardown, actionMap)
  if err != nil {
    // handle error
  }
  os.MkdirAll("workdir/corpus", 0700)
  ioutil.WriteFile(fmt.Sprintf("workdir/corpus/%d", i), byteSequence, 0600)
}
```

You can then either put this into a test case or a main application depending on your needs.

## Longevity Testing

Fuzzing is great, but most of your corpus is likely to be shorter meaningful sequences.  And go-fuzz works to find shortest sequences that cause problems, but sometimes you actually want to explore longer sequences that appear to go-fuzz as not triggering additional code coverage.

For these cases we have another helper you can use:

```
  Longevity(ctx, setup, teardown, actionMap, 0, closeChan)
```

The first four arguments are the same, the last two are:
* random seed used to ensure repeatable tests
* closeChan (chan struct{}) - close this channel if you want the function to stop and return ErrClosed, otherwise it will run forever

## Examples

See the examples directory for a working example that tests some BoltDB functionality.
