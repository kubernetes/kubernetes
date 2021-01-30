# A more minimal logging API for Go

Before you consider this package, please read [this blog post by the
inimitable Dave Cheney][warning-makes-no-sense].  I really appreciate what
he has to say, and it largely aligns with my own experiences.  Too many
choices of levels means inconsistent logs.

This package offers a purely abstract interface, based on these ideas but with
a few twists.  Code can depend on just this interface and have the actual
logging implementation be injected from callers.  Ideally only `main()` knows
what logging implementation is being used.

# Differences from Dave's ideas

The main differences are:

1) Dave basically proposes doing away with the notion of a logging API in favor
of `fmt.Printf()`.  I disagree, especially when you consider things like output
locations, timestamps, file and line decorations, and structured logging.  I
restrict the API to just 2 types of logs: info and error.

Info logs are things you want to tell the user which are not errors.  Error
logs are, well, errors.  If your code receives an `error` from a subordinate
function call and is logging that `error` *and not returning it*, use error
logs.

2) Verbosity-levels on info logs.  This gives developers a chance to indicate
arbitrary grades of importance for info logs, without assigning names with
semantic meaning such as "warning", "trace", and "debug".  Superficially this
may feel very similar, but the primary difference is the lack of semantics.
Because verbosity is a numerical value, it's safe to assume that an app running
with higher verbosity means more (and less important) logs will be generated.

This is a BETA grade API.

There are implementations for the following logging libraries:

- **github.com/google/glog**: [glogr](https://github.com/go-logr/glogr)
- **k8s.io/klog**: [klogr](https://git.k8s.io/klog/klogr)
- **go.uber.org/zap**: [zapr](https://github.com/go-logr/zapr)
- **log** (the Go standard library logger):
  [stdr](https://github.com/go-logr/stdr)
- **github.com/sirupsen/logrus**: [logrusr](https://github.com/bombsimon/logrusr)
- **github.com/wojas/genericr**: [genericr](https://github.com/wojas/genericr) (makes it easy to implement your own backend)
- **logfmt** (Heroku style [logging](https://www.brandur.org/logfmt)): [logfmtr](https://github.com/iand/logfmtr)

# FAQ

## Conceptual

## Why structured logging?

- **Structured logs are more easily queriable**: Since you've got
  key-value pairs, it's much easier to query your structured logs for
  particular values by filtering on the contents of a particular key --
  think searching request logs for error codes, Kubernetes reconcilers for
  the name and namespace of the reconciled object, etc

- **Structured logging makes it easier to have cross-referencable logs**:
  Similarly to searchability, if you maintain conventions around your
  keys, it becomes easy to gather all log lines related to a particular
  concept.
 
- **Structured logs allow better dimensions of filtering**: if you have
  structure to your logs, you've got more precise control over how much
  information is logged -- you might choose in a particular configuration
  to log certain keys but not others, only log lines where a certain key
  matches a certain value, etc, instead of just having v-levels and names
  to key off of.

- **Structured logs better represent structured data**: sometimes, the
  data that you want to log is inherently structured (think tuple-link
  objects).  Structured logs allow you to preserve that structure when
  outputting.

## Why V-levels?

**V-levels give operators an easy way to control the chattiness of log
operations**.  V-levels provide a way for a given package to distinguish
the relative importance or verbosity of a given log message.  Then, if
a particular logger or package is logging too many messages, the user
of the package can simply change the v-levels for that library. 

## Why not more named levels, like Warning?

Read [Dave Cheney's post][warning-makes-no-sense].  Then read [Differences
from Dave's ideas](#differences-from-daves-ideas).

## Why not allow format strings, too?

**Format strings negate many of the benefits of structured logs**:

- They're not easily searchable without resorting to fuzzy searching,
  regular expressions, etc

- They don't store structured data well, since contents are flattened into
  a string

- They're not cross-referencable

- They don't compress easily, since the message is not constant

(unless you turn positional parameters into key-value pairs with numerical
keys, at which point you've gotten key-value logging with meaningless
keys)

## Practical

## Why key-value pairs, and not a map?

Key-value pairs are *much* easier to optimize, especially around
allocations.  Zap (a structured logger that inspired logr's interface) has
[performance measurements](https://github.com/uber-go/zap#performance)
that show this quite nicely.

While the interface ends up being a little less obvious, you get
potentially better performance, plus avoid making users type
`map[string]string{}` every time they want to log.

## What if my V-levels differ between libraries?

That's fine.  Control your V-levels on a per-logger basis, and use the
`WithName` function to pass different loggers to different libraries.

Generally, you should take care to ensure that you have relatively
consistent V-levels within a given logger, however, as this makes deciding
on what verbosity of logs to request easier.

## But I *really* want to use a format string!

That's not actually a question.  Assuming your question is "how do
I convert my mental model of logging with format strings to logging with
constant messages":

1. figure out what the error actually is, as you'd write in a TL;DR style,
   and use that as a message

2. For every place you'd write a format specifier, look to the word before
   it, and add that as a key value pair

For instance, consider the following examples (all taken from spots in the
Kubernetes codebase):

- `klog.V(4).Infof("Client is returning errors: code %v, error %v",
  responseCode, err)` becomes `logger.Error(err, "client returned an
  error", "code", responseCode)`

- `klog.V(4).Infof("Got a Retry-After %ds response for attempt %d to %v",
  seconds, retries, url)` becomes `logger.V(4).Info("got a retry-after
  response when requesting url", "attempt", retries, "after
  seconds", seconds, "url", url)`

If you *really* must use a format string, place it as a key value, and
call `fmt.Sprintf` yourself -- for instance, `log.Printf("unable to
reflect over type %T")` becomes `logger.Info("unable to reflect over
type", "type", fmt.Sprintf("%T"))`.  In general though, the cases where
this is necessary should be few and far between.

## How do I choose my V-levels?

This is basically the only hard constraint: increase V-levels to denote
more verbose or more debug-y logs.

Otherwise, you can start out with `0` as "you always want to see this",
`1` as "common logging that you might *possibly* want to turn off", and
`10` as "I would like to performance-test your log collection stack".

Then gradually choose levels in between as you need them, working your way
down from 10 (for debug and trace style logs) and up from 1 (for chattier
info-type logs).

## How do I choose my keys

- make your keys human-readable
- constant keys are generally a good idea
- be consistent across your codebase
- keys should naturally match parts of the message string

While key names are mostly unrestricted (and spaces are acceptable),
it's generally a good idea to stick to printable ascii characters, or at
least match the general character set of your log lines.

[warning-makes-no-sense]: http://dave.cheney.net/2015/11/05/lets-talk-about-logging
