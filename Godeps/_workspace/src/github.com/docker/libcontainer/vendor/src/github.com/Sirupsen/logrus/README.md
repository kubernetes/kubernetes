# Logrus <img src="http://i.imgur.com/hTeVwmJ.png" width="40" height="40" alt=":walrus:" class="emoji" title=":walrus:"/>&nbsp;[![Build Status](https://travis-ci.org/Sirupsen/logrus.svg?branch=master)](https://travis-ci.org/Sirupsen/logrus)&nbsp;[![godoc reference](https://godoc.org/github.com/Sirupsen/logrus?status.png)][godoc]

Logrus is a structured logger for Go (golang), completely API compatible with
the standard library logger. [Godoc][godoc]. **Please note the Logrus API is not
yet stable (pre 1.0). Logrus itself is completely stable and has been used in
many large deployments. The core API is unlikely to change much but please
version control your Logrus to make sure you aren't fetching latest `master` on
every build.**

Nicely color-coded in development (when a TTY is attached, otherwise just
plain text):

![Colored](http://i.imgur.com/PY7qMwd.png)

With `log.Formatter = new(logrus.JSONFormatter)`, for easy parsing by logstash
or Splunk:

```json
{"animal":"walrus","level":"info","msg":"A group of walrus emerges from the
ocean","size":10,"time":"2014-03-10 19:57:38.562264131 -0400 EDT"}

{"level":"warning","msg":"The group's number increased tremendously!",
"number":122,"omg":true,"time":"2014-03-10 19:57:38.562471297 -0400 EDT"}

{"animal":"walrus","level":"info","msg":"A giant walrus appears!",
"size":10,"time":"2014-03-10 19:57:38.562500591 -0400 EDT"}

{"animal":"walrus","level":"info","msg":"Tremendously sized cow enters the ocean.",
"size":9,"time":"2014-03-10 19:57:38.562527896 -0400 EDT"}

{"level":"fatal","msg":"The ice breaks!","number":100,"omg":true,
"time":"2014-03-10 19:57:38.562543128 -0400 EDT"}
```

With the default `log.Formatter = new(logrus.TextFormatter)` when a TTY is not
attached, the output is compatible with the
[logfmt](http://godoc.org/github.com/kr/logfmt) format:

```text
time="2014-04-20 15:36:23.830442383 -0400 EDT" level="info" msg="A group of walrus emerges from the ocean" animal="walrus" size=10
time="2014-04-20 15:36:23.830584199 -0400 EDT" level="warning" msg="The group's number increased tremendously!" omg=true number=122
time="2014-04-20 15:36:23.830596521 -0400 EDT" level="info" msg="A giant walrus appears!" animal="walrus" size=10
time="2014-04-20 15:36:23.830611837 -0400 EDT" level="info" msg="Tremendously sized cow enters the ocean." animal="walrus" size=9
time="2014-04-20 15:36:23.830626464 -0400 EDT" level="fatal" msg="The ice breaks!" omg=true number=100
```

#### Example

The simplest way to use Logrus is simply the package-level exported logger:

```go
package main

import (
  log "github.com/Sirupsen/logrus"
)

func main() {
  log.WithFields(log.Fields{
    "animal": "walrus",
  }).Info("A walrus appears")
}
```

Note that it's completely api-compatible with the stdlib logger, so you can
replace your `log` imports everywhere with `log "github.com/Sirupsen/logrus"`
and you'll now have the flexibility of Logrus. You can customize it all you
want:

```go
package main

import (
  "os"
  log "github.com/Sirupsen/logrus"
  "github.com/Sirupsen/logrus/hooks/airbrake"
)

func init() {
  // Log as JSON instead of the default ASCII formatter.
  log.SetFormatter(&log.JSONFormatter{})

  // Use the Airbrake hook to report errors that have Error severity or above to
  // an exception tracker. You can create custom hooks, see the Hooks section.
  log.AddHook(&logrus_airbrake.AirbrakeHook{})

  // Output to stderr instead of stdout, could also be a file.
  log.SetOutput(os.Stderr)

  // Only log the warning severity or above.
  log.SetLevel(log.WarnLevel)
}

func main() {
  log.WithFields(log.Fields{
    "animal": "walrus",
    "size":   10,
  }).Info("A group of walrus emerges from the ocean")

  log.WithFields(log.Fields{
    "omg":    true,
    "number": 122,
  }).Warn("The group's number increased tremendously!")

  log.WithFields(log.Fields{
    "omg":    true,
    "number": 100,
  }).Fatal("The ice breaks!")
}
```

For more advanced usage such as logging to multiple locations from the same
application, you can also create an instance of the `logrus` Logger:

```go
package main

import (
  "github.com/Sirupsen/logrus"
)

// Create a new instance of the logger. You can have any number of instances.
var log = logrus.New()

func main() {
  // The API for setting attributes is a little different than the package level
  // exported logger. See Godoc.
  log.Out = os.Stderr

  log.WithFields(logrus.Fields{
    "animal": "walrus",
    "size":   10,
  }).Info("A group of walrus emerges from the ocean")
}
```

#### Fields

Logrus encourages careful, structured logging though logging fields instead of
long, unparseable error messages. For example, instead of: `log.Fatalf("Failed
to send event %s to topic %s with key %d")`, you should log the much more
discoverable:

```go
log.WithFields(log.Fields{
  "event": event,
  "topic": topic,
  "key": key,
}).Fatal("Failed to send event")
```

We've found this API forces you to think about logging in a way that produces
much more useful logging messages. We've been in countless situations where just
a single added field to a log statement that was already there would've saved us
hours. The `WithFields` call is optional.

In general, with Logrus using any of the `printf`-family functions should be
seen as a hint you should add a field, however, you can still use the
`printf`-family functions with Logrus.

#### Hooks

You can add hooks for logging levels. For example to send errors to an exception
tracking service on `Error`, `Fatal` and `Panic`, info to StatsD or log to
multiple places simultaneously, e.g. syslog.

```go
// Not the real implementation of the Airbrake hook. Just a simple sample.
import (
  log "github.com/Sirupsen/logrus"
)

func init() {
  log.AddHook(new(AirbrakeHook))
}

type AirbrakeHook struct{}

// `Fire()` takes the entry that the hook is fired for. `entry.Data[]` contains
// the fields for the entry. See the Fields section of the README.
func (hook *AirbrakeHook) Fire(entry *logrus.Entry) error {
  err := airbrake.Notify(entry.Data["error"].(error))
  if err != nil {
    log.WithFields(log.Fields{
      "source":   "airbrake",
      "endpoint": airbrake.Endpoint,
    }).Info("Failed to send error to Airbrake")
  }

  return nil
}

// `Levels()` returns a slice of `Levels` the hook is fired for.
func (hook *AirbrakeHook) Levels() []log.Level {
  return []log.Level{
    log.ErrorLevel,
    log.FatalLevel,
    log.PanicLevel,
  }
}
```

Logrus comes with built-in hooks. Add those, or your custom hook, in `init`:

```go
import (
  log "github.com/Sirupsen/logrus"
  "github.com/Sirupsen/logrus/hooks/airbrake"
  "github.com/Sirupsen/logrus/hooks/syslog"
  "log/syslog"
)

func init() {
  log.AddHook(new(logrus_airbrake.AirbrakeHook))

  hook, err := logrus_syslog.NewSyslogHook("udp", "localhost:514", syslog.LOG_INFO, "")
  if err != nil {
    log.Error("Unable to connect to local syslog daemon")
  } else {
    log.AddHook(hook)
  }
}
```

* [`github.com/Sirupsen/logrus/hooks/airbrake`](https://github.com/Sirupsen/logrus/blob/master/hooks/airbrake/airbrake.go)
  Send errors to an exception tracking service compatible with the Airbrake API.
  Uses [`airbrake-go`](https://github.com/tobi/airbrake-go) behind the scenes.

* [`github.com/Sirupsen/logrus/hooks/papertrail`](https://github.com/Sirupsen/logrus/blob/master/hooks/papertrail/papertrail.go)
  Send errors to the Papertrail hosted logging service via UDP.

* [`github.com/Sirupsen/logrus/hooks/syslog`](https://github.com/Sirupsen/logrus/blob/master/hooks/syslog/syslog.go)
  Send errors to remote syslog server.
  Uses standard library `log/syslog` behind the scenes.

* [`github.com/nubo/hiprus`](https://github.com/nubo/hiprus)
  Send errors to a channel in hipchat.

* [`github.com/sebest/logrusly`](https://github.com/sebest/logrusly)
  Send logs to Loggly (https://www.loggly.com/)

* [`github.com/johntdyer/slackrus`](https://github.com/johntdyer/slackrus)
  Hook for Slack chat.

* [`github.com/wercker/journalhook`](https://github.com/wercker/journalhook).
  Hook for logging to `systemd-journald`.

#### Level logging

Logrus has six logging levels: Debug, Info, Warning, Error, Fatal and Panic.

```go
log.Debug("Useful debugging information.")
log.Info("Something noteworthy happened!")
log.Warn("You should probably take a look at this.")
log.Error("Something failed but I'm not quitting.")
// Calls os.Exit(1) after logging
log.Fatal("Bye.")
// Calls panic() after logging
log.Panic("I'm bailing.")
```

You can set the logging level on a `Logger`, then it will only log entries with
that severity or anything above it:

```go
// Will log anything that is info or above (warn, error, fatal, panic). Default.
log.SetLevel(log.InfoLevel)
```

It may be useful to set `log.Level = logrus.DebugLevel` in a debug or verbose
environment if your application has that.

#### Entries

Besides the fields added with `WithField` or `WithFields` some fields are
automatically added to all logging events:

1. `time`. The timestamp when the entry was created.
2. `msg`. The logging message passed to `{Info,Warn,Error,Fatal,Panic}` after
   the `AddFields` call. E.g. `Failed to send event.`
3. `level`. The logging level. E.g. `info`.

#### Environments

Logrus has no notion of environment.

If you wish for hooks and formatters to only be used in specific environments,
you should handle that yourself. For example, if your application has a global
variable `Environment`, which is a string representation of the environment you
could do:

```go
import (
  log "github.com/Sirupsen/logrus"
)

init() {
  // do something here to set environment depending on an environment variable
  // or command-line flag
  if Environment == "production" {
    log.SetFormatter(logrus.JSONFormatter)
  } else {
    // The TextFormatter is default, you don't actually have to do this.
    log.SetFormatter(logrus.TextFormatter)
  }
}
```

This configuration is how `logrus` was intended to be used, but JSON in
production is mostly only useful if you do log aggregation with tools like
Splunk or Logstash.

#### Formatters

The built-in logging formatters are:

* `logrus.TextFormatter`. Logs the event in colors if stdout is a tty, otherwise
  without colors.
  * *Note:* to force colored output when there is no TTY, set the `ForceColors`
    field to `true`.  To force no colored output even if there is a TTY  set the
    `DisableColors` field to `true`
* `logrus.JSONFormatter`. Logs fields as JSON.

Third party logging formatters:

* [`zalgo`](https://github.com/aybabtme/logzalgo): invoking the P͉̫o̳̼̊w̖͈̰͎e̬͔̭͂r͚̼̹̲ ̫͓͉̳͈ō̠͕͖̚f̝͍̠ ͕̲̞͖͑Z̖̫̤̫ͪa͉̬͈̗l͖͎g̳̥o̰̥̅!̣͔̲̻͊̄ ̙̘̦̹̦.

You can define your formatter by implementing the `Formatter` interface,
requiring a `Format` method. `Format` takes an `*Entry`. `entry.Data` is a
`Fields` type (`map[string]interface{}`) with all your fields as well as the
default ones (see Entries section above):

```go
type MyJSONFormatter struct {
}

log.SetFormatter(new(MyJSONFormatter))

func (f *JSONFormatter) Format(entry *Entry) ([]byte, error) {
  // Note this doesn't include Time, Level and Message which are available on
  // the Entry. Consult `godoc` on information about those fields or read the
  // source of the official loggers.
  serialized, err := json.Marshal(entry.Data)
    if err != nil {
      return nil, fmt.Errorf("Failed to marshal fields to JSON, %v", err)
    }
  return append(serialized, '\n'), nil
}
```

#### Logger as an `io.Writer`

Logrus can be transormed into an `io.Writer`. That writer is the end of an `io.Pipe` and it is your responsibility to close it.

```go
w := logger.Writer()
defer w.Close()

srv := http.Server{
    // create a stdlib log.Logger that writes to
    // logrus.Logger.
    ErrorLog: log.New(w, "", 0),
}
```

Each line written to that writer will be printed the usual way, using formatters
and hooks. The level for those entries is `info`.

#### Rotation

Log rotation is not provided with Logrus. Log rotation should be done by an
external program (like `logrotate(8)`) that can compress and delete old log
entries. It should not be a feature of the application-level logger.


[godoc]: https://godoc.org/github.com/Sirupsen/logrus
