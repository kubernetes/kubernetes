# Sentry Hook for Logrus <img src="http://i.imgur.com/hTeVwmJ.png" width="40" height="40" alt=":walrus:" class="emoji" title=":walrus:" />

[Sentry](https://getsentry.com) provides both self-hosted and hosted
solutions for exception tracking.
Both client and server are
[open source](https://github.com/getsentry/sentry).

## Usage

Every sentry application defined on the server gets a different
[DSN](https://www.getsentry.com/docs/). In the example below replace
`YOUR_DSN` with the one created for your application.

```go
import (
  "github.com/Sirupsen/logrus"
  "github.com/Sirupsen/logrus/hooks/sentry"
)

func main() {
  log       := logrus.New()
  hook, err := logrus_sentry.NewSentryHook(YOUR_DSN, []logrus.Level{
    logrus.PanicLevel,
    logrus.FatalLevel,
    logrus.ErrorLevel,
  })

  if err == nil {
    log.Hooks.Add(hook)
  }
}
```

## Special fields

Some logrus fields have a special meaning in this hook,
these are server_name and logger.
When logs are sent to sentry these fields are treated differently.
- server_name (also known as hostname) is the name of the server which
is logging the event (hostname.example.com)
- logger is the part of the application which is logging the event.
In go this usually means setting it to the name of the package.

## Timeout

`Timeout` is the time the sentry hook will wait for a response
from the sentry server.

If this time elapses with no response from
the server an error will be returned.

If `Timeout` is set to 0 the SentryHook will not wait for a reply
and will assume a correct delivery.

The SentryHook has a default timeout of `100 milliseconds` when created
with a call to `NewSentryHook`. This can be changed by assigning a value to the `Timeout` field:

```go
hook, _ := logrus_sentry.NewSentryHook(...)
hook.Timeout = 20*time.Seconds
```
