package ttime

import "time"

var (
  currentTime time.Time
  timeFrozen  bool
)

type Duration time.Duration
type Location time.Location
type Month time.Month
type ParseError time.ParseError
type Ticker time.Ticker
type Time time.Time
type Timer time.Timer
type Weekday time.Weekday

var (
  // import a ton of constants so we can act like the time library.
  Parse           = time.Parse
  ParseDuration   = time.ParseDuration
  Date            = time.Date
  ParseInLocation = time.ParseInLocation
  FixedZone       = time.FixedZone
  LoadLocation    = time.LoadLocation
  Sunday          = time.Sunday
  Monday          = time.Monday
  Tuesday         = time.Tuesday
  Wednesday       = time.Wednesday
  Thursday        = time.Thursday
  Friday          = time.Friday
  Saturday        = time.Saturday
  ANSIC           = time.ANSIC
  UnixDate        = time.UnixDate
  RubyDate        = time.RubyDate
  RFC822          = time.RFC822
  RFC822Z         = time.RFC822Z
  RFC850          = time.RFC850
  RFC1123         = time.RFC1123
  RFC1123Z        = time.RFC1123Z
  RFC3339         = time.RFC3339
  RFC3339Nano     = time.RFC3339Nano
  Kitchen         = time.Kitchen
  Stamp           = time.Stamp
  StampMilli      = time.StampMilli
  StampMicro      = time.StampMicro
  StampNano       = time.StampNano
  // constants that I really should redefine:
  NewTimer  = time.NewTimer
  NewTicker = time.NewTicker
  Unix      = time.Unix
)

func Freeze(t time.Time) {
  currentTime = t
  timeFrozen = true
}

func Unfreeze() {
  timeFrozen = false
}

func IsFrozen() bool {
  return timeFrozen
}

func Now() time.Time {
  if timeFrozen {
    return currentTime
  } else {
    return time.Now()
  }
}

func After(d time.Duration) <-chan time.Time {
  if timeFrozen {
    currentTime = currentTime.Add(d)
    c := make(chan time.Time, 1)
    c <- currentTime
    return c
  } else {
    return time.After(d)
  }
}

func Tick(d time.Duration) <-chan time.Time {
  if timeFrozen {
    c := make(chan time.Time, 1)
    go func() {
      for {
        currentTime = currentTime.Add(d)
        c <- currentTime
      }
    }()
    return c
  } else {
    return time.Tick(d)
  }
}

func Sleep(d time.Duration) {
  if timeFrozen && d > 0 {
    currentTime = currentTime.Add(d)
  } else {
    time.Sleep(d)
  }
}
