This is an experiment in making time easier to mock in Go tests.

You should be able to alias the ttime library to time to avoid having to change all your time.Now() methods to ttime.Now() throughout your code.

All methods return actual time.Time structs (if they were supposed to).

example code:

    import (
      time "github.com/ssoroka/ttime"
    )

    fmt.Printf("The starting time is %v", time.Now().UTC())

    // in test this will not sleep at all, but it will advance the clock 5 seconds.
    // in production, it's identical to time.Sleep
    time.Sleep(5 * time.Second)
    fmt.Printf("The time after sleeping for 5 seconds is %v", time.Now().UTC())

    time.After(10 * time.Second, func() {
      // This will execute after 10 seconds in production and immediately in tests.
      fmt.Printf("It is now %v", time.Now().UTC())
    })

example test:

    func TestFreezingTime(t *testing.T) {
      time.Freeze(time.Now()) // freeze the system clock, at least as far as ttime is concerned.

      // or freeze time at a specific date/time (eg, test leap-year support!):
      now, err := time.Parse(time.RFC3339, "2012-02-29T00:00:00Z")
      if err != nil { panic("date time parse failed") }
      time.Freeze(now)
      defer time.Unfreeze()

      // test leap-year-specific code
      if !isLeapYear(time.Now()) {
        t.Error("Oh no! isLeapYear is broken!")
      }

      t.Logf("It is now %v", time.Now().UTC())
    }
