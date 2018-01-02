// Package health provides a generic health checking framework.
// The health package works expvar style. By importing the package the debug
// server is getting a "/debug/health" endpoint that returns the current
// status of the application.
// If there are no errors, "/debug/health" will return an HTTP 200 status,
// together with an empty JSON reply "{}". If there are any checks
// with errors, the JSON reply will include all the failed checks, and the
// response will be have an HTTP 503 status.
//
// A Check can either be run synchronously, or asynchronously. We recommend
// that most checks are registered as an asynchronous check, so a call to the
// "/debug/health" endpoint always returns immediately. This pattern is
// particularly useful for checks that verify upstream connectivity or
// database status, since they might take a long time to return/timeout.
//
// Installing
//
// To install health, just import it in your application:
//
//  import "github.com/docker/distribution/health"
//
// You can also (optionally) import "health/api" that will add two convenience
// endpoints: "/debug/health/down" and "/debug/health/up". These endpoints add
// "manual" checks that allow the service to quickly be brought in/out of
// rotation.
//
//  import _ "github.com/docker/distribution/health/api"
//
//  # curl localhost:5001/debug/health
//  {}
//  # curl -X POST localhost:5001/debug/health/down
//  # curl localhost:5001/debug/health
//  {"manual_http_status":"Manual Check"}
//
// After importing these packages to your main application, you can start
// registering checks.
//
// Registering Checks
//
// The recommended way of registering checks is using a periodic Check.
// PeriodicChecks run on a certain schedule and asynchronously update the
// status of the check. This allows CheckStatus to return without blocking
// on an expensive check.
//
// A trivial example of a check that runs every 5 seconds and shuts down our
// server if the current minute is even, could be added as follows:
//
//  func currentMinuteEvenCheck() error {
//    m := time.Now().Minute()
//    if m%2 == 0 {
//      return errors.New("Current minute is even!")
//    }
//    return nil
//  }
//
//  health.RegisterPeriodicFunc("minute_even", currentMinuteEvenCheck, time.Second*5)
//
// Alternatively, you can also make use of "RegisterPeriodicThresholdFunc" to
// implement the exact same check, but add a threshold of failures after which
// the check will be unhealthy. This is particularly useful for flaky Checks,
// ensuring some stability of the service when handling them.
//
//  health.RegisterPeriodicThresholdFunc("minute_even", currentMinuteEvenCheck, time.Second*5, 4)
//
// The lowest-level way to interact with the health package is calling
// "Register" directly. Register allows you to pass in an arbitrary string and
// something that implements "Checker" and runs your check. If your method
// returns an error with nil, it is considered a healthy check, otherwise it
// will make the health check endpoint "/debug/health" start returning a 503
// and list the specific check that failed.
//
// Assuming you wish to register a method called "currentMinuteEvenCheck()
// error" you could do that by doing:
//
//  health.Register("even_minute", health.CheckFunc(currentMinuteEvenCheck))
//
// CheckFunc is a convenience type that implements Checker.
//
// Another way of registering a check could be by using an anonymous function
// and the convenience method RegisterFunc. An example that makes the status
// endpoint always return an error:
//
//  health.RegisterFunc("my_check", func() error {
//   return Errors.new("This is an error!")
//  }))
//
// Examples
//
// You could also use the health checker mechanism to ensure your application
// only comes up if certain conditions are met, or to allow the developer to
// take the service out of rotation immediately. An example that checks
// database connectivity and immediately takes the server out of rotation on
// err:
//
//  updater = health.NewStatusUpdater()
//   health.RegisterFunc("database_check", func() error {
//    return updater.Check()
//  }))
//
//  conn, err := Connect(...) // database call here
//  if err != nil {
//    updater.Update(errors.New("Error connecting to the database: " + err.Error()))
//  }
//
// You can also use the predefined Checkers that come included with the health
// package. First, import the checks:
//
//  import "github.com/docker/distribution/health/checks
//
// After that you can make use of any of the provided checks. An example of
// using a `FileChecker` to take the application out of rotation if a certain
// file exists can be done as follows:
//
//  health.Register("fileChecker", health.PeriodicChecker(checks.FileChecker("/tmp/disable"), time.Second*5))
//
// After registering the check, it is trivial to take an application out of
// rotation from the console:
//
//  # curl localhost:5001/debug/health
//  {}
//  # touch /tmp/disable
//  # curl localhost:5001/debug/health
//  {"fileChecker":"file exists"}
//
// FileChecker only accepts absolute or relative file path. It does not work
// properly with tilde(~). You should make sure that the application has
// proper permission(read and execute permission for directory along with
// the specified file path). Otherwise, the FileChecker will report error
// and file health check is not ok.
//
// You could also test the connectivity to a downstream service by using a
// "HTTPChecker", but ensure that you only mark the test unhealthy if there
// are a minimum of two failures in a row:
//
//  health.Register("httpChecker", health.PeriodicThresholdChecker(checks.HTTPChecker("https://www.google.pt"), time.Second*5, 2))
package health
