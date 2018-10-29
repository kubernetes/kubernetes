clock
====

[![Build Status](https://travis-ci.org/jmhodges/clock.png?branch=master)](https://travis-ci.org/jmhodges/clock)

Package clock provides an abstraction for system time that enables
testing of time-sensitive code.

Where you'd use time.Now, instead use clk.Now where clk is an instance
of Clock.

When running your code in production, pass it a Clock given by
Default() and when you're running it in your tests, pass it an instance of Clock from NewFake().

When you do that, you can use FakeClock's Add and Set methods to
control how time behaves in your code making them more reliable while
also expanding the space of problems you can test.

This code intentionally does not attempt to provide an abstraction
over time.Ticker and time.Timer because Go does not have the runtime
or API hooks available to do reliably. See
https://github.com/golang/go/issues/8869

Be sure to test Time equality with time.Time#Equal, not ==.

For documentation, see the
[godoc](http://godoc.org/github.com/jmhodges/clock).
