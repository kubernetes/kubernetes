stathat
=======

This is a Go package for posting stats to your StatHat account.

For more information about StatHat, visit [www.stathat.com](http://www.stathat.com).

Installation
------------

Use `go get`:

    go get github.com/stathat/go

That's it.

Import it like this:

    import (
            "github.com/stathat/go"
    )

Usage
-----

The easiest way to use the package is with the EZ API functions.  You can add stats
directly in your code by just adding a call with a new stat name.  Once StatHat
receives the call, a new stat will be created for you.

To post a count of 1 to a stat:

    stathat.PostEZCountOne("messages sent - female to male", "something@stathat.com")

To specify the count:

    stathat.PostEZCount("messages sent - male to male", "something@stathat.com", 37)

To post a value:

    stathat.PostEZValue("ws0 load average", "something@stathat.com", 0.372)

There are also functions for the classic API.  The drawback to the classic API is
that you need to create the stats using the web interface and copy the keys it
gives you into your code.

To post a count of 1 to a stat using the classic API:

    stathat.PostCountOne("statkey", "userkey")

To specify the count:

    stathat.PostCount("statkey", "userkey", 37)

To post a value:

    stathat.PostValue("statkey", "userkey", 0.372)

Contact us
----------

We'd love to hear from you if you are using this in your projects!  Please drop us a
line: [@stat_hat](http://twitter.com/stat_hat) or [contact us here](http://www.stathat.com/docs/contact).

About
-----

Written by Patrick Crosby at [StatHat](http://www.stathat.com).  Twitter:  [@stat_hat](http://twitter.com/stat_hat)
