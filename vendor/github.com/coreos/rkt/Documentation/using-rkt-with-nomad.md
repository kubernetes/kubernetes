# Using rkt with Nomad

[Nomad][nomad] is a distributed scheduler which supports using a variety of different backends to execute _tasks_.
As of the v0.2.0 release, Nomad includes experimental support for using `rkt` as a task execution driver.
For more details, check out the [official Nomad documentation][rkt-driver].

[nomad]: https://www.nomadproject.io/
[rkt-driver]: https://www.nomadproject.io/docs/drivers/rkt.html
