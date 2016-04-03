# Run Heapster with a Riemann back end

[Riemann](https://riemann.io) is an event processing engine which allows
sophisticated transformation, analysis, and routing of events. Configuration is
provided using Clojure.  When Heapster is configured with a Riemann sink, it
will stream Riemann-format events to a separate Riemann instance.

The following sink options are supported as URL parameters:
    host: default "riemann-heapster:5555"
        The host-port connection string for the Riemann instance.
    ttl:  default 60.0
        The default TTL (in seconds) assigned to Riemann events
    state: default nil
        The default state (string) assigned to Riemann events
    tags: default []
        An array of strings that should be associated with each Riemann event
    storeEvents: default true
        Whether Kubernetes events should be forwarded to Riemann

See the [sample config](../riemann/riemann-pagerduty-sample.config) for configuration pointers.
