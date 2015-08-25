<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/devel/logging.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
Logging Conventions
===================

The following conventions for the glog levels to use.  [glog](http://godoc.org/github.com/golang/glog) is globally preferred to [log](http://golang.org/pkg/log/) for better runtime control.

* glog.Errorf() - Always an error
* glog.Warningf() - Something unexpected, but probably not an error
* glog.Infof() has multiple levels:
  * glog.V(0) - Generally useful for this to ALWAYS be visible to an operator
    * Programmer errors
    * Logging extra info about a panic
    * CLI argument handling
  * glog.V(1) - A reasonable default log level if you don't want verbosity.
    * Information about config (listening on X, watching Y)
    * Errors that repeat frequently that relate to conditions that can be corrected (pod detected as unhealthy)
  * glog.V(2) - Useful steady state information about the service and important log messages that may correlate to significant changes in the system.  This is the recommended default log level for most systems.
    * Logging HTTP requests and their exit code
    * System state changing (killing pod)
    * Controller state change events (starting pods)
    * Scheduler log messages
  * glog.V(3) - Extended information about changes
    * More info about system state changes
  * glog.V(4) - Debug level verbosity (for now)
    * Logging in particularly thorny parts of code where you may want to come back later and check it

As per the comments, the practical default level is V(2).  Developers and QE environments may wish to run at V(3) or V(4). If you wish to change the log level, you can pass in `-v=X` where X is the desired maximum level to log.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/logging.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
