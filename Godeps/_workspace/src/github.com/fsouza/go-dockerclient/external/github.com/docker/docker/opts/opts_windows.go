package opts

// TODO Windows. Identify bug in GOLang 1.5.1 and/or Windows Server 2016 TP4.
// @jhowardmsft, @swernli.
//
// On Windows, this mitigates a problem with the default options of running
// a docker client against a local docker daemon on TP4.
//
// What was found that if the default host is "localhost", even if the client
// (and daemon as this is local) is not physically on a network, and the DNS
// cache is flushed (ipconfig /flushdns), then the client will pause for
// exactly one second when connecting to the daemon for calls. For example
// using docker run windowsservercore cmd, the CLI will send a create followed
// by an attach. You see the delay between the attach finishing and the attach
// being seen by the daemon.
//
// Here's some daemon debug logs with additional debug spew put in. The
// AfterWriteJSON log is the very last thing the daemon does as part of the
// create call. The POST /attach is the second CLI call. Notice the second
// time gap.
//
// time="2015-11-06T13:38:37.259627400-08:00" level=debug msg="After createRootfs"
// time="2015-11-06T13:38:37.263626300-08:00" level=debug msg="After setHostConfig"
// time="2015-11-06T13:38:37.267631200-08:00" level=debug msg="before createContainerPl...."
// time="2015-11-06T13:38:37.271629500-08:00" level=debug msg=ToDiskLocking....
// time="2015-11-06T13:38:37.275643200-08:00" level=debug msg="loggin event...."
// time="2015-11-06T13:38:37.277627600-08:00" level=debug msg="logged event...."
// time="2015-11-06T13:38:37.279631800-08:00" level=debug msg="In defer func"
// time="2015-11-06T13:38:37.282628100-08:00" level=debug msg="After daemon.create"
// time="2015-11-06T13:38:37.286651700-08:00" level=debug msg="return 2"
// time="2015-11-06T13:38:37.289629500-08:00" level=debug msg="Returned from daemon.ContainerCreate"
// time="2015-11-06T13:38:37.311629100-08:00" level=debug msg="After WriteJSON"
// ... 1 second gap here....
// time="2015-11-06T13:38:38.317866200-08:00" level=debug msg="Calling POST /v1.22/containers/984758282b842f779e805664b2c95d563adc9a979c8a3973e68c807843ee4757/attach"
// time="2015-11-06T13:38:38.326882500-08:00" level=info msg="POST /v1.22/containers/984758282b842f779e805664b2c95d563adc9a979c8a3973e68c807843ee4757/attach?stderr=1&stdin=1&stdout=1&stream=1"
//
// We suspect this is either a bug introduced in GOLang 1.5.1, or that a change
// in GOLang 1.5.1 (from 1.4.3) is exposing a bug in Windows TP4. In theory,
// the Windows networking stack is supposed to resolve "localhost" internally,
// without hitting DNS, or even reading the hosts file (which is why localhost
// is commented out in the hosts file on Windows).
//
// We have validated that working around this using the actual IPv4 localhost
// address does not cause the delay.
//
// This does not occur with the docker client built with 1.4.3 on the same
// Windows TP4 build, regardless of whether the daemon is built using 1.5.1
// or 1.4.3. It does not occur on Linux. We also verified we see the same thing
// on a cross-compiled Windows binary (from Linux).
//
// Final note: This is a mitigation, not a 'real' fix. It is still susceptible
// to the delay in TP4 if a user were to do 'docker run -H=tcp://localhost:2375...'
// explicitly.

// DefaultHTTPHost Default HTTP Host used if only port is provided to -H flag e.g. docker daemon -H tcp://:8080
const DefaultHTTPHost = "127.0.0.1"
