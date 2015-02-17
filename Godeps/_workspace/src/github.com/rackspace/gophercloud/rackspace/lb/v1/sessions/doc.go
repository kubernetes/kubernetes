/*
Package sessions provides information and interaction with the Session
Persistence feature of the Rackspace Cloud Load Balancer service.

Session persistence is a feature of the load balancing service that forces
multiple requests from clients (of the same protocol) to be directed to the
same node. This is common with many web applications that do not inherently
share application state between back-end servers.

There are two modes to choose from: HTTP_COOKIE and SOURCE_IP. You can only set
one of the session persistence modes on a load balancer, and it can only
support one protocol. If you set HTTP_COOKIE mode for an HTTP load balancer, it
supports session persistence for HTTP requests only. Likewise, if you set
SOURCE_IP mode for an HTTPS load balancer, it supports session persistence for
only HTTPS requests.

To support session persistence for both HTTP and HTTPS requests concurrently,
choose one of the following options:

- Use two load balancers, one configured for session persistence for HTTP
requests and the other configured for session persistence for HTTPS requests.
That way, the load balancers support session persistence for both HTTP and
HTTPS requests concurrently, with each load balancer supporting one of the
protocols.

- Use one load balancer, configure it for session persistence for HTTP requests,
and then enable SSL termination for that load balancer. The load balancer
supports session persistence for both HTTP and HTTPS requests concurrently.
*/
package sessions
