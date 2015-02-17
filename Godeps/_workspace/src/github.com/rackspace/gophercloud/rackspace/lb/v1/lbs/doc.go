/*
Package lbs provides information and interaction with the Load Balancer API
resource for the Rackspace Cloud Load Balancer service.

A load balancer is a logical device which belongs to a cloud account. It is
used to distribute workloads between multiple back-end systems or services,
based on the criteria defined as part of its configuration. This configuration
is defined using the Create operation, and can be updated with Update.

To conserve IPv4 address space, it is highly recommended that you share Virtual
IPs between load balancers. If you have at least one load balancer, you may
create subsequent ones that share a single virtual IPv4 and/or a single IPv6 by
passing in a virtual IP ID to the Update operation (instead of a type). This
feature is also highly desirable if you wish to load balance both an insecure
and secure protocol using one IP or DNS name. In order to share a virtual IP,
each Load Balancer must utilize a unique port.

All load balancers have a Status attribute that shows the current configuration
status of the device. This status is immutable by the caller and is updated
automatically based on state changes within the service. When a load balancer
is first created, it is placed into a BUILD state while the configuration is
being generated and applied based on the request. Once the configuration is
applied and finalized, it is in an ACTIVE status. In the event of a
configuration change or update, the status of the load balancer changes to
PENDING_UPDATE to signify configuration changes are in progress but have not yet
been finalized. Load balancers in a SUSPENDED status are configured to reject
traffic and do not forward requests to back-end nodes.

An HTTP load balancer has the X-Forwarded-For (XFF) HTTP header set by default.
This header contains the originating IP address of a client connecting to a web
server through an HTTP proxy or load balancer, which many web applications are
already designed to use when determining the source address for a request.

It also includes the X-Forwarded-Proto (XFP) HTTP header, which has been added
for identifying the originating protocol of an HTTP request as "http" or
"https" depending on which protocol the client requested. This is useful when
using SSL termination.

Finally, it also includes the X-Forwarded-Port HTTP header, which has been
added for being able to generate secure URLs containing the specified port.
This header, along with the X-Forwarded-For header, provides the needed
information to the underlying application servers.
*/
package lbs
