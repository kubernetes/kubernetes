/*
Package ssl provides information and interaction with the SSL Termination
feature of the Rackspace Cloud Load Balancer service.

You may only enable and configure SSL termination on load balancers with
non-secure protocols, such as HTTP, but not HTTPS.

SSL-terminated load balancers decrypt the traffic at the traffic manager and
pass unencrypted traffic to the back-end node. Because of this, the customer's
back-end nodes don't know what protocol the client requested. For this reason,
the X-Forwarded-Proto (XFP) header has been added for identifying the
originating protocol of an HTTP request as "http" or "https" depending on what
protocol the client requested.

Not every service returns certificates in the proper order. Please verify that
your chain of certificates matches that of walking up the chain from the domain
to the CA root.

If used for HTTP to HTTPS redirection, the LoadBalancer's securePort attribute
must be set to 443, and its secureTrafficOnly attribute must be true.
*/
package ssl
