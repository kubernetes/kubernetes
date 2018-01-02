
This document provides a TLD&R version of https://docs.docker.com/v1.6/articles/networking/.
If more interested in detailed operational design, please refer to this link.

## Docker Networking design as of Docker v1.6

Prior to libnetwork, Docker Networking was handled in both Docker Engine and libcontainer.
Docker Engine makes use of the Bridge Driver to provide single-host networking solution with the help of linux bridge and IPTables.
Docker Engine provides simple configurations such as `--link`, `--expose`,... to enable container connectivity within the same host by abstracting away networking configuration completely from the Containers.
For external connectivity, it relied upon NAT & Port-mapping 

Docker Engine was responsible for providing the configuration for the container's networking stack.

Libcontainer would then use this information to create the necessary networking devices and move them in to a network namespace.
This namespace would then be used when the container is started.
