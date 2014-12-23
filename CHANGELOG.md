# Changelog

## 0.7
### Features
  - Make updating node labels easier
  - Support updating node capacity
  - kubectl streaming log support
  - Improve /validate validation
  - Fix GCE-PD to work across machine reboots
  - Don't delete other attached disks on cluster turn-down
  - Return errors if a user attempts to create a UDP external balancer
  - TLS version bump from SSLv3 to TLSv1.0
  - x509 request authenticator
  - Container VM on GCE updated to 20141208
  - Improvements to kubectl yaml handling
### Bug fixes
  - Fix kubelet panics when docker has no name for containers
  - Only count non-dead pods in replica controller status reporting
  - Fix version requirements for docker exec

## 0.6
### Features
  - Docker 1.3.3 (0.6.2)
  - Authentication for Kubelet/Apiserver communication
  - Kubectl clean ups
  - Enable Docker Cache on GCE
  - Better support for Private Repositories
### Bug fixes
  - Fixed Public IP support on non-GCE hosts
  - Fixed 32-bit build

## 0.5 (11/17/2014)
### Features
  - New client utility available: kubectl. This will eventually replace kubecfg. (#1325)
  - Services v2. We now assign IP addresses to services.  Details in #1107. (#1402)
  - Event support: (#1789, #2267, #2270, #2384)
  - Namespaces: (#1564)
  - Fixes for Docker 1.3 (#1841, #1842)
  - Support for automatically installing log saving and searching using fluentd and elasticsearch (#1610) and GCP logging (#1919).  If using elastic search, logs can be viewed with Kibana (#2013)
  - Read only API endpoint for internal lookups (#1916)
  - Lots of ground work for pluggable auth model. (#1847)
  - "run once" mode for the kubelet (#1707)
  - Restrict which minion a pod schedules on based on predicate tested agains minion labels. (#1946, #2007)
  - git based volumes: (#1945)
  - Container garbage collection.  Remove old instances of containers in the case of crash/fail loops. (#2022)
  - Publish the APIServer as a service to pods in the cluster (#1920)
  - Heapster monitoring (#2208)
  - cAdvisor 0.5.0
  - Switch default pull policy to PullIfNotPresent (#2388) except latest images
  - Initial IPv6 support (#2147)
  - Service proxy retry support (#2281)
  - Windows client build (largely untested) (#2332)
  - UDP Portals (#2191)
  - Capture application termination log (#2225)
  - pod update support (#1865, #2077, #2160)

### Cluster/Cloud support
  - Add OpenStack support with CloudProvider. (#1676)
  - Example systemd units (#1831)
  - Updated Rackspace support based on CoreOS (#1832)
  - Automatic security updates for debian based systems (#2012)
  - For debian (and GCE) pull docker (#2104), salt and etcd (#2245) from Google Cloud Storage.
  - For GCE, start with the Container VM image instead of stock debian.  This enables memcg support. (#2046)
  - Cluster install: Updated support for deploying to vSphere (#1747)
  - AWS support (#2260, #2216)

### Examples/Extras/Docs
  - Documentation on how to use SkyDNS with Kubernetes (#1845)
  - Podex (convert Docker image to pod desc) tool now supports multiple images. (#1898)
  - Documentation: 201 level walk through. (#1924)
  - Local Docker Setup: (#1716)

## 0.4 (10/14/2014)
 - Support Persistent Disk volume type
