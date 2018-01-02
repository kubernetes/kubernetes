# Changelog

## 0.8.0-dev.2 (2016-05-07)
- Fix an issue which may arise during sandbox cleanup (https://github.com/docker/libnetwork/pull/1157)
- Fix cleanup logic in case of ipv6 allocation failure
- Don't add /etc/hosts record if container's ip is empty (--net=none)
- Fix default gw logic for internal networks
- Error when updating IPv6 gateway (https://github.com/docker/libnetwork/issues/1142)
- Fixes https://github.com/docker/libnetwork/issues/1113
- Fixes https://github.com/docker/libnetwork/issues/1069
- Fxies https://github.com/docker/libnetwork/issues/1117
- Increase the concurrent query rate-limit count
- Changes to build libnetwork in Solaris

## 0.8.0-dev.1 (2016-04-16)
- Fixes docker/docker#16964
- Added maximum egress bandwidth qos for Windows

## 0.7.0-rc.6 (2016-04-10)
- Flush cached resolver socket on default gateway change

## 0.7.0-rc.5 (2016-04-08)
- Persist ipam driver options
- Fixes https://github.com/docker/libnetwork/issues/1087
- Use go vet from go tool
- Godep update to pick up latest docker/docker packages
- Validate remote driver response using docker plugins package method.

## 0.7.0-rc.4 (2016-04-06)
- Fix the handling for default gateway Endpoint join/leave.

## 0.7.0-rc.3 (2016-04-05)
- Revert fix for default gateway endoint join/leave. Needs to be reworked.
- Persist the network internal mode for bridge networks

## 0.7.0-rc.2 (2016-04-05)
- Fixes https://github.com/docker/libnetwork/issues/1070
- Move IPAM resource initialization out of init()
- Initialize overlay driver before network delete
- Fix the handling for default gateway Endpoint join/lean

## 0.7.0-rc.1 (2016-03-30)
- Fixes https://github.com/docker/libnetwork/issues/985
- Fixes https://github.com/docker/libnetwork/issues/945
- Log time taken to set sandbox key
- Limit number of concurrent DNS queries

## 0.7.0-dev.10 (2016-03-21)
- Add IPv6 service discovery (AAAA records) in embedded DNS server
- Honor enableIPv6 flag in network create for the IP allocation
- Avoid V6 queries in docker domain going to external nameservers

## 0.7.0-dev.9 (2016-03-18)
- Support labels on networks

## 0.7.0-dev.8 (2016-03-16)
- Windows driver to respect user set MAC address.
- Fix possible nil pointer reference in ServeDNS() with concurrent go routines.
- Fix netns path setting from hook (for containerd integration)
- Clear cached udp connections on resolver Stop()
- Avoid network/endpoint count inconsistences and remove stale networks after ungraceful shutdown
- Fix possible endpoint count inconsistency after ungraceful shutdown
- Reject a null v4 IPAM slice in exp vlan drivers
- Removed experimental drivers modprobe check

## 0.7.0-dev.7 (2016-03-11)
- Bumped up the minimum kernel version for ipvlan to 4.2
- Removed modprobe from macvlan/ipvlan drivers to resolve docker IT failures
- Close dbus connection if firewalld is not started

## 0.7.0-dev.6 (2016-03-10)
- Experimental support for macvlan and ipvlan drivers

## 0.7.0-dev.5 (2016-03-08)
- Fixes https://github.com/docker/docker/issues/20847
- Fixes https://github.com/docker/docker/issues/20997
- Fixes issues unveiled by docker integ test over 0.7.0-dev.4

## 0.7.0-dev.4 (2016-03-07)
- Changed ownership of exposed ports and port-mapping options from Endpoint to Sandbox
- Implement DNS RR in the Docker embedded DNS server
- Fixes https://github.com/docker/libnetwork/issues/984 (multi container overlay veth leak)
- Libnetwork to program container's interface MAC address
- Fixed bug in iptables.Exists() logic
- Fixes https://github.com/docker/docker/issues/20694
- Source external DNS queries from container namespace
- Added inbuilt nil IPAM driver
- Windows drivers integration fixes
- Extract hostname from (hostname.domainname). Related to https://github.com/docker/docker/issues/14282
- Fixed race in sandbox statistics read
- Fixes https://github.com/docker/libnetwork/issues/892 (docker start fails when ipv6.disable=1)
- Fixed error message on bridge network creation conflict

## 0.7.0-dev.3 (2016-02-17)
- Fixes https://github.com/docker/docker/issues/20350
- Fixes https://github.com/docker/docker/issues/20145
- Initial Windows HNS integration
- Allow passing global datastore config to libnetwork after boot
- Set Recursion Available bit in DNS query responses
- Make sure iptables chains are recreated on firewalld reload

## 0.7.0-dev.2 (2016-02-11)
- Fixes https://github.com/docker/docker/issues/20140

## 0.7.0-dev.1 (2016-02-10)
- Expose EnableIPV6 option
- discoverapi refactoring
- Fixed a few typos & docs update

## 0.6.1-rc2 (2016-02-09)
- Fixes https://github.com/docker/docker/issues/20132
- Fixes https://github.com/docker/docker/issues/20140
- Fixes https://github.com/docker/docker/issues/20019

## 0.6.1-rc1 (2016-02-05)
- Fixes https://github.com/docker/docker/issues/20026

## 0.6.0-rc7 (2016-02-01)
- Allow inter-network connections via exposed ports

## 0.6.0-rc6 (2016-01-30)
- Properly fixes https://github.com/docker/docker/issues/18814

## 0.6.0-rc5 (2016-01-26)
- Cleanup stale overlay sandboxes

## 0.6.0-rc4 (2016-01-25)
- Add Endpoints() API to Sandbox interface
- Fixed a race-condition in default gateway network creation

## 0.6.0-rc3 (2016-01-25)
- Fixes docker/docker#19576
- Fixed embedded DNS to listen in TCP as well
- Fixed a race-condition in IPAM to choose non-overlapping subnet for concurrent requests

## 0.6.0-rc2 (2016-01-21)
- Fixes docker/docker#19376
- Fixes docker/docker#15819
- Fixes libnetwork/#885, Not filter v6 DNS servers from resolv.conf
- Fixes docker/docker #19448, also handles the . in service and network names correctly.

## 0.6.0-rc1 (2016-01-14)
- Fixes docker/docker#19404
- Fixes the ungraceful daemon restart issue in systemd with remote network plugin
  (https://github.com/docker/libnetwork/issues/813)

## 0.5.6 (2016-01-14)
- Setup embedded DNS server correctly on container restart. Fixes docker/docker#19354

## 0.5.5 (2016-01-14)
- Allow network-scoped alias to be resolved for anonymous endpoint
- Self repair corrupted IP database that could happen in 1.9.0 & 1.9.1
- Skip IPTables cleanup if --iptables=false is set. Fixes docker/docker#19063

## 0.5.4 (2016-01-12)
- Removed the isNodeAlive protection when user forces an endpoint delete

## 0.5.3 (2016-01-12)
- Bridge driver supporting internal network option
- Backend implementation to support "force" option to network disconnect
- Fixing a regex in etchosts package to fix docker/docker#19080

## 0.5.2 (2016-01-08)
- Embedded DNS replacing /etc/hosts based Service Discovery
- Container local alias and Network-scoped alias support
- Backend support for internal network mode
- Support for IPAM driver options
- Fixes overlay veth cleanup issue : docker/docker#18814
- fixes docker/docker#19139
- disable IPv6 Duplicate Address Detection

## 0.5.1 (2015-12-07)
- Allowing user to assign IP Address for containers
- Fixes docker/docker#18214
- Fixes docker/docker#18380

## 0.5.0 (2015-10-30)

- Docker multi-host networking exiting experimental channel
- Introduced IP Address Management and IPAM drivers
- DEPRECATE service discovery from default bridge network
- Introduced new network UX
- Support for multiple networks in bridge driver
- Local persistence with boltdb

## 0.4.0 (2015-07-24)

- Introduce experimental version of Overlay driver
- Introduce experimental version of network plugins
- Introduce experimental version of network & service UX
- Introduced experimental /etc/hosts based service discovery
- Integrated with libkv
- Improving test coverage
- Fixed a bunch of issues with osl namespace mgmt

## 0.3.0 (2015-05-27)
 
- Introduce CNM (Container Networking Model)
- Replace docker networking with CNM & Bridge driver
