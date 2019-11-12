## 0.7.0 (Unreleased)

## 0.6.0 (October 17, 2019)

UPGRADE NOTES

* The way reauthentication works has been refactored. This should not cause a problem, but please report bugs if it does. See [GH-1746](https://github.com/gophercloud/gophercloud/pull/1746) for more information.

IMPROVEMENTS

* Added `networking/v2/extensions/quotas.Get` [GH-1742](https://github.com/gophercloud/gophercloud/pull/1742)
* Added `networking/v2/extensions/quotas.Update` [GH-1747](https://github.com/gophercloud/gophercloud/pull/1747)
* Refactored the reauthentication implementation to use goroutines and added a check to prevent an infinite loop in certain situations. [GH-1746](https://github.com/gophercloud/gophercloud/pull/1746)

BUG FIXES

* Changed `Flavor` to `FlavorID` in `loadbalancer/v2/loadbalancers` [GH-1744](https://github.com/gophercloud/gophercloud/pull/1744)
* Changed `Flavor` to `FlavorID` in `networking/v2/extensions/lbaas_v2/loadbalancers` [GH-1744](https://github.com/gophercloud/gophercloud/pull/1744)
* The `go-yaml` dependency was updated to `v2.2.4` to fix possible DDOS vulnerabilities [GH-1751](https://github.com/gophercloud/gophercloud/pull/1751)

## 0.5.0 (October 13, 2019)

IMPROVEMENTS

* Added `VolumeType` to `compute/v2/extensions/bootfromvolume.BlockDevice`[GH-1690](https://github.com/gophercloud/gophercloud/pull/1690)
* Added `networking/v2/extensions/layer3/portforwarding.List` [GH-1688](https://github.com/gophercloud/gophercloud/pull/1688)
* Added `networking/v2/extensions/layer3/portforwarding.Get` [GH-1698](https://github.com/gophercloud/gophercloud/pull/1696)
* Added `compute/v2/extensions/tags.ReplaceAll` [GH-1696](https://github.com/gophercloud/gophercloud/pull/1696)
* Added `compute/v2/extensions/tags.Add` [GH-1696](https://github.com/gophercloud/gophercloud/pull/1696)
* Added `networking/v2/extensions/layer3/portforwarding.Update` [GH-1703](https://github.com/gophercloud/gophercloud/pull/1703)
* Added `ExtractDomain` method to token results in `identity/v3/tokens` [GH-1712](https://github.com/gophercloud/gophercloud/pull/1712)
* Added `AllowedCIDRs` to `loadbalancer/v2/listeners.CreateOpts` [GH-1710](https://github.com/gophercloud/gophercloud/pull/1710)
* Added `AllowedCIDRs` to `loadbalancer/v2/listeners.UpdateOpts` [GH-1710](https://github.com/gophercloud/gophercloud/pull/1710)
* Added `AllowedCIDRs` to `loadbalancer/v2/listeners.Listener` [GH-1710](https://github.com/gophercloud/gophercloud/pull/1710)
* Added `compute/v2/extensions/tags.Add` [GH-1695](https://github.com/gophercloud/gophercloud/pull/1695)
* Added `compute/v2/extensions/tags.ReplaceAll` [GH-1694](https://github.com/gophercloud/gophercloud/pull/1694)
* Added `compute/v2/extensions/tags.Delete` [GH-1699](https://github.com/gophercloud/gophercloud/pull/1699)
* Added `compute/v2/extensions/tags.DeleteAll` [GH-1700](https://github.com/gophercloud/gophercloud/pull/1700)
* Added `ImageStatusImporting` as an image status [GH-1725](https://github.com/gophercloud/gophercloud/pull/1725)
* Added `ByPath` to `baremetalintrospection/v1/introspection.RootDiskType` [GH-1730](https://github.com/gophercloud/gophercloud/pull/1730)
* Added `AttachedVolumes` to `compute/v2/servers.Server` [GH-1732](https://github.com/gophercloud/gophercloud/pull/1732)
* Enable unmarshaling server tags to a `compute/v2/servers.Server` struct [GH-1734]
* Allow setting an empty members list in `loadbalancer/v2/pools.BatchUpdateMembers` [GH-1736](https://github.com/gophercloud/gophercloud/pull/1736)
* Allow unsetting members' subnet ID and name in `loadbalancer/v2/pools.BatchUpdateMemberOpts` [GH-1738](https://github.com/gophercloud/gophercloud/pull/1738)

BUG FIXES

* Changed struct type for options in `networking/v2/extensions/lbaas_v2/listeners` to `UpdateOptsBuilder` interface instead of specific UpdateOpts type [GH-1705](https://github.com/gophercloud/gophercloud/pull/1705)
* Changed struct type for options in `networking/v2/extensions/lbaas_v2/loadbalancers` to `UpdateOptsBuilder` interface instead of specific UpdateOpts type [GH-1706](https://github.com/gophercloud/gophercloud/pull/1706)
* Fixed issue with `blockstorage/v1/volumes.Create` where the response was expected to be 202 [GH-1720](https://github.com/gophercloud/gophercloud/pull/1720)
* Changed `DefaultTlsContainerRef` from `string` to `*string` in `loadbalancer/v2/listeners.UpdateOpts` to allow the value to be removed during update. [GH-1723](https://github.com/gophercloud/gophercloud/pull/1723)
* Changed `SniContainerRefs` from `[]string{}` to `*[]string{}` in `loadbalancer/v2/listeners.UpdateOpts` to allow the value to be removed during update. [GH-1723](https://github.com/gophercloud/gophercloud/pull/1723)
* Changed `DefaultTlsContainerRef` from `string` to `*string` in `networking/v2/extensions/lbaas_v2/listeners.UpdateOpts` to allow the value to be removed during update. [GH-1723](https://github.com/gophercloud/gophercloud/pull/1723)
* Changed `SniContainerRefs` from `[]string{}` to `*[]string{}` in `networking/v2/extensions/lbaas_v2/listeners.UpdateOpts` to allow the value to be removed during update. [GH-1723](https://github.com/gophercloud/gophercloud/pull/1723)


## 0.4.0 (September 3, 2019)

IMPROVEMENTS

* Added `blockstorage/extensions/quotasets.results.QuotaSet.Groups` [GH-1668](https://github.com/gophercloud/gophercloud/pull/1668)
* Added `blockstorage/extensions/quotasets.results.QuotaUsageSet.Groups` [GH-1668](https://github.com/gophercloud/gophercloud/pull/1668)
* Added `containerinfra/v1/clusters.CreateOpts.FixedNetwork` [GH-1674](https://github.com/gophercloud/gophercloud/pull/1674)
* Added `containerinfra/v1/clusters.CreateOpts.FixedSubnet` [GH-1676](https://github.com/gophercloud/gophercloud/pull/1676)
* Added `containerinfra/v1/clusters.CreateOpts.FloatingIPEnabled` [GH-1677](https://github.com/gophercloud/gophercloud/pull/1677)
* Added `CreatedAt` and `UpdatedAt` to `loadbalancers/v2/loadbalancers.LoadBalancer` [GH-1681](https://github.com/gophercloud/gophercloud/pull/1681)
* Added `networking/v2/extensions/layer3/portforwarding.Create` [GH-1651](https://github.com/gophercloud/gophercloud/pull/1651)
* Added `networking/v2/extensions/agents.ListDHCPNetworks` [GH-1686](https://github.com/gophercloud/gophercloud/pull/1686)
* Added `networking/v2/extensions/layer3/portforwarding.Delete` [GH-1652](https://github.com/gophercloud/gophercloud/pull/1652)
* Added `compute/v2/extensions/tags.List` [GH-1679](https://github.com/gophercloud/gophercloud/pull/1679)
* Added `compute/v2/extensions/tags.Check` [GH-1679](https://github.com/gophercloud/gophercloud/pull/1679)

BUG FIXES

* Changed `identity/v3/endpoints.ListOpts.RegionID` from `int` to `string` [GH-1664](https://github.com/gophercloud/gophercloud/pull/1664)
* Fixed issue where older time formats in some networking APIs/resources were unable to be parsed [GH-1671](https://github.com/gophercloud/gophercloud/pull/1664)
* Changed `SATA`, `SCSI`, and `SAS` types to `InterfaceType` in `baremetal/v1/nodes` [GH-1683]

## 0.3.0 (July 31, 2019)

IMPROVEMENTS

* Added `baremetal/apiversions.List` [GH-1577](https://github.com/gophercloud/gophercloud/pull/1577)
* Added `baremetal/apiversions.Get` [GH-1577](https://github.com/gophercloud/gophercloud/pull/1577)
* Added `compute/v2/extensions/servergroups.CreateOpts.Policy` [GH-1636](https://github.com/gophercloud/gophercloud/pull/1636)
* Added `identity/v3/extensions/trusts.Create` [GH-1644](https://github.com/gophercloud/gophercloud/pull/1644)
* Added `identity/v3/extensions/trusts.Delete` [GH-1644](https://github.com/gophercloud/gophercloud/pull/1644)
* Added `CreatedAt` and `UpdatedAt` to `networking/v2/extensions/layer3/floatingips.FloatingIP` [GH-1647](https://github.com/gophercloud/gophercloud/issues/1646)
* Added `CreatedAt` and `UpdatedAt` to `networking/v2/extensions/security/groups.SecGroup` [GH-1654](https://github.com/gophercloud/gophercloud/issues/1654)
* Added `CreatedAt` and `UpdatedAt` to `networking/v2/networks.Network` [GH-1657](https://github.com/gophercloud/gophercloud/issues/1657)
* Added `keymanager/v1/containers.CreateSecretRef` [GH-1659](https://github.com/gophercloud/gophercloud/issues/1659)
* Added `keymanager/v1/containers.DeleteSecretRef` [GH-1659](https://github.com/gophercloud/gophercloud/issues/1659)
* Added `sharedfilesystems/v2/shares.GetMetadata` [GH-1656](https://github.com/gophercloud/gophercloud/issues/1656)
* Added `sharedfilesystems/v2/shares.GetMetadatum` [GH-1656](https://github.com/gophercloud/gophercloud/issues/1656)
* Added `sharedfilesystems/v2/shares.SetMetadata` [GH-1656](https://github.com/gophercloud/gophercloud/issues/1656)
* Added `sharedfilesystems/v2/shares.UpdateMetadata` [GH-1656](https://github.com/gophercloud/gophercloud/issues/1656)
* Added `sharedfilesystems/v2/shares.DeleteMetadatum` [GH-1656](https://github.com/gophercloud/gophercloud/issues/1656)
* Added `sharedfilesystems/v2/sharetypes.IDFromName` [GH-1662](https://github.com/gophercloud/gophercloud/issues/1662)



BUG FIXES

* Changed `baremetal/v1/nodes.CleanStep.Args` from `map[string]string` to `map[string]interface{}` [GH-1638](https://github.com/gophercloud/gophercloud/pull/1638)
* Removed `URLPath` and `ExpectedCodes` from `loadbalancer/v2/monitors.ToMonitorCreateMap` since Octavia now provides default values when these fields are not specified [GH-1640](https://github.com/gophercloud/gophercloud/pull/1540)


## 0.2.0 (June 17, 2019)

IMPROVEMENTS

* Added `networking/v2/extensions/qos/rules.ListBandwidthLimitRules` [GH-1584](https://github.com/gophercloud/gophercloud/pull/1584)
* Added `networking/v2/extensions/qos/rules.GetBandwidthLimitRule` [GH-1584](https://github.com/gophercloud/gophercloud/pull/1584)
* Added `networking/v2/extensions/qos/rules.CreateBandwidthLimitRule` [GH-1584](https://github.com/gophercloud/gophercloud/pull/1584)
* Added `networking/v2/extensions/qos/rules.UpdateBandwidthLimitRule` [GH-1589](https://github.com/gophercloud/gophercloud/pull/1589)
* Added `networking/v2/extensions/qos/rules.DeleteBandwidthLimitRule` [GH-1590](https://github.com/gophercloud/gophercloud/pull/1590)
* Added `networking/v2/extensions/qos/policies.List` [GH-1591](https://github.com/gophercloud/gophercloud/pull/1591)
* Added `networking/v2/extensions/qos/policies.Get` [GH-1593](https://github.com/gophercloud/gophercloud/pull/1593)
* Added `networking/v2/extensions/qos/rules.ListDSCPMarkingRules` [GH-1594](https://github.com/gophercloud/gophercloud/pull/1594)
* Added `networking/v2/extensions/qos/policies.Create` [GH-1595](https://github.com/gophercloud/gophercloud/pull/1595)
* Added `compute/v2/extensions/diagnostics.Get` [GH-1592](https://github.com/gophercloud/gophercloud/pull/1592)
* Added `networking/v2/extensions/qos/policies.Update` [GH-1603](https://github.com/gophercloud/gophercloud/pull/1603)
* Added `networking/v2/extensions/qos/policies.Delete` [GH-1603](https://github.com/gophercloud/gophercloud/pull/1603)
* Added `networking/v2/extensions/qos/rules.CreateDSCPMarkingRule` [GH-1605](https://github.com/gophercloud/gophercloud/pull/1605)
* Added `networking/v2/extensions/qos/rules.UpdateDSCPMarkingRule` [GH-1605](https://github.com/gophercloud/gophercloud/pull/1605)
* Added `networking/v2/extensions/qos/rules.GetDSCPMarkingRule` [GH-1609](https://github.com/gophercloud/gophercloud/pull/1609)
* Added `networking/v2/extensions/qos/rules.DeleteDSCPMarkingRule` [GH-1609](https://github.com/gophercloud/gophercloud/pull/1609)
* Added `networking/v2/extensions/qos/rules.ListMinimumBandwidthRules` [GH-1615](https://github.com/gophercloud/gophercloud/pull/1615)
* Added `networking/v2/extensions/qos/rules.GetMinimumBandwidthRule` [GH-1615](https://github.com/gophercloud/gophercloud/pull/1615)
* Added `networking/v2/extensions/qos/rules.CreateMinimumBandwidthRule` [GH-1615](https://github.com/gophercloud/gophercloud/pull/1615)
* Added `Hostname` to `baremetalintrospection/v1/introspection.Data` [GH-1627](https://github.com/gophercloud/gophercloud/pull/1627)
* Added `networking/v2/extensions/qos/rules.UpdateMinimumBandwidthRule` [GH-1624](https://github.com/gophercloud/gophercloud/pull/1624)
* Added `networking/v2/extensions/qos/rules.DeleteMinimumBandwidthRule` [GH-1624](https://github.com/gophercloud/gophercloud/pull/1624)
* Added `networking/v2/extensions/qos/ruletypes.GetRuleType` [GH-1625](https://github.com/gophercloud/gophercloud/pull/1625)
* Added `Extra` to `baremetalintrospection/v1/introspection.Data` [GH-1611](https://github.com/gophercloud/gophercloud/pull/1611)
* Added `blockstorage/extensions/volumeactions.SetImageMetadata` [GH-1621](https://github.com/gophercloud/gophercloud/pull/1621)

BUG FIXES

* Updated `networking/v2/extensions/qos/rules.UpdateBandwidthLimitRule` to use return code 200 [GH-1606](https://github.com/gophercloud/gophercloud/pull/1606)
* Fixed bug in `compute/v2/extensions/schedulerhints.SchedulerHints.Query` where contents will now be marshalled to a string [GH-1620](https://github.com/gophercloud/gophercloud/pull/1620)

## 0.1.0 (May 27, 2019)

Initial tagged release. 
