# QingCloud Golang SDK [![Build Status](https://travis-ci.org/magicshui/qingcloud-go.svg?branch=master)](https://travis-ci.org/magicshui/qingcloud-go)

# V2 WIP 
V2 是一个重写的版本，和老的版本不兼容，目前正在开发中。

## API List

如下是请求的 API 列表，目前的状态：

## 主机

- [x] DescribeInstances
- [x] RunInstances
- [x] TerminateInstances
- [x] StartInstances
- [x] StopInstances
- [x] RestartInstances
- [x] ResetInstances
- [x] ResizeInstances
- [x] ModifyInstanceAttributes

## 硬盘

- [x] DescribeVolumes
- [x] CreateVolumes
- [x] DeleteVolumes
- [x] AttachVolumes
- [x] DetachVolumes
- [x] ResizeVolumes
- [x] ModifyVolumeAttributes

## 私有网络
- [x] DescribeVxnets
- [x] CreateVxnets
- [x] DeleteVxnets
- [x] JoinVxnet
- [x] LeaveVxnet
- [x] ModifyVxnetAttributes
- [x] DescribeVxnetInstances

## 路由器

- [x] DescribeRouters
- [x] CreateRouters
- [x] DeleteRouters
- [x] UpdateRouters
- [x] PowerOffRouters
- [x] PowerOnRouters
- [x] JoinRouter
- [x] LeaveRouter
- [x] ModifyRouterAttributes
- [x] DescribeRouterStatics
- [x] AddRouterStatics
- [x] ModifyRouterStaticAttributes
- [x] DeleteRouterStatics
- [x] DescribeRouterVxnets
- [x] AddRouterStaticEntries
- [x] DeleteRouterStaticEntries
- [x] ModifyRouterStaticEntryAttributes
- [x] DescribeRouterStaticEntries

## 公网IP
- [x] DescribeEips
- [x] AllocateEips
- [x] ReleaseEips
- [x] AssociateEip
- [x] DissociateEips
- [x] ChangeEipsBandwidth
- [x] ChangeEipsBillingMode
- [x] ModifyEipAttributes

## 防火墙
- [x] DescribeSecurityGroups
- [x] CreateSecurityGroup
- [x] DeleteSecurityGroups
- [x] ApplySecurityGroup
- [x] ModifySecurityGroupAttributes
- [x] DescribeSecurityGroupRules
- [x] AddSecurityGroupRules
- [x] DeleteSecurityGroupRules
- [x] ModifySecurityGroupRuleAttributes
- [x] CreateSecurityGroupSnapshot
- [x] DescribeSecurityGroupSnapshots
- [x] DeleteSecurityGroupSnapshots
- [x] RollbackSecurityGroup

## SSH 密钥
- [x] DescribeKeyPairs
- [x] CreateKeyPair
- [x] DeleteKeyPairs
- [x] AttachKeyPairs
- [x] DetachKeyPairs
- [x] ModifyKeyPairAttributes

## 映像
- [x] DescribeImages
- [x] CaptureInstance
- [x] DeleteImages
- [x] ModifyImageAttributes
- [x] GrantImageToUsers
- [x] RevokeImageFromUsers
- [x] DescribeImageUsers

## 负载均衡
- [x] CreateLoadBalancer
- [x] DescribeLoadBalancers
- [x] DeleteLoadBalancers
- [x] ModifyLoadBalancerAttributes
- [x] StartLoadBalancers
- [x] StopLoadBalancers
- [x] UpdateLoadBalancers
- [x] ResizeLoadBalancers
- [x] AssociateEipsToLoadBalancer
- [x] DissociateEipsFromLoadBalancer
- [x] AddLoadBalancerListeners
- [x] DescribeLoadBalancerListeners
- [x] DeleteLoadBalancerListeners
- [x] ModifyLoadBalancerListenerAttributes
- [x] AddLoadBalancerBackends
- [x] DescribeLoadBalancerBackends
- [x] DeleteLoadBalancerBackends
- [x] ModifyLoadBalancerBackendAttributes
- [x] CreateLoadBalancerPolicy
- [x] DescribeLoadBalancerPolicies
- [x] ModifyLoadBalancerPolicyAttributes
- [x] ApplyLoadBalancerPolicy
- [x] DeleteLoadBalancerPolicies
- [x] AddLoadBalancerPolicyRules
- [x] DescribeLoadBalancerPolicyRules
- [x] ModifyLoadBalancerPolicyRuleAttributes
- [x] DeleteLoadBalancerPolicyRules
- [x] CreateServerCertificate
- [x] DescribeServerCertificates
- [x] ModifyServerCertificateAttributes
- [x] DeleteServerCertificates

## 资源监控
- [x] GetMonitor
- [x] GetLoadBalancerMonitor
- [x] GetRDBMonitor
- [x] GetCacheMonitor
- [ ] GetZooKeeperMonitor
- [ ] GetQueueMonitor

## 备份
- [x] DescribeSnapshots
- [x] CreateSnapshots
- [x] DeleteSnapshots
- [x] ApplySnapshots
- [x] ModifySnapshotAttributes
- [x] CaptureInstanceFromSnapshot
- [x] CreateVolumeFromSnapshot

## User Data
- [x] UploadUserDataAttachment

## 内网域名别名
- [x] DescribeDNSAliases
- [x] AssociateDNSAlias
- [x] DissociateDNSAliases
- [x] GetDNSLabel

## 操作日志
- [x] DescribeJobs

## 标签
- [x] DescribeTags
- [x] CreateTag
- [x] DeleteTags
- [x] ModifyTagAttributes
- [x] AttachTags
- [x] DetachTags

## 区域
- [x] DescribeZones

## 数据库
- [x] CreateRDB
- [x] DescribeRDBs
- [x] DeleteRDBs
- [x] StartRDBs
- [x] StopRDBs
- [x] ResizeRDBs
- [x] RDBsLeaveVxnet
- [x] RDBsJoinVxnet
- [x] CreateRDBFromSnapshot
- [x] CreateTempRDBInstanceFromSnapshot
- [x] GetRDBInstanceFiles
- [ ] CopyRDBInstanceFilesToFTP
- [ ] CeaseRDBInstance
- [ ] CreateTempRDBInstanceFromSnapshot
- [ ] GetRDBMonitor
- [ ] ModifyRDBParameters
- [ ] ApplyRDBParameterGroup
- [ ] DescribeRDBParameters

## Mongo 集群
- [x] DescribeMongoNodes
- [x] DescribeMongoParameters
- [x] ResizeMongos
- [x] CreateMongo
- [x] StopMongos
- [x] StartMongos
- [x] DescribeMongos
- [x] DeleteMongos
- [x] CreateMongoFromSnapshot
- [x] ChangeMongoVxnet
- [x] AddMongoInstances
- [x] RemoveMongoInstances
- [x] ModifyMongoAttributes
- [x] ModifyMongoInstances
- [ ] GetMongoMonitor

## 缓存服务
- [x] DescribeCaches
- [x] CreateCache
- [x] StopCaches
- [x] StartCaches
- [x] RestartCaches
- [x] DeleteCaches
- [x] ResizeCaches
- [x] UpdateCache
- [x] ChangeCacheVxnet
- [x] ModifyCacheAttributes
- [x] DescribeCacheNodes
- [x] AddCacheNodes
- [x] DeleteCacheNodes
- [x] RestartCacheNodes
- [x] ModifyCacheNodeAttributes
- [x] CreateCacheFromSnapshot
- [x] DescribeCacheParameterGroups
- [x] CreateCacheParameterGroup
- [x] ApplyCacheParameterGroup
- [x] DeleteCacheParameterGroups
- [x] ModifyCacheParameterGroupAttributes
- [x] DescribeCacheParameters
- [x] UpdateCacheParameters
- [x] ResetCacheParameters

## Virtual SAN
- [ ] CreateS2Server
- [ ] DescribeS2Servers
- [ ] ModifyS2Server
- [ ] ResizeS2Servers
- [ ] DeleteS2Servers
- [ ] PowerOnS2Servers
- [ ] PowerOffS2Servers
- [ ] UpdateS2Servers
- [ ] ChangeS2ServerVxnet
- [ ] CreateS2SharedTarget
- [ ] DescribeS2SharedTargets
- [ ] DeleteS2SharedTargets
- [ ] EnableS2SharedTargets
- [ ] DisableS2SharedTargets
- [ ] ModifyS2SharedTargets
- [ ] AttachToS2SharedTarget
- [ ] DetachFromS2SharedTarget
- [ ] DescribeS2DefaultParameters

## Spark
- [x] AddSparkNodes
- [x] DeleteSparkNodes
- [x] StartSparks
- [x] StopSparks


# V1 （已经废弃）
如果还希望继续使用`v1`版本，可以通过如下的方式：

```
go get gopkg.in/magicshui/qingcloud-go.v1
```
