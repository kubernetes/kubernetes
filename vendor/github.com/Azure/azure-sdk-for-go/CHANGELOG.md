# CHANGELOG

## `v11.1.0-beta`

### ARM

- trafficmanager and containerregistry SDKs now reflect the services faithfully
- trafficmanager also has a new operation group: user metrics.

### Generated code notes
- [Azure REST API specs](https://github.com/Azure/azure-rest-api-specs) commit: c97a18ed775029207715b09c80761334724740b9
- [AutoRest Go Generator](https://github.com/Azure/autorest.go) commit: 5d984152f2e9cff6878ea5060bed7e8d8a2ae1cc

## `v11.0.0-beta`

### ARM

| api                                 | version            | note                                |
|:------------------------------------|:-------------------|:------------------------------------|
| arm/analysisservices                | 2017-08-01-beta    | update                              |
| arm/batch                           | 2017-05-01         | update                              |
| arm/cdn                             | 2017-04-02         | update                              |
| arm/cognitiveservices               | 2017-04-18         | update                              |
| arm/compute                         | multiple           | update                              |
| arm/containerregistry               | 2017-10-01         | update                              |
| arm/customerinsights                | 2017-04-26         | update                              |
| arm/eventgrid                       | 2017-09-15-preview | update                              |
| arm/eventhub                        | 2017-04-01         | update                              |
| arm/graphrbac                       | 1.6                | update                              |
| arm/iothub                          | 2017-07-01         | update                              |
| arm/keyvault                        | 2016-10-01         | update                              |
| arm/marketplaceordering             | 2015-06-01         | new                                 |
| arm/opertionalinsights              | multiple           | update                              |
| arm/operationsmanagement            | 2015-11-01-preview | new                                 |
| arm/recoveryservices                | multiple           | update                              |
| arm/recoveryservicesbackup          | multiple           | update                              |
| arm/redis                           | 2017-02-01         | update                              |
| arm/relay                           | 2017-04-01         | update                              |
| arm/resourcehealth                  | 017-07-01          | update                              |
| arm/resources/resources             | 2017-05-10         | update                              |
| arm/servicebus                      | 2017-04-01         | update                              |
| arm/storage                         | 2017-06-01         | update                              |
| arm/streamanalytics                 | 2016-03-01         | update                              |
| arm/trafficmanager                  | 2017-09-01-preview | update                              |
| arm/visualstudio                    | 2014-04-01-preview | update                              |

### Data plane

| dataplane/cognitiveservices/face          | 1.0          | new                                 |
| dataplane/cognitiveservices/textanalytics | v2.0         | new                                 |

### Storage

- Support for queue SAS.
- Refactored GetSASURI blob operation to be more complete.
- Added a SAS client for some operations (`container.Exists()`, and `container.ListBlobs()`)

- [Azure REST API specs](https://github.com/Azure/azure-rest-api-specs) commit: 0c2a12b50d8598f68d6715b507f7dd53e163407e
- [AutoRest Go Generator](https://github.com/Azure/autorest.go) commit: 678110f012c7cde6528a1e61d125bdc7ea636b7f

## `v10.3.1-beta`
- Added Apache notice file.

### ARM
- Fixed package name on some `version.go` files.

### Storage
- Fixed bug related to SAS URI generation and storage emulator support.

### Generated code notes
- [Azure REST API specs](https://github.com/Azure/azure-rest-api-specs) commit: ad55af74f3f0e2b390a4306532528168ba742fef
- [AutoRest Go extension](https://github.com/Azure/autorest.go) commit: 28a531c59c82cf67bc90c87095c1d34a936461b4

## `v10.3.0-beta`
### ARM

| api                                 | version            | note                                |
|:------------------------------------|:-------------------|:------------------------------------|
| arm/containerinstance               | 2017-08-01-preview | new                                 |
| arm/eventgrid                       | 2017-06-15-preview | new                                 |

### ASM
- Marked as in mainteinance mode.
- Added Go 1.7 build tags.

### Storage
- Support for Go 1.7 and Go 1.6 (except table batch operation tests).

### Generated code notes
- [Azure REST API specs](https://github.com/Azure/azure-rest-api-specs) commit: ad55af74f3f0e2b390a4306532528168ba742fef
- [AutoRest](https://github.com/Azure/autorest) commit: cfb296f153f948f85afab637f7212fcfdc4a8bbb

## `v10.2.1-beta`
- Fixes polymorphic structs in `mysql` and `postgresql` packages.

## `v10.2.0-beta`
### ARM

| api                                 | version            | note                                |
|:------------------------------------|:-------------------|:------------------------------------|
| arm/cosmos-db                       | 2015-04-08         | new                                 |
| arm/mysql                           | 2017-04-30-preview | new                                 |
| arm/postgresql                      | 2017-04-30-preview | new                                 |

### Storage
- Bug fixes.

### Generated code notes
- [Azure REST API specs](https://github.com/Azure/azure-rest-api-specs) commit: 485ded7560c6309efb2f795ec6e46b7436dc6fdb
- [AutoRest](https://github.com/Azure/autorest) commit: c180952b850e677a8624655abeaded307d95cae3

## `v10.1.0-beta`
### ARM

| arm/recoveryservicessiterecovery    | 2016-08-10         | new                                 |
| arm/managedapplications             | 2016-09-01-preview | new                                 |
| arm/storsimple8000series            | 2017-06-01         | new                                 |
| arm/streamanalytics                 | multiple           | new                                 |

### Storage
- Bug fixes.

### Generated code notes
- [Azure REST API specs](https://github.com/Azure/azure-rest-api-specs) commit: a2cdf005407b81edb161c1f7b5c49b5ce8e7f041
- [AutoRest](https://github.com/Azure/autorest) commit: 8e9c2d3704a04913a175ab76972b7d9597c77687

-----
## `v10.0.0-beta`
### ARM
In addition to the tabulated changes below, each package had the following updates:
- Long running operations now run inside a goroutine and return channels for the response and the errors.
- Some functions changed from returning `autorest.Response` to return the already unmarshaled struct.
- Uses go-autorest v8.0.0.

| api                                 | version            | note                                |
|:------------------------------------|:-------------------|:------------------------------------|
| arm/advisor                         | 2017-04-19         | new                                 |
| arm/analysisservices                | 2016-05-16         | refactor                            |
| arm/apimanagement                   | 2016-10-10         | update to latest swagger & refactor |
| arm/appinsights                     | 2015-05-01         | new                                 |
| arm/automation                      | 2015-10-31         | new                                 |
| arm/billing                         | 2017-04-24-preview | update to latest swagger & refactor |
| arm/cdn                             | 2016-10-02         | refactor                            |
| arm/commerce                        | 2015-06-01-preview | refactor                            |
| arm/compute                         | 2016-04-30-preview | refactor                            |
| arm/consumption                     | 2017-04-24-preview | new                                 |
| arm/containerregistry               | 2017-03-01         | update to latest swagger & refactor |
| arm/containerservice                | 2017-01-31         | update to latest swagger & refactor |
| arm/customer-insights               | 2017-01-01         | refactor                            |
| arm/datalake-analytics/account      | 2016-11-01         | refactor                            |
| arm/datalake-store/account          | 2016-11-01         | refactor                            |
| arm/devtestlabs                     | 2016-05-15         | refactor                            |
| arm/disk                            | 2016-04-30-preview | refactor                            |
| arm/dns                             | 2016-04-01         | refactor                            |
| arm/documentdb                      | 2015-04-08         | refactor                            |
| arm/eventhub                        | 2015-08-01         | refactor                            |
| arm/graphrbac                       | 1.6                | refactor                            |
| arm/hdinsight                       | 2015-03-01-preview | new                                 |
| arm/insights                        | multiple           | new                                 |
| arm/intune                          | 2015-01-14-preview | refactor                            |
| arm/iothub                          | 2016-02-03         | refactor                            |
| arm/machinelearning/commitmentplans | 2016-05-01-preview | refactor                            |
| arm/machinelearning/webservices     | 2017-01-01         | update to latest swagger & refactor |
| arm/monitor                         | multiple           | new                                 |
| arm/network                         | 2017-03-01         | update to latest swagger & refactor |
| arm/notificationhubs                | 2017-04-01         | update to latest swagger & refactor |
| arm/operationalinsights             | 2015-11-01-preview | update to latest swagger & refactor |
| arm/powerbiembedded                 | 2016-01-29         | refactor                            |
| arm/recoveryservices                | 2016-12-01         | refactor                            |
| arm/recoveryservicesbackup          | 2016-12-01         | new                                 |
| arm/redis                           | 2016-04-01         | refactor                            |
| arm/relay                           | 2016-07-01         | new                                 |
| arm/resourcehealth                  | 2015-01-01         | new                                 |
| arm/resources/features              | 2015-12-01         | refactor                            |
| arm/resources/links                 | 2016-09-01         | refactor                            |
| arm/resources/resources             | 2016-09-01         | refactor                            |
| arm/resources/subscriptions         | 2016-06-01         | refactor                            |
| arm/scheduler                       | 2016-03-01         | refactor                            |
| arm/servermanagement                | 2016-07-01-preview | refactor                            |
| arm/servicebus                      | 2015-08-01         | refactor                            |
| arm/servicefabric                   | 2016-09-01         | new                                 |
| arm/service-map                     | 2015-11-01-preview | refactor                            |
| arm/sql                             | multiple           | update to latest swagger & refactor |
| arm/storage                         | 2016-12-01         | update to latest swagger & refactor |
| arm/storageimportexport             | 2016-11-01         | refactor                            |
| arm/web                             | multiple           | refactor                            |

### Data plane
| api                                 | version            | note                                |
|:------------------------------------|:-------------------|:------------------------------------|
| dataplane/keyvault                  | 2016-10-01         | refactor                            |

### Storage
Storage has returned to this repo.
It has also been refactored:
- Blobs, containers, tables, etc are now method receivers. These structs are the ones being
  updated with each operation.
- When creating a client, the SDK checks if the storage account provided is valid.
- Added retry logic. It provides the flexibility for user to provide their own retry logic.
- Added operations:
   - Get table
   - Get entity
   - Get and set queue ACL
   - Table batch
   - Page blob incremental copy
- All operations that previously had `extraHeaders` as parameter now recieve a struct with well
  defined possible headers and other options. Some functions are easier to use.
- Storage tests now use HTTP recordings.

### Generated code notes
- [Azure REST API specs](https://github.com/Azure/azure-rest-api-specs) commit: 519980465d9c195622d466dc4601b1999a448ed5
- [AutoRest](https://github.com/Azure/autorest) commit: ced950d64e39735b84d41876a56b54b27c227dc7

## `v9.0.0-beta`
### ARM
In addition to the tabulated changes below, each package had the following updates:
 - API Version is now associated with individual methods, instead of the client. This was done to
   support composite swaggers, which logically may contain more than one API Version.
 - Version numbers are now calculated in the generator instead of at runtime. This keeps us from
   adding new allocations, while removing the race-conditions that were added.

| api                                 | version            | note                               |
|:------------------------------------|:-------------------|:-----------------------------------|
| arm/analysisservices                | 2016-05-16         | update to latest swagger           |
| arm/authorization                   | 2015-07-01         | refactoring                        |
| arm/batch                           | 2017-01-01         | update to latest swagger &refactor |
| arm/cdn                             | 2016-10-02         | update to latest swagger           |
| arm/compute                         | 2016-04-30-preview | update to latest swagger           |
| arm/dns                             | 2016-04-01         | update to latest swagger &refactor |
| arm/eventhub                        | 2015-08-01         | refactoring                        |
| arm/logic                           | 2016-06-01         | update to latest swagger &refactor |
| arm/notificationshub                | 2016-03-01         | update to latest swagger &refactor |
| arm/redis                           | 2016-04-01         | update to latest swagger &refactor |
| arm/resources/resources             | 2016-09-01         | update to latest swagger           |
| arm/servicebus                      | 2015-08-01         | update to latest swagger           |
| arm/sql                             | 2014-04-01         | update to latest swagger           |
| arm/web                             | multiple           | generating from composite          |
| datalake-analytics/account          | 2016-11-01         | update to latest swagger           |
| datalake-store/filesystem           | 2016-11-01         | update to latest swagger           |

### Storage
Storage has been moved to its own repository which can be found here:
https://github.com/Azure/azure-storage-go

For backwards compatibility, a submodule has been added to this repo. However, consuming storage
via this repository is deprecated and may be deleted in future versions. 

## `v8.1.0-beta`
### ARM
| api                                 | version            | note                               |
|:------------------------------------|:-------------------|:-----------------------------------|
| arm/apimanagement                   | 2016-07-07         | new                                |
| arm/apideployment                   | 2016-07-07         | new                                |
| arm/billing                         | 2017-02-27-preview | new                                |
| arm/compute                         | 2016-04-30-preview | update to latest swagger           |
| arm/containerservice                | 2017-01-31         | update to latest swagger           |
| arm/customer-insights               | 2017-01-01         | new                                |
| arm/graphrbac                       | 1.6                | new                                |
| arm/networkwatcher                  | 2016-12-01         | new                                |
| arm/operationalinsights             | 2015-11-01-preview | new                                |
| arm/service-map                     | 2015-11-01-preview | new                                |
| arm/storageimportexport             | 2016-11-01         | new                                |

### Data plane
| api                                 | version            | note                               |
|:------------------------------------|:-------------------|:-----------------------------------|
| dataplane/keyvault                  | 2016-10-01         | new                                |

- Uses go-autorest v7.3.0


## `v8.0.0-beta`
### ARM
- In addition to the tablulated changes below, all updated packages received performance
  improvements to their Version() method.
- Some validation that was taking place in the runtime was erroneously blocking calls.
  all packages have been updated to take that bug fix.

| api                                 | version            | note                               |
|:------------------------------------|:-------------------|:-----------------------------------|
| arm/analysisservices                | 2016-05-16         | update to latest swagger           |
| arm/cdn                             | 2016-10-02         | update to latest swagger           |
| arm/cognitiveservices               | 2016-02-01-preview | update to latest swagger           |
| arm/compute                         | 2016-03-30         | update to latest swagger, refactor |
| arm/containerregistry               | 2016-06-27-preview | update to latest swagger           |
| arm/containerservice                | 2016-09-30         | update to latest swagger           |
| arm/datalake-analytics              | 2016-11-01         | update to latest swagger           |
| arm/datalake-store                  | 2016-11-01         | update to latest swagger           |
| arm/disk                            | 2016-04-30-preview | new                                |
| arm/documentdb                      | 2015-04-08         | update to latest swagger           |
| arm/iothub                          | 2016-02-03         | update to latest swagger           |
| arm/keyvault                        | 2015-06-01         | update to latest swagger           |
| arm/logic                           | 2016-06-01         | update to latest swagger           |
| arm/machinelearning                 | 2016-05-01-preview | update to latest swagger           |
| arm/mobileengagement                | 2014-12-01         | update to latest swagger, refactor |
| arm/redis                           | 2016-04-01         | update to latest swagger           |
| arm/resources/locks                 | 2016-09-01         | refactor                           |
| arm/resources/policy                | 2016-12-01         | previous version was deleted       |
| arm/resources/resources             | 2016-09-01         | update to latest swagger, refactor |
| arm/scheduler                       | 2016-03-01         | refactor                           |
| arm/search                          | 2015-08-19         | refactor                           |
| arm/web                             | 2015-08-01         | refactor                           |

## `v7.0.0-beta`

| api                                 | version            | note                               |
|:------------------------------------|:-------------------|:-----------------------------------|
| arm/analysisservices                | 2016-05-16         | new                                |
| arm/cdn                             | 2016-10-02         | update to latest swagger           |
| arm/commerce                        | 2015-06-01-preview | new                                |
| arm/containerservice                | 2016-09-30         | update to latest swagger           |
| arm/containerregistry               | 2016-06-27-preview | new                                |
| arm/datalake-analytics/account      | 2016-11-01         | update to latest swagger           |
| arm/datalake-store/account          | 2016-11-01         | update to latest swagger           |
| arm/datalake-store/filesystem       | 2016-11-01         | update to latest swagger           |
| arm/documentdb                      | 2015-04-08         | new                                |
| arm/machinelearning/commitmentplans | 2016-05-01-preview | new                                |
| arm/recoveryservices                | 2016-06-01         | new                                |
| arm/resources/subscriptions         | 2016-06-01         | new                                |
| arm/search                          | 2015-08-19         | update to latest swagger           |
| arm/sql                             | 2014-04-01         | previous version was deleted       |

### Storage
- Can now update messages in storage queues.
- Added support for blob snapshots and aborting blob copy operations.
- Added support for getting and setting ACLs on containers.
- Added various APIs for file and directory manipulation.

### Support for the following swagger extensions was added to the Go generator which affected codegen.
- x-ms-client-flatten
- x-ms-paramater-location

## `v6.0.0-beta`

| api                            | version            | note                               |
|:-------------------------------|:-------------------|:-----------------------------------|
| arm/authorization              | no change          | code refactoring                   |
| arm/batch                      | no change          | code refactoring                   |
| arm/compute                    | no change          | code refactoring                   |
| arm/containerservice           | 2016-03-30         | return                             |
| arm/datalake-analytics/account | 2015-10-01-preview | new                                |
| arm/datalake-store/filesystem  | no change          | moved to datalake-store/filesystem |
| arm/eventhub                   | no change          | code refactoring                   |
| arm/intune                     | no change          | code refactoring                   |
| arm/iothub                     | no change          | code refactoring                   |
| arm/keyvault                   | no change          | code refactoring                   |
| arm/mediaservices              | no change          | code refactoring                   |
| arm/network                    | no change          | code refactoring                   |
| arm/notificationhubs           | no change          | code refactoring                   |
| arm/redis                      | no change          | code refactoring                   |
| arm/resources/resources        | no change          | code refactoring                   |
| arm/resources/links            | 2016-09-01         | new                                |
| arm/resources/locks            | 2016-09-01         | updated                            |
| arm/resources/policy           | no change          | code refactoring                   |
| arm/resources/resources        | 2016-09-01         | updated                            |
| arm/servermanagement           | 2016-07-01-preview | updated                            |
| arm/web                        | no change          | code refactoring                   |

- storage: Added blob lease functionality and tests

## `v5.0.0-beta`

| api                           | version             | note             |
|:------------------------------|:--------------------|:-----------------|
| arm/network                   | 2016-09-01          | updated          |
| arm/servermanagement          | 2015-07-01-preview  | new              |
| arm/eventhub                  | 2015-08-01          | new              |
| arm/containerservice          | --                  | removed          |
| arm/resources/subscriptions   | no change           | code refactoring |
| arm/resources/features        | no change           | code refactoring |
| arm/resources/resources       | no change           | code refactoring |
| arm/datalake-store/accounts   | no change           | code refactoring |
| arm/datalake-store/filesystem | no change           | code refactoring |
| arm/notificationhubs          | no change           | code refactoring |
| arm/redis                     | no change           | code refactoring |

- storage: Add more file storage share operations.
- azure-rest-api-specs/commit/b8cdc2c50a0872fc0039f20c2b6b33aa0c2af4bf
- Uses go-autorest v7.2.1

## `v4.0.0-beta`

- arm/logic: breaking change in package logic.
- arm: parameter validation code added in all arm packages.
- Uses go-autorest v7.2.0.


## `v3.2.0-beta`

| api                         | version             | note      |
|:----------------------------|:--------------------|:----------|
| arm/mediaservices           | 2015-10-01          | new       |
| arm/keyvault                | 2015-06-01          | new       |
| arm/iothub                  | 2016-02-03          | new       |
| arm/datalake-store          | 2015-12-01          | new       |
| arm/network                 | 2016-06-01          | updated   |
| arm/resources/resources     | 2016-07-01          | updated   |
| arm/resources/policy        | 2016-04-01          | updated   |
| arm/servicebus              | 2015-08-01          | updated   |

- arm: uses go-autorest version v7.1.0.
- storage: fix for operating on blobs names containing special characters.
- storage: add SetBlobProperties(), update BlobProperties response fields.
- storage: make storage client work correctly with read-only secondary account.
- storage: add Azure Storage Emulator support.


## `v3.1.0-beta`

- Added a new arm/compute/containerservice (2016-03-30) package
- Reintroduced NewxxClientWithBaseURI method.
- Uses go-autorest version - v7.0.7.


## `v3.0.0-beta`

This release brings the Go SDK ARM packages up-to-date with Azure ARM Swagger files for most
services. Since the underlying [Swagger files](https://github.com/Azure/azure-rest-api-specs)
continue to change substantially, the ARM packages are still in *beta* status.

The ARM packages now align with the following API versions (*highlighted* packages are new or
updated in this release):

| api                         | version             | note      |
|:----------------------------|:--------------------|:----------|
| arm/authorization           | 2015-07-01          | no change |
| arm/intune                  | 2015-01-14-preview  | no change |
| arm/notificationhubs        | 2014-09-01          | no change |
| arm/resources/features      | 2015-12-01          | no change |
| arm/resources/subscriptions | 2015-11-01          | no change |
| arm/web                     | 2015-08-01          | no change |
| arm/cdn                     | 2016-04-02          | updated   |
| arm/compute                 | 2016-03-30          | updated   |
| arm/dns                     | 2016-04-01          | updated   |
| arm/logic                   | 2015-08-01-preview  | updated   |
| arm/network                 | 2016-03-30          | updated   |
| arm/redis                   | 2016-04-01          | updated   |
| arm/resources/resources     | 2016-02-01          | updated   |
| arm/resources/policy        | 2015-10-01-preview  | updated   |
| arm/resources/locks         | 2015-01-01          | updated (resources/authorization earlier)|
| arm/scheduler               | 2016-03-01          | updated   |
| arm/storage                 | 2016-01-01          | updated   |
| arm/search                  | 2015-02-28          | updated   |
| arm/batch                   | 2015-12-01          | new       |
| arm/cognitiveservices       | 2016-02-01-preview  | new       |
| arm/devtestlabs             | 2016-05-15          | new       |
| arm/machinelearning         | 2016-05-01-preview  | new       |
| arm/powerbiembedded         | 2016-01-29          | new       |
| arm/mobileengagement        | 2014-12-01          | new       |
| arm/servicebus              | 2014-09-01          | new       |
| arm/sql                     | 2015-05-01          | new       |
| arm/trafficmanager          | 2015-11-01          | new       |


Below are some design changes.
- Removed Api version from method arguments.
- Removed New...ClientWithBaseURI() method in all clients. BaseURI value is set in client.go.
- Uses go-autorest version v7.0.6.


## `v2.2.0-beta`

- Uses go-autorest version v7.0.5.
- Update version of pacakges "jwt-go" and "crypto" in glide.lock.


## `v2.1.1-beta`

- arm: Better error messages for long running operation failures (Uses go-autorest version v7.0.4).


## `v2.1.0-beta`

- arm: Uses go-autorest v7.0.3 (polling related updates).
- arm: Cancel channel argument added in long-running calls.
- storage: Allow caller to provide headers for DeleteBlob methods.
- storage: Enables connection sharing with http keepalive.
- storage: Add BlobPrefixes and Delimiter to BlobListResponse


## `v2.0.0-beta`

- Uses go-autorest v6.0.0 (Polling and Asynchronous requests related changes).

 
## `v0.5.0-beta`

Updated following packages to new API versions:
- arm/resources/features 2015-12-01
- arm/resources/resources 2015-11-01
- arm/resources/subscriptions 2015-11-01


### Changes 

 - SDK now uses go-autorest v3.0.0.



## `v0.4.0-beta`

This release brings the Go SDK ARM packages up-to-date with Azure ARM Swagger files for most
services. Since the underlying [Swagger files](https://github.com/Azure/azure-rest-api-specs)
continue to change substantially, the ARM packages are still in *beta* status.

The ARM packages now align with the following API versions (*highlighted* packages are new or
updated in this release):

- *arm/authorization 2015-07-01*
- *arm/cdn 2015-06-01*
- arm/compute 2015-06-15
- arm/dns 2015-05-04-preview
- *arm/intune 2015-01-14-preview*
- arm/logic 2015-02-01-preview
- *arm/network 2015-06-15*
- *arm/notificationhubs 2014-09-01*
- arm/redis 2015-08-01
- *arm/resources/authorization 2015-01-01*
- *arm/resources/features 2014-08-01-preview*
- *arm/resources/resources 2014-04-01-preview*
- *arm/resources/subscriptions 2014-04-01-preview*
- *arm/scheduler 2016-01-01*
- arm/storage 2015-06-15
- arm/web 2015-08-01

### Changes

- Moved the arm/authorization, arm/features, arm/resources, and arm/subscriptions packages under a new, resources, package (to reflect the corresponding Swagger structure)
- Added a new arm/authoriation (2015-07-01) package
- Added a new arm/cdn (2015-06-01) package
- Added a new arm/intune (2015-01-14-preview) package
- Udated arm/network (2015-06-01)
- Added a new arm/notificationhubs (2014-09-01) package
- Updated arm/scheduler (2016-01-01) package


-----

## `v0.3.0-beta`

- Corrected unintentional struct field renaming and client renaming in v0.2.0-beta

-----

## `v0.2.0-beta`

- Added support for DNS, Redis, and Web site services
- Updated Storage service to API version 2015-06-15
- Updated Network to include routing table support
- Address https://github.com/Azure/azure-sdk-for-go/issues/232
- Address https://github.com/Azure/azure-sdk-for-go/issues/231
- Address https://github.com/Azure/azure-sdk-for-go/issues/230
- Address https://github.com/Azure/azure-sdk-for-go/issues/224
- Address https://github.com/Azure/azure-sdk-for-go/issues/184
- Address https://github.com/Azure/azure-sdk-for-go/issues/183

------

## `v0.1.1-beta`

- Improves the UserAgent string to disambiguate arm packages from others in the SDK
- Improves setting the http.Response into generated results (reduces likelihood of a nil reference)
- Adds gofmt, golint, and govet to Travis CI for the arm packages

##### Fixed Issues

- https://github.com/Azure/azure-sdk-for-go/issues/196
- https://github.com/Azure/azure-sdk-for-go/issues/213

------

## v0.1.0-beta

This release addresses the issues raised against the alpha release and adds more features. Most
notably, to address the challenges of encoding JSON
(see the [comments](https://github.com/Azure/go-autorest#handling-empty-values) in the
[go-autorest](https://github.com/Azure/go-autorest) package) by using pointers for *all* structure
fields (with the exception of enumerations). The
[go-autorest/autorest/to](https://github.com/Azure/go-autorest/tree/master/autorest/to) package
provides helpers to convert to / from pointers. The examples demonstrate their usage.

Additionally, the packages now align with Go coding standards and pass both `golint` and `govet`.
Accomplishing this required renaming various fields and parameters (such as changing Url to URL).

##### Changes

- Changed request / response structures to use pointer fields.
- Changed methods to return `error` instead of `autorest.Error`.
- Re-divided methods to ease asynchronous requests.
- Added paged results support.
- Added a UserAgent string.
- Added changes necessary to pass golint and govet.
- Updated README.md with details on asynchronous requests and paging.
- Saved package dependencies through Godep (for the entire SDK).

##### Fixed Issues:

- https://github.com/Azure/azure-sdk-for-go/issues/205
- https://github.com/Azure/azure-sdk-for-go/issues/206
- https://github.com/Azure/azure-sdk-for-go/issues/211
- https://github.com/Azure/azure-sdk-for-go/issues/212

-----

## v0.1.0-alpha

This release introduces the Azure Resource Manager packages generated from the corresponding
[Swagger API](http://swagger.io) [definitions](https://github.com/Azure/azure-rest-api-specs).