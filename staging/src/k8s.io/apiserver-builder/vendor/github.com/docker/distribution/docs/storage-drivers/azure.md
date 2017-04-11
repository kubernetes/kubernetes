<!--[metadata]>
+++
title = "Microsoft Azure storage driver"
description = "Explains how to use the Azure storage drivers"
keywords = ["registry, service, driver, images, storage,  azure"]
+++
<![end-metadata]-->


# Microsoft Azure storage driver

An implementation of the `storagedriver.StorageDriver` interface which uses [Microsoft Azure Blob Storage](http://azure.microsoft.com/en-us/services/storage/) for object storage.

## Parameters

<table>
  <tr>
    <th>Parameter</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>
      <code>accountname</code>
    </td>
    <td>
      yes
    </td>
    <td>
      Name of the Azure Storage Account.
    </td>
  </tr>
  <tr>
    <td>
      <code>accountkey</code>
    </td>
    <td>
      yes
    </td>
    <td>
      Primary or Secondary Key for the Storage Account.
    </td>
  </tr>
  <tr>
    <td>
      <code>container</code>
    </td>
    <td>
      yes
    </td>
    <td>
      Name of the Azure root storage container in which all registry data will be stored. Must comply the storage container name [requirements][create-container-api].
    </td>
  </tr>
   <tr>
    <td>
      <code>realm</code>
    </td>
    <td>
      no
    </td>
    <td>
      Domain name suffix for the Storage Service API endpoint. For example realm for "Azure in China" would be `core.chinacloudapi.cn` and realm for "Azure Government" would be `core.usgovcloudapi.net`. By default, this
      is <code>core.windows.net</code>.
    </td>
  </tr>

</table>


## Related Information

* To get information about
[azure-blob-storage](http://azure.microsoft.com/en-us/services/storage/) visit
the Microsoft website.
* You can use Microsoft's [Blob Service REST API](https://msdn.microsoft.com/en-us/library/azure/dd135733.aspx) to [create a container] (https://msdn.microsoft.com/en-us/library/azure/dd179468.aspx).
