# CHANGELOG

## `v55.0.0`

NOTE: Package `github.com/Azure/azure-sdk-for-go/services/virtualmachineimagebuilder/mgmt/2020-02-01/virtualmachineimagebuilder` renamed to `github.com/Azure/azure-sdk-for-go/services/virtualmachineimagebuilder/mgmt/2020-02-14/virtualmachineimagebuilder`

This major version enroll the code generator fix for [this issue](https://github.com/Azure/azure-sdk-for-go/issues/14478), with a side effect that after this version, if a struct only has properties that are marked as `READ-ONLY` in the comment, their `MarshalJSON` function will give you an empty JSON string.

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/mgmt/2021-04-30/cognitiveservices`
- `github.com/Azure/azure-sdk-for-go/services/preview/logz/mgmt/2020-10-01-preview/logz`
- `github.com/Azure/azure-sdk-for-go/services/preview/storagepool/mgmt/2021-04-01-preview/storagepool`
- `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2021-01-01/backup`
- `github.com/Azure/azure-sdk-for-go/services/virtualmachineimagebuilder/mgmt/2020-02-14/virtualmachineimagebuilder`

### Updated Packages

| Package Path | Changelog |
| :--- | :---: |
| `github.com/Azure/azure-sdk-for-go/services/aad/mgmt/2017-04-01/aad` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/aad/mgmt/2017-04-01/aad/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/advisor/mgmt/2017-03-31/advisor` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/advisor/mgmt/2017-03-31/advisor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/advisor/mgmt/2017-04-19/advisor` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/advisor/mgmt/2017-04-19/advisor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/advisor/mgmt/2020-01-01/advisor` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/advisor/mgmt/2020-01-01/advisor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/alertsmanagement/mgmt/2018-05-05/alertsmanagement` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/alertsmanagement/mgmt/2018-05-05/alertsmanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/alertsmanagement/mgmt/2019-03-01/alertsmanagement` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/alertsmanagement/mgmt/2019-03-01/alertsmanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/analysisservices/mgmt/2017-08-01/analysisservices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/analysisservices/mgmt/2017-08-01/analysisservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2016-07-07/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/apimanagement/mgmt/2016-07-07/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2016-10-10/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/apimanagement/mgmt/2016-10-10/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2017-03-01/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/apimanagement/mgmt/2017-03-01/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2018-01-01/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/apimanagement/mgmt/2018-01-01/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2019-01-01/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/apimanagement/mgmt/2019-01-01/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2019-12-01/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/apimanagement/mgmt/2019-12-01/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2020-12-01/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/apimanagement/mgmt/2020-12-01/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/appconfiguration/mgmt/2019-10-01/appconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/appconfiguration/mgmt/2019-10-01/appconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/appconfiguration/mgmt/2020-06-01/appconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/appconfiguration/mgmt/2020-06-01/appconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/appinsights/mgmt/2015-05-01/insights` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/appinsights/mgmt/2015-05-01/insights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/appplatform/mgmt/2020-07-01/appplatform` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/appplatform/mgmt/2020-07-01/appplatform/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/attestation/mgmt/2018-09-01/attestation` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/attestation/mgmt/2018-09-01/attestation/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/attestation/mgmt/2020-10-01/attestation` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/attestation/mgmt/2020-10-01/attestation/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/automation/mgmt/2015-10-31/automation` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/automation/mgmt/2015-10-31/automation/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/avs/mgmt/2020-03-20/avs` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/avs/mgmt/2020-03-20/avs/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/azurestack/mgmt/2017-06-01/azurestack` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/azurestack/mgmt/2017-06-01/azurestack/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/azurestackhci/mgmt/2020-10-01/azurestackhci` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/azurestackhci/mgmt/2020-10-01/azurestackhci/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2017-01-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/batch/mgmt/2017-01-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2017-05-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/batch/mgmt/2017-05-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2017-09-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/batch/mgmt/2017-09-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2018-12-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/batch/mgmt/2018-12-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2019-04-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/batch/mgmt/2019-04-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2019-08-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/batch/mgmt/2019-08-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2020-03-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/batch/mgmt/2020-03-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2020-05-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/batch/mgmt/2020-05-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2020-09-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/batch/mgmt/2020-09-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batchai/mgmt/2018-05-01/batchai` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/batchai/mgmt/2018-05-01/batchai/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2015-06-01/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cdn/mgmt/2015-06-01/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2016-04-02/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cdn/mgmt/2016-04-02/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2016-10-02/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cdn/mgmt/2016-10-02/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2017-04-02/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cdn/mgmt/2017-04-02/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2017-10-12/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cdn/mgmt/2017-10-12/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2019-04-15/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cdn/mgmt/2019-04-15/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2019-06-15/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cdn/mgmt/2019-06-15/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2020-04-15/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cdn/mgmt/2020-04-15/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2020-09-01/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cdn/mgmt/2020-09-01/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/changeanalysis/mgmt/2021-04-01/changeanalysis` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/changeanalysis/mgmt/2021-04-01/changeanalysis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/mgmt/2017-04-18/cognitiveservices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cognitiveservices/mgmt/2017-04-18/cognitiveservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v1.0/customsearch` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cognitiveservices/v1.0/customsearch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v1.0/imagesearch` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cognitiveservices/v1.0/imagesearch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v1.0/videosearch` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cognitiveservices/v1.0/videosearch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v1.0/websearch` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cognitiveservices/v1.0/websearch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v1.1/customvision/prediction` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cognitiveservices/v1.1/customvision/prediction/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v1.2/customvision/training` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cognitiveservices/v1.2/customvision/training/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v2.0/computervision` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cognitiveservices/v2.0/computervision/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v2.0/textanalytics` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cognitiveservices/v2.0/textanalytics/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v2.1/computervision` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cognitiveservices/v2.1/computervision/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v2.1/customvision/training` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cognitiveservices/v2.1/customvision/training/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v2.1/textanalytics` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cognitiveservices/v2.1/textanalytics/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v2.2/customvision/training` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cognitiveservices/v2.2/customvision/training/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v3.0/computervision` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cognitiveservices/v3.0/computervision/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v3.0/customvision/prediction` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cognitiveservices/v3.0/customvision/prediction/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v3.0/customvision/training` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cognitiveservices/v3.0/customvision/training/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v3.1/computervision` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cognitiveservices/v3.1/computervision/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v3.1/customvision/prediction` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cognitiveservices/v3.1/customvision/prediction/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v3.1/customvision/training` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cognitiveservices/v3.1/customvision/training/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v3.2/customvision/training` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cognitiveservices/v3.2/customvision/training/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v3.3/customvision/training` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cognitiveservices/v3.3/customvision/training/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/communication/mgmt/2020-08-20/communication` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/communication/mgmt/2020-08-20/communication/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2015-06-15/compute` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/compute/mgmt/2015-06-15/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2016-03-30/compute` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/compute/mgmt/2016-03-30/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2017-03-30/compute` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/compute/mgmt/2017-03-30/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2017-09-01/skus` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/compute/mgmt/2017-09-01/skus/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2017-12-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/compute/mgmt/2017-12-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2018-04-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/compute/mgmt/2018-04-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2018-06-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/compute/mgmt/2018-06-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2018-10-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/compute/mgmt/2018-10-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-03-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/compute/mgmt/2019-03-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-07-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/compute/mgmt/2019-07-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-12-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/compute/mgmt/2019-12-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2020-06-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/compute/mgmt/2020-06-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2020-06-30/compute` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/compute/mgmt/2020-06-30/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2020-12-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/compute/mgmt/2020-12-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/confluent/mgmt/2020-03-01/confluent` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/confluent/mgmt/2020-03-01/confluent/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/consumption/mgmt/2017-11-30/consumption` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/consumption/mgmt/2017-11-30/consumption/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/consumption/mgmt/2018-01-31/consumption` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/consumption/mgmt/2018-01-31/consumption/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/consumption/mgmt/2018-03-31/consumption` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/consumption/mgmt/2018-03-31/consumption/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/consumption/mgmt/2018-05-31/consumption` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/consumption/mgmt/2018-05-31/consumption/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/consumption/mgmt/2018-06-30/consumption` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/consumption/mgmt/2018-06-30/consumption/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/consumption/mgmt/2018-08-31/consumption` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/consumption/mgmt/2018-08-31/consumption/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/consumption/mgmt/2018-10-01/consumption` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/consumption/mgmt/2018-10-01/consumption/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/consumption/mgmt/2019-01-01/consumption` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/consumption/mgmt/2019-01-01/consumption/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/consumption/mgmt/2019-10-01/consumption` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/consumption/mgmt/2019-10-01/consumption/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerinstance/mgmt/2018-04-01/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerinstance/mgmt/2018-04-01/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerinstance/mgmt/2018-06-01/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerinstance/mgmt/2018-06-01/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerinstance/mgmt/2018-09-01/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerinstance/mgmt/2018-09-01/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerinstance/mgmt/2018-10-01/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerinstance/mgmt/2018-10-01/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerinstance/mgmt/2019-12-01/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerinstance/mgmt/2019-12-01/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerinstance/mgmt/2020-11-01/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerinstance/mgmt/2020-11-01/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2017-10-01/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerregistry/mgmt/2017-10-01/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2018-09-01/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerregistry/mgmt/2018-09-01/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2019-04-01/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerregistry/mgmt/2019-04-01/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2019-05-01/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerregistry/mgmt/2019-05-01/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2018-03-31/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerservice/mgmt/2018-03-31/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2019-04-30/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerservice/mgmt/2019-04-30/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2019-06-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerservice/mgmt/2019-06-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2019-08-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerservice/mgmt/2019-08-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2019-10-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerservice/mgmt/2019-10-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2019-11-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerservice/mgmt/2019-11-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-01-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerservice/mgmt/2020-01-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-02-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerservice/mgmt/2020-02-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-03-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerservice/mgmt/2020-03-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-04-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerservice/mgmt/2020-04-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-06-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerservice/mgmt/2020-06-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-07-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerservice/mgmt/2020-07-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-09-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerservice/mgmt/2020-09-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-11-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerservice/mgmt/2020-11-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-12-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerservice/mgmt/2020-12-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2021-02-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerservice/mgmt/2021-02-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2021-03-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/containerservice/mgmt/2021-03-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cosmos-db/mgmt/2015-04-08/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cosmos-db/mgmt/2015-04-08/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cosmos-db/mgmt/2019-08-01/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cosmos-db/mgmt/2019-08-01/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cosmos-db/mgmt/2021-01-15/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cosmos-db/mgmt/2021-01-15/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cosmos-db/mgmt/2021-03-15/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/cosmos-db/mgmt/2021-03-15/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/costmanagement/mgmt/2018-05-31/costmanagement` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/costmanagement/mgmt/2018-05-31/costmanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/costmanagement/mgmt/2019-01-01/costmanagement` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/costmanagement/mgmt/2019-01-01/costmanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/costmanagement/mgmt/2019-10-01/costmanagement` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/costmanagement/mgmt/2019-10-01/costmanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/costmanagement/mgmt/2020-06-01/costmanagement` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/costmanagement/mgmt/2020-06-01/costmanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/customerinsights/mgmt/2017-01-01/customerinsights` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/customerinsights/mgmt/2017-01-01/customerinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/customerinsights/mgmt/2017-04-26/customerinsights` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/customerinsights/mgmt/2017-04-26/customerinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databox/mgmt/2018-01-01/databox` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/databox/mgmt/2018-01-01/databox/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databox/mgmt/2019-09-01/databox` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/databox/mgmt/2019-09-01/databox/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databox/mgmt/2020-04-01/databox` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/databox/mgmt/2020-04-01/databox/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databox/mgmt/2020-11-01/databox` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/databox/mgmt/2020-11-01/databox/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databoxedge/mgmt/2019-03-01/databoxedge` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/databoxedge/mgmt/2019-03-01/databoxedge/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databoxedge/mgmt/2019-07-01/databoxedge` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/databoxedge/mgmt/2019-07-01/databoxedge/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databoxedge/mgmt/2019-08-01/databoxedge` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/databoxedge/mgmt/2019-08-01/databoxedge/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databoxedge/mgmt/2020-12-01/databoxedge` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/databoxedge/mgmt/2020-12-01/databoxedge/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databricks/mgmt/2018-04-01/databricks` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/databricks/mgmt/2018-04-01/databricks/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datalake/analytics/2016-11-01-preview/catalog` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/datalake/analytics/2016-11-01-preview/catalog/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datalake/analytics/2016-11-01/job` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/datalake/analytics/2016-11-01/job/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datalake/analytics/mgmt/2016-11-01/account` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/datalake/analytics/mgmt/2016-11-01/account/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datalake/store/2016-11-01/filesystem` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/datalake/store/2016-11-01/filesystem/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datalake/store/mgmt/2016-11-01/account` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/datalake/store/mgmt/2016-11-01/account/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datamigration/mgmt/2018-04-19/datamigration` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/datamigration/mgmt/2018-04-19/datamigration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/dataprotection/mgmt/2021-01-01/dataprotection` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/dataprotection/mgmt/2021-01-01/dataprotection/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datashare/mgmt/2019-11-01/datashare` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/datashare/mgmt/2019-11-01/datashare/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/devspaces/mgmt/2019-04-01/devspaces` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/devspaces/mgmt/2019-04-01/devspaces/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/devtestlabs/mgmt/2016-05-15/dtl` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/devtestlabs/mgmt/2016-05-15/dtl/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/devtestlabs/mgmt/2018-09-15/dtl` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/devtestlabs/mgmt/2018-09-15/dtl/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/digitaltwins/mgmt/2020-10-31/digitaltwins` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/digitaltwins/mgmt/2020-10-31/digitaltwins/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/digitaltwins/mgmt/2020-12-01/digitaltwins` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/digitaltwins/mgmt/2020-12-01/digitaltwins/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/dns/mgmt/2016-04-01/dns` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/dns/mgmt/2016-04-01/dns/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/dns/mgmt/2017-09-01/dns` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/dns/mgmt/2017-09-01/dns/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/dns/mgmt/2017-10-01/dns` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/dns/mgmt/2017-10-01/dns/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/domainservices/mgmt/2017-01-01/aad` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/domainservices/mgmt/2017-01-01/aad/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/domainservices/mgmt/2017-06-01/aad` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/domainservices/mgmt/2017-06-01/aad/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/domainservices/mgmt/2020-01-01/aad` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/domainservices/mgmt/2020-01-01/aad/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/eventgrid/2018-01-01/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/eventgrid/2018-01-01/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/eventgrid/mgmt/2018-01-01/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/eventgrid/mgmt/2018-01-01/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/eventgrid/mgmt/2019-01-01/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/eventgrid/mgmt/2019-01-01/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/eventgrid/mgmt/2019-06-01/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/eventgrid/mgmt/2019-06-01/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/eventgrid/mgmt/2020-06-01/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/eventgrid/mgmt/2020-06-01/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/eventhub/mgmt/2015-08-01/eventhub` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/eventhub/mgmt/2015-08-01/eventhub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/eventhub/mgmt/2017-04-01/eventhub` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/eventhub/mgmt/2017-04-01/eventhub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2019-04-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/frontdoor/mgmt/2019-04-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2019-05-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/frontdoor/mgmt/2019-05-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2019-10-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/frontdoor/mgmt/2019-10-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2019-11-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/frontdoor/mgmt/2019-11-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2020-01-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/frontdoor/mgmt/2020-01-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2020-04-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/frontdoor/mgmt/2020-04-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2020-05-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/frontdoor/mgmt/2020-05-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/guestconfiguration/mgmt/2020-06-25/guestconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/guestconfiguration/mgmt/2020-06-25/guestconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/hdinsight/mgmt/2018-06-01/hdinsight` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/hdinsight/mgmt/2018-06-01/hdinsight/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/healthbot/mgmt/2020-12-08/healthbot` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/healthbot/mgmt/2020-12-08/healthbot/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/healthcareapis/mgmt/2019-09-16/healthcareapis` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/healthcareapis/mgmt/2019-09-16/healthcareapis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/healthcareapis/mgmt/2020-03-15/healthcareapis` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/healthcareapis/mgmt/2020-03-15/healthcareapis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/healthcareapis/mgmt/2020-03-30/healthcareapis` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/healthcareapis/mgmt/2020-03-30/healthcareapis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/hybridcompute/mgmt/2019-12-12/hybridcompute` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/hybridcompute/mgmt/2019-12-12/hybridcompute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/hybridcompute/mgmt/2020-08-02/hybridcompute` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/hybridcompute/mgmt/2020-08-02/hybridcompute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/hybriddatamanager/mgmt/2016-06-01/hybriddata` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/hybriddatamanager/mgmt/2016-06-01/hybriddata/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/hybriddatamanager/mgmt/2019-06-01/hybriddata` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/hybriddatamanager/mgmt/2019-06-01/hybriddata/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/hybridkubernetes/mgmt/2021-03-01/hybridkubernetes` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/hybridkubernetes/mgmt/2021-03-01/hybridkubernetes/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iotcentral/mgmt/2018-09-01/iotcentral` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/iotcentral/mgmt/2018-09-01/iotcentral/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iothub/mgmt/2016-02-03/devices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/iothub/mgmt/2016-02-03/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iothub/mgmt/2017-01-19/devices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/iothub/mgmt/2017-01-19/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iothub/mgmt/2017-07-01/devices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/iothub/mgmt/2017-07-01/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iothub/mgmt/2018-01-22/devices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/iothub/mgmt/2018-01-22/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iothub/mgmt/2018-04-01/devices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/iothub/mgmt/2018-04-01/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iothub/mgmt/2020-03-01/devices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/iothub/mgmt/2020-03-01/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/keyvault/2015-06-01/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/keyvault/2015-06-01/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/keyvault/2016-10-01/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/keyvault/2016-10-01/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/keyvault/mgmt/2016-10-01/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/keyvault/mgmt/2016-10-01/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/keyvault/mgmt/2018-02-14/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/keyvault/mgmt/2018-02-14/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/keyvault/mgmt/2019-09-01/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/keyvault/mgmt/2019-09-01/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/keyvault/v7.0/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/keyvault/v7.0/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/keyvault/v7.1/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/keyvault/v7.1/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kubernetesconfiguration/mgmt/2021-03-01/kubernetesconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/kubernetesconfiguration/mgmt/2021-03-01/kubernetesconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2019-01-21/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/kusto/mgmt/2019-01-21/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2019-05-15/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/kusto/mgmt/2019-05-15/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2019-09-07/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/kusto/mgmt/2019-09-07/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2019-11-09/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/kusto/mgmt/2019-11-09/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2020-02-15/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/kusto/mgmt/2020-02-15/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2020-06-14/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/kusto/mgmt/2020-06-14/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2020-09-18/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/kusto/mgmt/2020-09-18/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/labservices/mgmt/2018-10-15/labservices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/labservices/mgmt/2018-10-15/labservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/logic/mgmt/2016-06-01/logic` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/logic/mgmt/2016-06-01/logic/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/logic/mgmt/2019-05-01/logic` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/logic/mgmt/2019-05-01/logic/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearning/mgmt/2017-01-01/webservices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/machinelearning/mgmt/2017-01-01/webservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2018-11-19/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/machinelearningservices/mgmt/2018-11-19/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2019-05-01/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/machinelearningservices/mgmt/2019-05-01/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2019-06-01/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/machinelearningservices/mgmt/2019-06-01/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2019-11-01/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/machinelearningservices/mgmt/2019-11-01/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2020-01-01/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/machinelearningservices/mgmt/2020-01-01/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2020-03-01/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/machinelearningservices/mgmt/2020-03-01/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2020-04-01/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/machinelearningservices/mgmt/2020-04-01/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/maintenance/mgmt/2020-04-01/maintenance` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/maintenance/mgmt/2020-04-01/maintenance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/maintenance/mgmt/2021-05-01/maintenance` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/maintenance/mgmt/2021-05-01/maintenance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/managedservices/mgmt/2019-06-01/managedservices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/managedservices/mgmt/2019-06-01/managedservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/maps/mgmt/2017-01-01-preview/maps` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/maps/mgmt/2017-01-01-preview/maps/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/maps/mgmt/2018-05-01/maps` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/maps/mgmt/2018-05-01/maps/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/maps/mgmt/2021-02-01/maps` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/maps/mgmt/2021-02-01/maps/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mariadb/mgmt/2018-06-01/mariadb` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/mariadb/mgmt/2018-06-01/mariadb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mariadb/mgmt/2020-01-01/mariadb` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/mariadb/mgmt/2020-01-01/mariadb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/marketplaceordering/mgmt/2015-06-01/marketplaceordering` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/marketplaceordering/mgmt/2015-06-01/marketplaceordering/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mediaservices/mgmt/2015-10-01/media` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/mediaservices/mgmt/2015-10-01/media/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mediaservices/mgmt/2018-07-01/media` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/mediaservices/mgmt/2018-07-01/media/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mediaservices/mgmt/2020-05-01/media` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/mediaservices/mgmt/2020-05-01/media/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/migrate/mgmt/2018-02-02/migrate` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/migrate/mgmt/2018-02-02/migrate/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/migrate/mgmt/2020-01-01/migrate` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/migrate/mgmt/2020-01-01/migrate/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mixedreality/mgmt/2021-01-01/mixedreality` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/mixedreality/mgmt/2021-01-01/mixedreality/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/monitor/mgmt/2020-10-01/insights` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/monitor/mgmt/2020-10-01/insights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/msi/mgmt/2018-11-30/msi` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/msi/mgmt/2018-11-30/msi/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mysql/mgmt/2017-12-01/mysql` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/mysql/mgmt/2017-12-01/mysql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mysql/mgmt/2020-01-01/mysql` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/mysql/mgmt/2020-01-01/mysql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-02-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/netapp/mgmt/2020-02-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-03-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/netapp/mgmt/2020-03-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-05-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/netapp/mgmt/2020-05-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-06-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/netapp/mgmt/2020-06-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-07-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/netapp/mgmt/2020-07-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-08-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/netapp/mgmt/2020-08-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-09-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/netapp/mgmt/2020-09-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-11-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/netapp/mgmt/2020-11-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-12-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/netapp/mgmt/2020-12-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2016-09-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2016-09-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2016-12-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2016-12-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-03-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2017-03-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-06-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2017-06-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-08-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2017-08-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-09-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2017-09-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-10-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2017-10-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-11-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2017-11-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-01-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2018-01-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-02-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2018-02-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-04-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2018-04-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-06-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2018-06-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-07-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2018-07-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-08-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2018-08-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-10-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2018-10-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-11-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2018-11-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-12-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2018-12-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-02-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2019-02-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-04-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2019-04-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2019-06-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-07-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2019-07-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-08-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2019-08-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-09-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2019-09-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-11-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2019-11-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-12-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2019-12-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-03-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2020-03-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-04-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2020-04-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-05-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2020-05-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-06-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2020-06-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-07-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2020-07-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-08-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2020-08-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/notificationhubs/mgmt/2017-04-01/notificationhubs` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/notificationhubs/mgmt/2017-04-01/notificationhubs/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/operationalinsights/mgmt/2015-03-20/operationalinsights` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/operationalinsights/mgmt/2015-03-20/operationalinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/operationalinsights/mgmt/2020-08-01/operationalinsights` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/operationalinsights/mgmt/2020-08-01/operationalinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/operationalinsights/mgmt/2020-10-01/operationalinsights` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/operationalinsights/mgmt/2020-10-01/operationalinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/peering/mgmt/2020-04-01/peering` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/peering/mgmt/2020-04-01/peering/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/peering/mgmt/2020-10-01/peering` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/peering/mgmt/2020-10-01/peering/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/personalizer/v1.0/personalizer` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/personalizer/v1.0/personalizer/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/policyinsights/mgmt/2018-04-04/policyinsights` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/policyinsights/mgmt/2018-04-04/policyinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/postgresql/mgmt/2017-12-01/postgresql` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/postgresql/mgmt/2017-12-01/postgresql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/postgresql/mgmt/2020-01-01/postgresql` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/postgresql/mgmt/2020-01-01/postgresql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/powerbidedicated/mgmt/2017-10-01/powerbidedicated` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/powerbidedicated/mgmt/2017-10-01/powerbidedicated/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/alertsmanagement/mgmt/2018-05-05-preview/alertsmanagement` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/alertsmanagement/mgmt/2018-05-05-preview/alertsmanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/alertsmanagement/mgmt/2019-05-05-preview/alertsmanagement` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/alertsmanagement/mgmt/2019-05-05-preview/alertsmanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/alertsmanagement/mgmt/2019-06-01-preview/alertsmanagement` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/alertsmanagement/mgmt/2019-06-01-preview/alertsmanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/apimanagement/ctrl/2017-03-01/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/apimanagement/ctrl/2017-03-01/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/apimanagement/mgmt/2018-06-01-preview/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/apimanagement/mgmt/2018-06-01-preview/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/apimanagement/mgmt/2019-12-01-preview/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/apimanagement/mgmt/2019-12-01-preview/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/appconfiguration/mgmt/2019-02-01-preview/appconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/appconfiguration/mgmt/2019-02-01-preview/appconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/appconfiguration/mgmt/2019-11-01-preview/appconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/appconfiguration/mgmt/2019-11-01-preview/appconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/appconfiguration/mgmt/2020-07-01-preview/appconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/appconfiguration/mgmt/2020-07-01-preview/appconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/appplatform/mgmt/2019-05-01-preview/appplatform` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/appplatform/mgmt/2019-05-01-preview/appplatform/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/appplatform/mgmt/2020-11-01-preview/appplatform` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/appplatform/mgmt/2020-11-01-preview/appplatform/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/authorization/mgmt/2018-07-01-preview/authorization` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/authorization/mgmt/2018-07-01-preview/authorization/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/authorization/mgmt/2018-09-01-preview/authorization` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/authorization/mgmt/2018-09-01-preview/authorization/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/automanage/mgmt/2020-06-30-preview/automanage` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/automanage/mgmt/2020-06-30-preview/automanage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/automation/mgmt/2017-05-15-preview/automation` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/automation/mgmt/2017-05-15-preview/automation/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/automation/mgmt/2018-01-15-preview/automation` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/automation/mgmt/2018-01-15-preview/automation/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/automation/mgmt/2018-06-30-preview/automation` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/automation/mgmt/2018-06-30-preview/automation/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/avs/mgmt/2020-07-17-preview/avs` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/avs/mgmt/2020-07-17-preview/avs/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/azureadb2c/mgmt/2020-05-01-preview/azureadb2c` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/azureadb2c/mgmt/2020-05-01-preview/azureadb2c/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/azuredata/mgmt/2017-03-01-preview/azuredata` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/azuredata/mgmt/2017-03-01-preview/azuredata/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/azuredata/mgmt/2019-07-24-preview/azuredata` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/azuredata/mgmt/2019-07-24-preview/azuredata/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/azurestackhci/mgmt/2020-03-01-preview/azurestackhci` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/azurestackhci/mgmt/2020-03-01-preview/azurestackhci/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/baremetalinfrastructure/mgmt/2020-08-06-preview/baremetalinfrastructure` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/baremetalinfrastructure/mgmt/2020-08-06-preview/baremetalinfrastructure/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/billing/mgmt/2017-02-27-preview/billing` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/billing/mgmt/2017-02-27-preview/billing/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/billing/mgmt/2017-04-24-preview/billing` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/billing/mgmt/2017-04-24-preview/billing/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/billing/mgmt/2018-03-01-preview/billing` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/billing/mgmt/2018-03-01-preview/billing/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/billing/mgmt/2018-11-01-preview/billing` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/billing/mgmt/2018-11-01-preview/billing/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/billing/mgmt/2020-05-01-preview/billing` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/billing/mgmt/2020-05-01-preview/billing/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/blockchain/mgmt/2018-06-01-preview/blockchain` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/blockchain/mgmt/2018-06-01-preview/blockchain/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/blueprint/mgmt/2018-11-01-preview/blueprint` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/blueprint/mgmt/2018-11-01-preview/blueprint/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/botservice/mgmt/2017-12-01/botservice` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/botservice/mgmt/2017-12-01/botservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/botservice/mgmt/2018-07-12/botservice` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/botservice/mgmt/2018-07-12/botservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cognitiveservices/mgmt/2016-02-01-preview/cognitiveservices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/cognitiveservices/mgmt/2016-02-01-preview/cognitiveservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cognitiveservices/v1.0/visualsearch` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/cognitiveservices/v1.0/visualsearch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cognitiveservices/v3.4-preview/customvision/training` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/cognitiveservices/v3.4-preview/customvision/training/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/communication/mgmt/2020-08-20-preview/communication` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/communication/mgmt/2020-08-20-preview/communication/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/compute/mgmt/2016-04-30-preview/compute` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/compute/mgmt/2016-04-30-preview/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/compute/mgmt/2020-10-01-preview/compute` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/compute/mgmt/2020-10-01-preview/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/confidentialledger/mgmt/2020-12-01-preview/confidentialledger` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/confidentialledger/mgmt/2020-12-01-preview/confidentialledger/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/confluent/mgmt/2020-03-01-preview/confluent` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/confluent/mgmt/2020-03-01-preview/confluent/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/confluent/mgmt/2021-03-01-preview/confluent` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/confluent/mgmt/2021-03-01-preview/confluent/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/consumption/mgmt/2017-04-24-preview/consumption` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/consumption/mgmt/2017-04-24-preview/consumption/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/consumption/mgmt/2017-12-30-preview/consumption` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/consumption/mgmt/2017-12-30-preview/consumption/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerinstance/mgmt/2017-10-01-preview/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/containerinstance/mgmt/2017-10-01-preview/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerinstance/mgmt/2017-12-01-preview/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/containerinstance/mgmt/2017-12-01-preview/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerinstance/mgmt/2018-02-01-preview/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/containerinstance/mgmt/2018-02-01-preview/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerregistry/mgmt/2017-06-01-preview/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/containerregistry/mgmt/2017-06-01-preview/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerregistry/mgmt/2018-02-01/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/containerregistry/mgmt/2018-02-01/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerregistry/mgmt/2019-06-01-preview/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/containerregistry/mgmt/2019-06-01-preview/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerregistry/mgmt/2019-12-01-preview/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/containerregistry/mgmt/2019-12-01-preview/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerregistry/mgmt/2020-11-01-preview/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/containerregistry/mgmt/2020-11-01-preview/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerservice/mgmt/2018-08-01-preview/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/containerservice/mgmt/2018-08-01-preview/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerservice/mgmt/2018-09-30-preview/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/containerservice/mgmt/2018-09-30-preview/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerservice/mgmt/2019-02-01-preview/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/containerservice/mgmt/2019-02-01-preview/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerservice/mgmt/2019-09-30-preview/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/containerservice/mgmt/2019-09-30-preview/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerservice/mgmt/2019-10-27-preview/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/containerservice/mgmt/2019-10-27-preview/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2019-08-01-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/cosmos-db/mgmt/2019-08-01-preview/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2019-12-12-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/cosmos-db/mgmt/2019-12-12-preview/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2020-03-01-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/cosmos-db/mgmt/2020-03-01-preview/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2020-04-01-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/cosmos-db/mgmt/2020-04-01-preview/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2020-06-01-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/cosmos-db/mgmt/2020-06-01-preview/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2020-09-01-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/cosmos-db/mgmt/2020-09-01-preview/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2021-03-01-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/cosmos-db/mgmt/2021-03-01-preview/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/costmanagement/mgmt/2018-08-01-preview/costmanagement` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/costmanagement/mgmt/2018-08-01-preview/costmanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/costmanagement/mgmt/2019-03-01/costmanagement` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/costmanagement/mgmt/2019-03-01/costmanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/customerlockbox/mgmt/2018-02-28-preview/customerlockbox` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/customerlockbox/mgmt/2018-02-28-preview/customerlockbox/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/customproviders/mgmt/2018-09-01-preview/customproviders` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/customproviders/mgmt/2018-09-01-preview/customproviders/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/databoxedge/mgmt/2020-05-01-preview/databoxedge` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/databoxedge/mgmt/2020-05-01-preview/databoxedge/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datafactory/mgmt/2017-09-01-preview/datafactory` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/datafactory/mgmt/2017-09-01-preview/datafactory/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datalake/analytics/2015-11-01-preview/job` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/datalake/analytics/2015-11-01-preview/job/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datalake/analytics/2016-03-20-preview/job` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/datalake/analytics/2016-03-20-preview/job/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datalake/analytics/2017-09-01-preview/job` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/datalake/analytics/2017-09-01-preview/job/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datalake/analytics/mgmt/2015-10-01-preview/account` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/datalake/analytics/mgmt/2015-10-01-preview/account/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datalake/store/2015-10-01-preview/filesystem` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/datalake/store/2015-10-01-preview/filesystem/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datalake/store/mgmt/2015-10-01-preview/account` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/datalake/store/mgmt/2015-10-01-preview/account/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datamigration/mgmt/2017-11-15-preview/datamigration` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/datamigration/mgmt/2017-11-15-preview/datamigration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datamigration/mgmt/2018-03-31-preview/datamigration` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/datamigration/mgmt/2018-03-31-preview/datamigration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datamigration/mgmt/2018-07-15-preview/datamigration` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/datamigration/mgmt/2018-07-15-preview/datamigration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datashare/mgmt/2018-11-01-preview/datashare` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/datashare/mgmt/2018-11-01-preview/datashare/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/delegatednetwork/mgmt/2020-08-08-preview/delegatednetwork` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/delegatednetwork/mgmt/2020-08-08-preview/delegatednetwork/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/deploymentmanager/mgmt/2018-09-01-preview/deploymentmanager` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/deploymentmanager/mgmt/2018-09-01-preview/deploymentmanager/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/deploymentmanager/mgmt/2019-11-01-preview/deploymentmanager` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/deploymentmanager/mgmt/2019-11-01-preview/deploymentmanager/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/desktopvirtualization/mgmt/2019-01-23-preview/desktopvirtualization` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/desktopvirtualization/mgmt/2019-01-23-preview/desktopvirtualization/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/desktopvirtualization/mgmt/2019-09-24-preview/desktopvirtualization` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/desktopvirtualization/mgmt/2019-09-24-preview/desktopvirtualization/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/desktopvirtualization/mgmt/2019-12-10-preview/desktopvirtualization` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/desktopvirtualization/mgmt/2019-12-10-preview/desktopvirtualization/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/desktopvirtualization/mgmt/2020-09-21-preview/desktopvirtualization` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/desktopvirtualization/mgmt/2020-09-21-preview/desktopvirtualization/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/desktopvirtualization/mgmt/2020-10-19-preview/desktopvirtualization` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/desktopvirtualization/mgmt/2020-10-19-preview/desktopvirtualization/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/desktopvirtualization/mgmt/2020-11-02-preview/desktopvirtualization` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/desktopvirtualization/mgmt/2020-11-02-preview/desktopvirtualization/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/deviceupdate/mgmt/2020-03-01-preview/deviceupdate` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/deviceupdate/mgmt/2020-03-01-preview/deviceupdate/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/devops/mgmt/2019-07-01-preview/devops` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/devops/mgmt/2019-07-01-preview/devops/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/digitaltwins/mgmt/2020-03-01-preview/digitaltwins` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/digitaltwins/mgmt/2020-03-01-preview/digitaltwins/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/dns/mgmt/2015-05-04-preview/dns` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/dns/mgmt/2015-05-04-preview/dns/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/dns/mgmt/2018-03-01-preview/dns` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/dns/mgmt/2018-03-01-preview/dns/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/elastic/mgmt/2020-07-01-preview/elastic` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/elastic/mgmt/2020-07-01-preview/elastic/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/engagementfabric/mgmt/2018-09-01/engagementfabric` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/engagementfabric/mgmt/2018-09-01/engagementfabric/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2017-06-15-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/eventgrid/mgmt/2017-06-15-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2017-09-15-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/eventgrid/mgmt/2017-09-15-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2018-05-01-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/eventgrid/mgmt/2018-05-01-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2018-09-15-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/eventgrid/mgmt/2018-09-15-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2019-02-01-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/eventgrid/mgmt/2019-02-01-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2020-01-01-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/eventgrid/mgmt/2020-01-01-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2020-04-01-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/eventgrid/mgmt/2020-04-01-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2020-10-15-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/eventgrid/mgmt/2020-10-15-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventhub/mgmt/2018-01-01-preview/eventhub` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/eventhub/mgmt/2018-01-01-preview/eventhub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/extendedlocation/mgmt/2021-03-15-preview/extendedlocation` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/extendedlocation/mgmt/2021-03-15-preview/extendedlocation/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/frontdoor/mgmt/2018-08-01-preview/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/frontdoor/mgmt/2018-08-01-preview/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/hanaonazure/mgmt/2017-11-03-preview/hanaonazure` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/hanaonazure/mgmt/2017-11-03-preview/hanaonazure/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/hardwaresecuritymodules/mgmt/2018-10-31-preview/hardwaresecuritymodules` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/hardwaresecuritymodules/mgmt/2018-10-31-preview/hardwaresecuritymodules/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/hdinsight/mgmt/2015-03-01-preview/hdinsight` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/hdinsight/mgmt/2015-03-01-preview/hdinsight/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/healthcareapis/mgmt/2018-08-20-preview/healthcareapis` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/healthcareapis/mgmt/2018-08-20-preview/healthcareapis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/hybridkubernetes/mgmt/2020-01-01-preview/hybridkubernetes` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/hybridkubernetes/mgmt/2020-01-01-preview/hybridkubernetes/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/hybridnetwork/mgmt/2020-01-01-preview/hybridnetwork` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/hybridnetwork/mgmt/2020-01-01-preview/hybridnetwork/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/iothub/mgmt/2018-12-01-preview/devices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/iothub/mgmt/2018-12-01-preview/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/iothub/mgmt/2019-03-22-preview/devices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/iothub/mgmt/2019-03-22-preview/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/iothub/mgmt/2019-07-01-preview/devices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/iothub/mgmt/2019-07-01-preview/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/iothub/mgmt/2021-03-03-preview/devices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/iothub/mgmt/2021-03-03-preview/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/iotspaces/mgmt/2017-10-01-preview/iotspaces` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/iotspaces/mgmt/2017-10-01-preview/iotspaces/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/keyvault/mgmt/2020-04-01-preview/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/keyvault/mgmt/2020-04-01-preview/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/keyvault/v7.2-preview/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/keyvault/v7.2-preview/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/kubernetesconfiguration/mgmt/2019-11-01-preview/kubernetesconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/kubernetesconfiguration/mgmt/2019-11-01-preview/kubernetesconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/kubernetesconfiguration/mgmt/2020-07-01-preview/kubernetesconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/kubernetesconfiguration/mgmt/2020-07-01-preview/kubernetesconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/kusto/mgmt/2018-09-07-preview/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/kusto/mgmt/2018-09-07-preview/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/logic/mgmt/2015-02-01-preview/logic` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/logic/mgmt/2015-02-01-preview/logic/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/logic/mgmt/2018-07-01-preview/logic` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/logic/mgmt/2018-07-01-preview/logic/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/machinelearning/mgmt/2016-05-01-preview/commitmentplans` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/machinelearning/mgmt/2016-05-01-preview/commitmentplans/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/machinelearning/mgmt/2017-08-01-preview/compute` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/machinelearning/mgmt/2017-08-01-preview/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/machinelearningservices/mgmt/2018-03-01-preview/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/machinelearningservices/mgmt/2018-03-01-preview/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/machinelearningservices/mgmt/2020-02-18-preview/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/machinelearningservices/mgmt/2020-02-18-preview/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/maintenance/mgmt/2018-06-01-preview/maintenance` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/maintenance/mgmt/2018-06-01-preview/maintenance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/managednetwork/mgmt/2019-06-01-preview/managednetwork` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/managednetwork/mgmt/2019-06-01-preview/managednetwork/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/managedservices/mgmt/2018-06-01/managedservices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/managedservices/mgmt/2018-06-01/managedservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/managedservices/mgmt/2019-04-01/managedservices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/managedservices/mgmt/2019-04-01/managedservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/maps/mgmt/2020-02-01-preview/maps` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/maps/mgmt/2020-02-01-preview/maps/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mediaservices/mgmt/2018-03-30-preview/media` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/mediaservices/mgmt/2018-03-30-preview/media/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mediaservices/mgmt/2018-06-01-preview/media` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/mediaservices/mgmt/2018-06-01-preview/media/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mediaservices/mgmt/2019-05-01-preview/media` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/mediaservices/mgmt/2019-05-01-preview/media/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/migrate/mgmt/2018-09-01-preview/migrate` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/migrate/mgmt/2018-09-01-preview/migrate/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mixedreality/mgmt/2019-02-28/mixedreality` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/mixedreality/mgmt/2019-02-28/mixedreality/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mixedreality/mgmt/2020-05-01-preview/mixedreality` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/mixedreality/mgmt/2020-05-01-preview/mixedreality/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mixedreality/mgmt/2021-03-01-preview/mixedreality` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/mixedreality/mgmt/2021-03-01-preview/mixedreality/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/monitor/mgmt/2017-05-01-preview/insights` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/monitor/mgmt/2017-05-01-preview/insights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/monitor/mgmt/2018-03-01/insights` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/monitor/mgmt/2018-03-01/insights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/monitor/mgmt/2018-09-01/insights` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/monitor/mgmt/2018-09-01/insights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/monitor/mgmt/2018-11-01-preview/insights` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/monitor/mgmt/2018-11-01-preview/insights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/monitor/mgmt/2019-03-01/insights` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/monitor/mgmt/2019-03-01/insights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/monitor/mgmt/2019-06-01/insights` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/monitor/mgmt/2019-06-01/insights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/monitor/mgmt/2019-11-01-preview/insights` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/monitor/mgmt/2019-11-01-preview/insights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/msi/mgmt/2015-08-31-preview/msi` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/msi/mgmt/2015-08-31-preview/msi/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mysql/mgmt/2017-12-01-preview/mysql` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/mysql/mgmt/2017-12-01-preview/mysql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mysql/mgmt/2020-07-01-preview/mysqlflexibleservers` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/mysql/mgmt/2020-07-01-preview/mysqlflexibleservers/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/operationalinsights/mgmt/2015-11-01-preview/servicemap` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/operationalinsights/mgmt/2015-11-01-preview/servicemap/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/operationalinsights/mgmt/2020-03-01-preview/operationalinsights` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/operationalinsights/mgmt/2020-03-01-preview/operationalinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/peering/mgmt/2019-08-01-preview/peering` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/peering/mgmt/2019-08-01-preview/peering/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/peering/mgmt/2019-09-01-preview/peering` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/peering/mgmt/2019-09-01-preview/peering/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/peering/mgmt/2020-01-01-preview/peering` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/peering/mgmt/2020-01-01-preview/peering/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/policyinsights/mgmt/2018-07-01-preview/policyinsights` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/policyinsights/mgmt/2018-07-01-preview/policyinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/policyinsights/mgmt/2019-10-01-preview/policyinsights` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/policyinsights/mgmt/2019-10-01-preview/policyinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/policyinsights/mgmt/2020-07-01-preview/policyinsights` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/policyinsights/mgmt/2020-07-01-preview/policyinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/portal/mgmt/2015-08-01-preview/portal` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/portal/mgmt/2015-08-01-preview/portal/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/portal/mgmt/2018-10-01-preview/portal` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/portal/mgmt/2018-10-01-preview/portal/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/portal/mgmt/2019-01-01-preview/portal` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/portal/mgmt/2019-01-01-preview/portal/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/postgresql/mgmt/2017-12-01-preview/postgresql` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/postgresql/mgmt/2017-12-01-preview/postgresql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/postgresql/mgmt/2020-02-14-preview/postgresqlflexibleservers` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/postgresql/mgmt/2020-02-14-preview/postgresqlflexibleservers/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/postgresql/mgmt/2020-11-05-preview/postgresqlflexibleservers` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/postgresql/mgmt/2020-11-05-preview/postgresqlflexibleservers/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/powerplatform/mgmt/2020-10-30/powerplatform` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/powerplatform/mgmt/2020-10-30/powerplatform/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/provisioningservices/mgmt/2017-08-21-preview/iothub` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/provisioningservices/mgmt/2017-08-21-preview/iothub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/purview/mgmt/2020-12-01-preview/purview` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/purview/mgmt/2020-12-01-preview/purview/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/quantum/mgmt/2019-11-04-preview/quantum` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/quantum/mgmt/2019-11-04-preview/quantum/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/redis/mgmt/2019-07-01-preview/redis` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/redis/mgmt/2019-07-01-preview/redis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/redisenterprise/mgmt/2020-10-01-preview/redisenterprise` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/redisenterprise/mgmt/2020-10-01-preview/redisenterprise/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/reservations/mgmt/2018-06-01/reservations` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/reservations/mgmt/2018-06-01/reservations/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/reservations/mgmt/2019-07-19-preview/reservations` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/reservations/mgmt/2019-07-19-preview/reservations/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/reservations/mgmt/2020-10-25/reservations` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/reservations/mgmt/2020-10-25/reservations/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resourcemover/mgmt/2019-10-01-preview/resourcemover` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/resourcemover/mgmt/2019-10-01-preview/resourcemover/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2017-08-31-preview/managementgroups` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/resources/mgmt/2017-08-31-preview/managementgroups/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2017-11-01-preview/managementgroups` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/resources/mgmt/2017-11-01-preview/managementgroups/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2018-01-01-preview/managementgroups` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/resources/mgmt/2018-01-01-preview/managementgroups/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2018-03-01-preview/managementgroups` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/resources/mgmt/2018-03-01-preview/managementgroups/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2019-06-01-preview/templatespecs` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/resources/mgmt/2019-06-01-preview/templatespecs/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2020-03-01-preview/policy` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/resources/mgmt/2020-03-01-preview/policy/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2020-09-01-preview/policy` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/resources/mgmt/2020-09-01-preview/policy/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/scheduler/mgmt/2014-08-01-preview/scheduler` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/scheduler/mgmt/2014-08-01-preview/scheduler/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/security/mgmt/v1.0/security` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/security/mgmt/v1.0/security/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/security/mgmt/v2.0/security` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/security/mgmt/v2.0/security/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/security/mgmt/v3.0/security` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/security/mgmt/v3.0/security/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/servicebus/mgmt/2018-01-01-preview/servicebus` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/servicebus/mgmt/2018-01-01-preview/servicebus/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/servicefabric/mgmt/2017-07-01-preview/servicefabric` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/servicefabric/mgmt/2017-07-01-preview/servicefabric/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/servicefabric/mgmt/2018-02-01-preview/servicefabric` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/servicefabric/mgmt/2018-02-01-preview/servicefabric/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/servicefabric/mgmt/2019-03-01-preview/servicefabric` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/servicefabric/mgmt/2019-03-01-preview/servicefabric/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/servicefabricmesh/mgmt/2018-07-01-preview/servicefabricmesh` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/servicefabricmesh/mgmt/2018-07-01-preview/servicefabricmesh/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/servicefabricmesh/mgmt/2018-09-01-preview/servicefabricmesh` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/servicefabricmesh/mgmt/2018-09-01-preview/servicefabricmesh/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/signalr/mgmt/2018-03-01-preview/signalr` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/signalr/mgmt/2018-03-01-preview/signalr/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/signalr/mgmt/2020-07-01-preview/signalr` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/signalr/mgmt/2020-07-01-preview/signalr/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/signalr/mgmt/2021-04-01-preview/signalr` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/signalr/mgmt/2021-04-01-preview/signalr/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/softwareplan/mgmt/2019-06-01-preview/softwareplan` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/softwareplan/mgmt/2019-06-01-preview/softwareplan/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sql/mgmt/2015-05-01-preview/sql` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/sql/mgmt/2015-05-01-preview/sql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sql/mgmt/2017-03-01-preview/sql` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/sql/mgmt/2017-03-01-preview/sql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sql/mgmt/2017-10-01-preview/sql` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/sql/mgmt/2017-10-01-preview/sql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sql/mgmt/2018-06-01-preview/sql` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/sql/mgmt/2018-06-01-preview/sql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sql/mgmt/v3.0/sql` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/sql/mgmt/v3.0/sql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sql/mgmt/v4.0/sql` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/sql/mgmt/v4.0/sql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sqlvirtualmachine/mgmt/2017-03-01-preview/sqlvirtualmachine` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/sqlvirtualmachine/mgmt/2017-03-01-preview/sqlvirtualmachine/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/storage/mgmt/2018-03-01-preview/storage` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/storage/mgmt/2018-03-01-preview/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/storage/mgmt/2018-07-01-preview/storage` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/storage/mgmt/2018-07-01-preview/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/storage/mgmt/2020-08-01-preview/storage` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/storage/mgmt/2020-08-01-preview/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/storagecache/mgmt/2019-08-01-preview/storagecache` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/storagecache/mgmt/2019-08-01-preview/storagecache/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/storagepool/mgmt/2020-03-15-preview/storagepool` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/storagepool/mgmt/2020-03-15-preview/storagepool/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/streamanalytics/mgmt/2020-03-01-preview/streamanalytics` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/streamanalytics/mgmt/2020-03-01-preview/streamanalytics/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/subscription/mgmt/2018-03-01-preview/subscription` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/subscription/mgmt/2018-03-01-preview/subscription/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/subscription/mgmt/2019-10-01-preview/subscription` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/subscription/mgmt/2019-10-01-preview/subscription/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/support/mgmt/2019-05-01-preview/support` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/support/mgmt/2019-05-01-preview/support/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/synapse/2019-06-01-preview/artifacts` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/synapse/2019-06-01-preview/artifacts/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/synapse/mgmt/2019-06-01-preview/synapse` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/synapse/mgmt/2019-06-01-preview/synapse/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/timeseriesinsights/mgmt/2017-02-28-preview/timeseriesinsights` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/timeseriesinsights/mgmt/2017-02-28-preview/timeseriesinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/timeseriesinsights/mgmt/2018-08-15-preview/timeseriesinsights` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/timeseriesinsights/mgmt/2018-08-15-preview/timeseriesinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/trafficmanager/mgmt/2017-09-01-preview/trafficmanager` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/trafficmanager/mgmt/2017-09-01-preview/trafficmanager/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/trafficmanager/mgmt/2018-02-01-preview/trafficmanager` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/trafficmanager/mgmt/2018-02-01-preview/trafficmanager/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/virtualmachineimagebuilder/mgmt/2019-05-01-preview/virtualmachineimagebuilder` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/virtualmachineimagebuilder/mgmt/2019-05-01-preview/virtualmachineimagebuilder/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/web/mgmt/2015-08-01-preview/web` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/web/mgmt/2015-08-01-preview/web/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/webpubsub/mgmt/2021-04-01-preview/webpubsub` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/webpubsub/mgmt/2021-04-01-preview/webpubsub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/windowsesu/2019-09-16-preview/windowsesu` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/windowsesu/2019-09-16-preview/windowsesu/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/workloadmonitor/mgmt/2018-08-31-preview/workloadmonitor` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/workloadmonitor/mgmt/2018-08-31-preview/workloadmonitor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/workloadmonitor/mgmt/2020-01-13-preview/workloadmonitor` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/workloadmonitor/mgmt/2020-01-13-preview/workloadmonitor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/privatedns/mgmt/2018-09-01/privatedns` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/privatedns/mgmt/2018-09-01/privatedns/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/provisioningservices/mgmt/2017-11-15/iothub` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/provisioningservices/mgmt/2017-11-15/iothub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/provisioningservices/mgmt/2018-01-22/iothub` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/provisioningservices/mgmt/2018-01-22/iothub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2016-06-01/recoveryservices` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/recoveryservices/mgmt/2016-06-01/recoveryservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2016-08-10/siterecovery` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/recoveryservices/mgmt/2016-08-10/siterecovery/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2016-12-01/backup` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/recoveryservices/mgmt/2016-12-01/backup/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2018-01-10/siterecovery` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/recoveryservices/mgmt/2018-01-10/siterecovery/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2018-07-10/siterecovery` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/recoveryservices/mgmt/2018-07-10/siterecovery/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2019-05-13/backup` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/recoveryservices/mgmt/2019-05-13/backup/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2019-06-15/backup` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/recoveryservices/mgmt/2019-06-15/backup/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2020-02-02/backup` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/recoveryservices/mgmt/2020-02-02/backup/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redhatopenshift/mgmt/2020-04-30/redhatopenshift` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/redhatopenshift/mgmt/2020-04-30/redhatopenshift/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2016-04-01/redis` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/redis/mgmt/2016-04-01/redis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2017-02-01/redis` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/redis/mgmt/2017-02-01/redis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2017-10-01/redis` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/redis/mgmt/2017-10-01/redis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2018-03-01/redis` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/redis/mgmt/2018-03-01/redis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2020-06-01/redis` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/redis/mgmt/2020-06-01/redis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redisenterprise/mgmt/2021-03-01/redisenterprise` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/redisenterprise/mgmt/2021-03-01/redisenterprise/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/relay/mgmt/2016-07-01/relay` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/relay/mgmt/2016-07-01/relay/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/relay/mgmt/2017-04-01/relay` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/relay/mgmt/2017-04-01/relay/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/reservations/mgmt/2017-11-01/reservations` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/reservations/mgmt/2017-11-01/reservations/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resourcehealth/mgmt/2015-01-01/resourcehealth` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resourcehealth/mgmt/2015-01-01/resourcehealth/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resourcehealth/mgmt/2017-07-01/resourcehealth` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resourcehealth/mgmt/2017-07-01/resourcehealth/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resourcemover/mgmt/2021-01-01/resourcemover` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resourcemover/mgmt/2021-01-01/resourcemover/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2015-11-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resources/mgmt/2015-11-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2016-02-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resources/mgmt/2016-02-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2016-06-01/subscriptions` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resources/mgmt/2016-06-01/subscriptions/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2016-07-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resources/mgmt/2016-07-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2016-09-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resources/mgmt/2016-09-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2017-05-10/resources` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resources/mgmt/2017-05-10/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2018-02-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resources/mgmt/2018-02-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2018-05-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resources/mgmt/2018-05-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2018-06-01/subscriptions` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resources/mgmt/2018-06-01/subscriptions/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-03-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resources/mgmt/2019-03-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-05-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resources/mgmt/2019-05-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-06-01/subscriptions` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resources/mgmt/2019-06-01/subscriptions/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-07-01/features` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resources/mgmt/2019-07-01/features/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-07-01/managedapplications` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resources/mgmt/2019-07-01/managedapplications/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-09-01/policy` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resources/mgmt/2019-09-01/policy/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-10-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resources/mgmt/2019-10-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-11-01/managementgroups` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resources/mgmt/2019-11-01/managementgroups/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-11-01/subscriptions` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resources/mgmt/2019-11-01/subscriptions/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2020-02-01/managementgroups` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resources/mgmt/2020-02-01/managementgroups/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2020-05-01/managementgroups` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resources/mgmt/2020-05-01/managementgroups/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2020-06-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resources/mgmt/2020-06-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2020-10-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resources/mgmt/2020-10-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2021-01-01/subscriptions` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/resources/mgmt/2021-01-01/subscriptions/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/scheduler/mgmt/2016-01-01/scheduler` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/scheduler/mgmt/2016-01-01/scheduler/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/scheduler/mgmt/2016-03-01/scheduler` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/scheduler/mgmt/2016-03-01/scheduler/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/search/mgmt/2015-02-28/search` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/search/mgmt/2015-02-28/search/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/search/mgmt/2015-08-19/search` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/search/mgmt/2015-08-19/search/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/search/mgmt/2020-03-13/search` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/search/mgmt/2020-03-13/search/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/search/mgmt/2020-08-01/search` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/search/mgmt/2020-08-01/search/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/securityinsight/mgmt/v1.0/securityinsight` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/securityinsight/mgmt/v1.0/securityinsight/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/servicebus/mgmt/2015-08-01/servicebus` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/servicebus/mgmt/2015-08-01/servicebus/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/servicebus/mgmt/2017-04-01/servicebus` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/servicebus/mgmt/2017-04-01/servicebus/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/servicefabric/mgmt/2016-09-01/servicefabric` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/servicefabric/mgmt/2016-09-01/servicefabric/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/servicefabric/mgmt/2019-03-01/servicefabric` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/servicefabric/mgmt/2019-03-01/servicefabric/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/signalr/mgmt/2018-10-01/signalr` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/signalr/mgmt/2018-10-01/signalr/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/signalr/mgmt/2020-05-01/signalr` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/signalr/mgmt/2020-05-01/signalr/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/sql/mgmt/2014-04-01/sql` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/sql/mgmt/2014-04-01/sql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2016-01-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/storage/mgmt/2016-01-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2016-05-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/storage/mgmt/2016-05-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2016-12-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/storage/mgmt/2016-12-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2017-06-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/storage/mgmt/2017-06-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2017-10-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/storage/mgmt/2017-10-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2018-02-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/storage/mgmt/2018-02-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2018-11-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/storage/mgmt/2018-11-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2019-04-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/storage/mgmt/2019-04-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2019-06-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/storage/mgmt/2019-06-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2021-01-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/storage/mgmt/2021-01-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2021-02-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/storage/mgmt/2021-02-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagecache/mgmt/2019-11-01/storagecache` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/storagecache/mgmt/2019-11-01/storagecache/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagecache/mgmt/2020-03-01/storagecache` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/storagecache/mgmt/2020-03-01/storagecache/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagecache/mgmt/2020-10-01/storagecache` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/storagecache/mgmt/2020-10-01/storagecache/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagecache/mgmt/2021-03-01/storagecache` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/storagecache/mgmt/2021-03-01/storagecache/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2018-04-02/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/storagesync/mgmt/2018-04-02/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2018-07-01/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/storagesync/mgmt/2018-07-01/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2018-10-01/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/storagesync/mgmt/2018-10-01/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2019-02-01/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/storagesync/mgmt/2019-02-01/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2019-06-01/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/storagesync/mgmt/2019-06-01/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2019-10-01/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/storagesync/mgmt/2019-10-01/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2020-03-01/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/storagesync/mgmt/2020-03-01/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storsimple1200series/mgmt/2016-10-01/storsimple` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/storsimple1200series/mgmt/2016-10-01/storsimple/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/streamanalytics/mgmt/2016-03-01/streamanalytics` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/streamanalytics/mgmt/2016-03-01/streamanalytics/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/subscription/mgmt/2020-09-01/subscription` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/subscription/mgmt/2020-09-01/subscription/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/support/mgmt/2020-04-01/support` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/support/mgmt/2020-04-01/support/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/synapse/mgmt/2020-12-01/synapse` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/synapse/mgmt/2020-12-01/synapse/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/synapse/mgmt/2021-03-01/synapse` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/synapse/mgmt/2021-03-01/synapse/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/timeseriesinsights/mgmt/2017-11-15/timeseriesinsights` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/timeseriesinsights/mgmt/2017-11-15/timeseriesinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/timeseriesinsights/mgmt/2020-05-15/timeseriesinsights` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/timeseriesinsights/mgmt/2020-05-15/timeseriesinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/trafficmanager/mgmt/2017-03-01/trafficmanager` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/trafficmanager/mgmt/2017-03-01/trafficmanager/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/trafficmanager/mgmt/2017-05-01/trafficmanager` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/trafficmanager/mgmt/2017-05-01/trafficmanager/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/trafficmanager/mgmt/2018-03-01/trafficmanager` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/trafficmanager/mgmt/2018-03-01/trafficmanager/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/trafficmanager/mgmt/2018-04-01/trafficmanager` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/trafficmanager/mgmt/2018-04-01/trafficmanager/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/vmwarecloudsimple/mgmt/2019-04-01/vmwarecloudsimple` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/vmwarecloudsimple/mgmt/2019-04-01/vmwarecloudsimple/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/web/mgmt/2016-09-01/web` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/web/mgmt/2016-09-01/web/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/web/mgmt/2018-02-01/web` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/web/mgmt/2018-02-01/web/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/web/mgmt/2019-08-01/web` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/web/mgmt/2019-08-01/web/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/web/mgmt/2020-06-01/web` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/web/mgmt/2020-06-01/web/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/web/mgmt/2020-09-01/web` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/web/mgmt/2020-09-01/web/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/web/mgmt/2020-12-01/web` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/web/mgmt/2020-12-01/web/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/windowsiot/mgmt/2019-06-01/windowsiot` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/windowsiot/mgmt/2019-06-01/windowsiot/CHANGELOG.md) |

### Breaking Changes

| Package Path | Changelog |
| :--- | :---: |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2021-03-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/compute/mgmt/2021-03-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datadog/mgmt/2021-03-01/datadog` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/datadog/mgmt/2021-03-01/datadog/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datafactory/mgmt/2018-06-01/datafactory` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/datafactory/mgmt/2018-06-01/datafactory/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/delegatednetwork/mgmt/2021-03-15/delegatednetwork` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/delegatednetwork/mgmt/2021-03-15/delegatednetwork/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2021-01-01/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/kusto/mgmt/2021-01-01/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-11-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/network/mgmt/2020-11-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2021-04-01-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/cosmos-db/mgmt/2021-04-01-preview/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resourcegraph/mgmt/2021-03-01-preview/resourcegraph` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/resourcegraph/mgmt/2021-03-01-preview/resourcegraph/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/securityinsight/mgmt/2019-01-01-preview/securityinsight` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/securityinsight/mgmt/2019-01-01-preview/securityinsight/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/videoanalyzer/mgmt/2021-05-01-preview/videoanalyzer` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/preview/videoanalyzer/mgmt/2021-05-01-preview/videoanalyzer/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/securityinsight/mgmt/2020-01-01/securityinsight` | [details](https://github.com/Azure/azure-sdk-for-go/tree/master/services/securityinsight/mgmt/2020-01-01/securityinsight/CHANGELOG.md) |

### Removed Packages

- `github.com/Azure/azure-sdk-for-go/services/virtualmachineimagebuilder/mgmt/2020-02-01/virtualmachineimagebuilder`

## `v54.3.0`

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2021-02-01/network`
- `github.com/Azure/azure-sdk-for-go/services/preview/appconfiguration/mgmt/2021-03-01-preview/appconfiguration`
- `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2021-04-01/storage`

### Breaking Changes

| Package Path | Changelog |
| :--- | :---: |
| `github.com/Azure/azure-sdk-for-go/services/preview/authorization/mgmt/2020-04-01-preview/authorization` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v54.3.0/services/preview/authorization/mgmt/2020-04-01-preview/authorization/CHANGELOG.md) |

## `v54.2.0`

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/botservice/mgmt/2021-03-01/botservice`
- `github.com/Azure/azure-sdk-for-go/services/mediaservices/mgmt/2021-05-01/media`
- `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2021-02-01/netapp`

## `v54.1.0`

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/maintenance/mgmt/2021-05-01/maintenance`
- `github.com/Azure/azure-sdk-for-go/services/preview/confluent/mgmt/2021-03-01-preview/confluent`
- `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2020-09-01-preview/policy`
- `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2021-01-01/subscriptions`

## `v54.0.0`

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/dataprotection/mgmt/2021-01-01/dataprotection`
- `github.com/Azure/azure-sdk-for-go/services/maps/mgmt/2021-02-01/maps`
- `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2021-04-01-preview/documentdb`
- `github.com/Azure/azure-sdk-for-go/services/preview/elastic/mgmt/2020-07-01-preview/elastic`
- `github.com/Azure/azure-sdk-for-go/services/preview/resourcegraph/mgmt/2021-03-01-preview/resourcegraph`
- `github.com/Azure/azure-sdk-for-go/services/preview/videoanalyzer/mgmt/2021-05-01-preview/videoanalyzer`
- `github.com/Azure/azure-sdk-for-go/services/securityinsight/mgmt/2020-01-01/securityinsight`

### Breaking Changes

| Package Path | Changelog |
| :--- | :---: |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2021-03-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v54.0.0/services/containerservice/mgmt/2021-03-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datafactory/mgmt/2018-06-01/datafactory` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v54.0.0/services/datafactory/mgmt/2018-06-01/datafactory/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/hdinsight/mgmt/2018-06-01/hdinsight` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v54.0.0/services/hdinsight/mgmt/2018-06-01/hdinsight/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2021-01-01/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v54.0.0/services/kusto/mgmt/2021-01-01/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-11-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v54.0.0/services/network/mgmt/2020-11-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/confidentialledger/mgmt/2020-12-01-preview/confidentialledger` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v54.0.0/services/preview/confidentialledger/mgmt/2020-12-01-preview/confidentialledger/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/extendedlocation/mgmt/2021-03-15-preview/extendedlocation` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v54.0.0/services/preview/extendedlocation/mgmt/2021-03-15-preview/extendedlocation/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/iothub/mgmt/2021-03-03-preview/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v54.0.0/services/preview/iothub/mgmt/2021-03-03-preview/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/securityinsight/mgmt/2019-01-01-preview/securityinsight` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v54.0.0/services/preview/securityinsight/mgmt/2019-01-01-preview/securityinsight/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/signalr/mgmt/2021-04-01-preview/signalr` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v54.0.0/services/preview/signalr/mgmt/2021-04-01-preview/signalr/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/webpubsub/mgmt/2021-04-01-preview/webpubsub` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v54.0.0/services/preview/webpubsub/mgmt/2021-04-01-preview/webpubsub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2020-06-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v54.0.0/services/resources/mgmt/2020-06-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2020-10-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v54.0.0/services/resources/mgmt/2020-10-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2021-02-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v54.0.0/services/storage/mgmt/2021-02-01/storage/CHANGELOG.md) |

## `v53.4.0`

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2021-01-01/kusto`
- `github.com/Azure/azure-sdk-for-go/services/preview/confidentialledger/mgmt/2020-12-01-preview/confidentialledger`
- `github.com/Azure/azure-sdk-for-go/services/preview/iothub/mgmt/2021-03-03-preview/devices`
- `github.com/Azure/azure-sdk-for-go/services/preview/webpubsub/mgmt/2021-04-01-preview/webpubsub`

### Breaking Changes

| Package Path | Changelog |
| :--- | :---: |
| `github.com/Azure/azure-sdk-for-go/services/preview/securityinsight/mgmt/2019-01-01-preview/securityinsight` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.4.0/services/preview/securityinsight/mgmt/2019-01-01-preview/securityinsight/CHANGELOG.md) |

## `v53.3.0`

Fixes https://github.com/Azure/azure-sdk-for-go/issues/14540

## `v53.2.0`

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/preview/signalr/mgmt/2021-04-01-preview/signalr`
- `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2021-02-01/storage`

### Updated Packages

| Package Path | Changelog |
| :--- | :---: |
| `github.com/Azure/azure-sdk-for-go/services/datafactory/mgmt/2018-06-01/datafactory` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.2.0/services/datafactory/mgmt/2018-06-01/datafactory/CHANGELOG.md) |

## `v53.1.0`

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2021-03-01/containerservice`

## `v53.0.0`

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/cosmos-db/mgmt/2021-03-15/documentdb`
- `github.com/Azure/azure-sdk-for-go/services/costmanagement/mgmt/2020-06-01/costmanagement`
- `github.com/Azure/azure-sdk-for-go/services/resourcegraph/mgmt/2021-03-01/resourcegraph`
- `github.com/Azure/azure-sdk-for-go/services/synapse/mgmt/2021-03-01/synapse`
- `github.com/Azure/azure-sdk-for-go/services/web/mgmt/2020-12-01/web`

### Updated Packages

| Package Path | Changelog |
| :--- | :---: |
| `github.com/Azure/azure-sdk-for-go/services/analysisservices/mgmt/2016-05-16/analysisservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/analysisservices/mgmt/2016-05-16/analysisservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/analysisservices/mgmt/2017-07-14/analysisservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/analysisservices/mgmt/2017-07-14/analysisservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/analysisservices/mgmt/2017-08-01/analysisservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/analysisservices/mgmt/2017-08-01/analysisservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2016-07-07/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/apimanagement/mgmt/2016-07-07/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2016-10-10/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/apimanagement/mgmt/2016-10-10/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2017-03-01/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/apimanagement/mgmt/2017-03-01/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2018-01-01/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/apimanagement/mgmt/2018-01-01/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2019-01-01/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/apimanagement/mgmt/2019-01-01/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2019-12-01/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/apimanagement/mgmt/2019-12-01/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2020-12-01/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/apimanagement/mgmt/2020-12-01/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/appconfiguration/mgmt/2019-10-01/appconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/appconfiguration/mgmt/2019-10-01/appconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/appconfiguration/mgmt/2020-06-01/appconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/appconfiguration/mgmt/2020-06-01/appconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/appplatform/mgmt/2020-07-01/appplatform` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/appplatform/mgmt/2020-07-01/appplatform/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/automation/mgmt/2015-10-31/automation` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/automation/mgmt/2015-10-31/automation/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/avs/mgmt/2020-03-20/avs` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/avs/mgmt/2020-03-20/avs/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2015-12-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/batch/mgmt/2015-12-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2017-01-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/batch/mgmt/2017-01-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2017-05-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/batch/mgmt/2017-05-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2017-09-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/batch/mgmt/2017-09-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2018-12-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/batch/mgmt/2018-12-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2019-04-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/batch/mgmt/2019-04-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2019-08-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/batch/mgmt/2019-08-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2020-03-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/batch/mgmt/2020-03-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2020-05-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/batch/mgmt/2020-05-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2020-09-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/batch/mgmt/2020-09-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batchai/mgmt/2018-03-01/batchai` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/batchai/mgmt/2018-03-01/batchai/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batchai/mgmt/2018-05-01/batchai` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/batchai/mgmt/2018-05-01/batchai/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2015-06-01/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/cdn/mgmt/2015-06-01/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2016-04-02/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/cdn/mgmt/2016-04-02/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2016-10-02/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/cdn/mgmt/2016-10-02/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2017-04-02/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/cdn/mgmt/2017-04-02/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2017-10-12/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/cdn/mgmt/2017-10-12/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2019-04-15/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/cdn/mgmt/2019-04-15/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2019-06-15/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/cdn/mgmt/2019-06-15/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2020-04-15/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/cdn/mgmt/2020-04-15/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2020-09-01/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/cdn/mgmt/2020-09-01/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/communication/mgmt/2020-08-20/communication` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/communication/mgmt/2020-08-20/communication/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2015-06-15/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/compute/mgmt/2015-06-15/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2016-03-30/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/compute/mgmt/2016-03-30/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2017-03-30/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/compute/mgmt/2017-03-30/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2017-12-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/compute/mgmt/2017-12-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2018-04-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/compute/mgmt/2018-04-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2018-06-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/compute/mgmt/2018-06-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2018-10-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/compute/mgmt/2018-10-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-03-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/compute/mgmt/2019-03-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-07-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/compute/mgmt/2019-07-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-12-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/compute/mgmt/2019-12-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2020-06-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/compute/mgmt/2020-06-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2020-06-30/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/compute/mgmt/2020-06-30/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2021-03-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/compute/mgmt/2021-03-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/confluent/mgmt/2020-03-01/confluent` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/confluent/mgmt/2020-03-01/confluent/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerinstance/mgmt/2018-04-01/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerinstance/mgmt/2018-04-01/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerinstance/mgmt/2018-06-01/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerinstance/mgmt/2018-06-01/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerinstance/mgmt/2018-09-01/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerinstance/mgmt/2018-09-01/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerinstance/mgmt/2018-10-01/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerinstance/mgmt/2018-10-01/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerinstance/mgmt/2019-12-01/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerinstance/mgmt/2019-12-01/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerinstance/mgmt/2020-11-01/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerinstance/mgmt/2020-11-01/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2017-03-01/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerregistry/mgmt/2017-03-01/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2017-10-01/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerregistry/mgmt/2017-10-01/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2018-09-01/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerregistry/mgmt/2018-09-01/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2019-04-01/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerregistry/mgmt/2019-04-01/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2019-05-01/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerregistry/mgmt/2019-05-01/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2016-03-30/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerservice/mgmt/2016-03-30/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2016-09-30/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerservice/mgmt/2016-09-30/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2017-01-31/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerservice/mgmt/2017-01-31/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2017-07-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerservice/mgmt/2017-07-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2017-08-31/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerservice/mgmt/2017-08-31/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2017-09-30/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerservice/mgmt/2017-09-30/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2018-03-31/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerservice/mgmt/2018-03-31/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2019-04-30/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerservice/mgmt/2019-04-30/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2019-06-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerservice/mgmt/2019-06-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2019-08-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerservice/mgmt/2019-08-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2019-10-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerservice/mgmt/2019-10-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2019-11-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerservice/mgmt/2019-11-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-01-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerservice/mgmt/2020-01-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-02-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerservice/mgmt/2020-02-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-03-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerservice/mgmt/2020-03-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-04-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerservice/mgmt/2020-04-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-06-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerservice/mgmt/2020-06-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-07-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerservice/mgmt/2020-07-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-09-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerservice/mgmt/2020-09-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-11-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerservice/mgmt/2020-11-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-12-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerservice/mgmt/2020-12-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2021-02-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/containerservice/mgmt/2021-02-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cosmos-db/mgmt/2015-04-08/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/cosmos-db/mgmt/2015-04-08/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cosmos-db/mgmt/2019-08-01/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/cosmos-db/mgmt/2019-08-01/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cosmos-db/mgmt/2021-01-15/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/cosmos-db/mgmt/2021-01-15/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/customerinsights/mgmt/2017-01-01/customerinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/customerinsights/mgmt/2017-01-01/customerinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/customerinsights/mgmt/2017-04-26/customerinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/customerinsights/mgmt/2017-04-26/customerinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databox/mgmt/2018-01-01/databox` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/databox/mgmt/2018-01-01/databox/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databox/mgmt/2019-09-01/databox` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/databox/mgmt/2019-09-01/databox/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databox/mgmt/2020-04-01/databox` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/databox/mgmt/2020-04-01/databox/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databox/mgmt/2020-11-01/databox` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/databox/mgmt/2020-11-01/databox/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databoxedge/mgmt/2019-03-01/databoxedge` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/databoxedge/mgmt/2019-03-01/databoxedge/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databoxedge/mgmt/2019-07-01/databoxedge` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/databoxedge/mgmt/2019-07-01/databoxedge/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databoxedge/mgmt/2019-08-01/databoxedge` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/databoxedge/mgmt/2019-08-01/databoxedge/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databoxedge/mgmt/2020-12-01/databoxedge` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/databoxedge/mgmt/2020-12-01/databoxedge/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databricks/mgmt/2018-04-01/databricks` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/databricks/mgmt/2018-04-01/databricks/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datacatalog/mgmt/2016-03-30/datacatalog` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/datacatalog/mgmt/2016-03-30/datacatalog/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datadog/mgmt/2021-03-01/datadog` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/datadog/mgmt/2021-03-01/datadog/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datalake/analytics/mgmt/2016-11-01/account` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/datalake/analytics/mgmt/2016-11-01/account/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datalake/store/mgmt/2016-11-01/account` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/datalake/store/mgmt/2016-11-01/account/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datamigration/mgmt/2018-04-19/datamigration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/datamigration/mgmt/2018-04-19/datamigration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datashare/mgmt/2019-11-01/datashare` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/datashare/mgmt/2019-11-01/datashare/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/delegatednetwork/mgmt/2021-03-15/delegatednetwork` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/delegatednetwork/mgmt/2021-03-15/delegatednetwork/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/devspaces/mgmt/2019-04-01/devspaces` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/devspaces/mgmt/2019-04-01/devspaces/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/devtestlabs/mgmt/2016-05-15/dtl` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/devtestlabs/mgmt/2016-05-15/dtl/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/devtestlabs/mgmt/2018-09-15/dtl` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/devtestlabs/mgmt/2018-09-15/dtl/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/digitaltwins/mgmt/2020-10-31/digitaltwins` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/digitaltwins/mgmt/2020-10-31/digitaltwins/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/digitaltwins/mgmt/2020-12-01/digitaltwins` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/digitaltwins/mgmt/2020-12-01/digitaltwins/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/dns/mgmt/2016-04-01/dns` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/dns/mgmt/2016-04-01/dns/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/dns/mgmt/2017-09-01/dns` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/dns/mgmt/2017-09-01/dns/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/dns/mgmt/2017-10-01/dns` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/dns/mgmt/2017-10-01/dns/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/dns/mgmt/2018-05-01/dns` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/dns/mgmt/2018-05-01/dns/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/domainservices/mgmt/2017-01-01/aad` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/domainservices/mgmt/2017-01-01/aad/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/domainservices/mgmt/2017-06-01/aad` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/domainservices/mgmt/2017-06-01/aad/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/domainservices/mgmt/2020-01-01/aad` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/domainservices/mgmt/2020-01-01/aad/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/eventgrid/mgmt/2018-01-01/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/eventgrid/mgmt/2018-01-01/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/eventgrid/mgmt/2019-01-01/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/eventgrid/mgmt/2019-01-01/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/eventgrid/mgmt/2019-06-01/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/eventgrid/mgmt/2019-06-01/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/eventgrid/mgmt/2020-06-01/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/eventgrid/mgmt/2020-06-01/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/eventhub/mgmt/2015-08-01/eventhub` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/eventhub/mgmt/2015-08-01/eventhub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/eventhub/mgmt/2017-04-01/eventhub` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/eventhub/mgmt/2017-04-01/eventhub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2019-04-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/frontdoor/mgmt/2019-04-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2019-05-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/frontdoor/mgmt/2019-05-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2019-10-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/frontdoor/mgmt/2019-10-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2019-11-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/frontdoor/mgmt/2019-11-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2020-01-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/frontdoor/mgmt/2020-01-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2020-04-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/frontdoor/mgmt/2020-04-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2020-05-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/frontdoor/mgmt/2020-05-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/hdinsight/mgmt/2018-06-01/hdinsight` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/hdinsight/mgmt/2018-06-01/hdinsight/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/healthbot/mgmt/2020-12-08/healthbot` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/healthbot/mgmt/2020-12-08/healthbot/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/healthcareapis/mgmt/2019-09-16/healthcareapis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/healthcareapis/mgmt/2019-09-16/healthcareapis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/healthcareapis/mgmt/2020-03-15/healthcareapis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/healthcareapis/mgmt/2020-03-15/healthcareapis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/healthcareapis/mgmt/2020-03-30/healthcareapis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/healthcareapis/mgmt/2020-03-30/healthcareapis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/hybridcompute/mgmt/2019-12-12/hybridcompute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/hybridcompute/mgmt/2019-12-12/hybridcompute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/hybridcompute/mgmt/2020-08-02/hybridcompute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/hybridcompute/mgmt/2020-08-02/hybridcompute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/hybriddatamanager/mgmt/2016-06-01/hybriddata` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/hybriddatamanager/mgmt/2016-06-01/hybriddata/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/hybriddatamanager/mgmt/2019-06-01/hybriddata` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/hybriddatamanager/mgmt/2019-06-01/hybriddata/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/hybridkubernetes/mgmt/2021-03-01/hybridkubernetes` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/hybridkubernetes/mgmt/2021-03-01/hybridkubernetes/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iotcentral/mgmt/2018-09-01/iotcentral` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/iotcentral/mgmt/2018-09-01/iotcentral/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iothub/mgmt/2016-02-03/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/iothub/mgmt/2016-02-03/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iothub/mgmt/2017-01-19/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/iothub/mgmt/2017-01-19/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iothub/mgmt/2017-07-01/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/iothub/mgmt/2017-07-01/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iothub/mgmt/2018-01-22/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/iothub/mgmt/2018-01-22/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iothub/mgmt/2018-04-01/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/iothub/mgmt/2018-04-01/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iothub/mgmt/2020-03-01/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/iothub/mgmt/2020-03-01/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/keyvault/mgmt/2016-10-01/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/keyvault/mgmt/2016-10-01/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/keyvault/mgmt/2018-02-14/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/keyvault/mgmt/2018-02-14/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/keyvault/mgmt/2019-09-01/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/keyvault/mgmt/2019-09-01/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kubernetesconfiguration/mgmt/2021-03-01/kubernetesconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/kubernetesconfiguration/mgmt/2021-03-01/kubernetesconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2019-01-21/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/kusto/mgmt/2019-01-21/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2019-05-15/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/kusto/mgmt/2019-05-15/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2019-09-07/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/kusto/mgmt/2019-09-07/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2019-11-09/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/kusto/mgmt/2019-11-09/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2020-02-15/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/kusto/mgmt/2020-02-15/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2020-06-14/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/kusto/mgmt/2020-06-14/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2020-09-18/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/kusto/mgmt/2020-09-18/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/labservices/mgmt/2018-10-15/labservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/labservices/mgmt/2018-10-15/labservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/logic/mgmt/2019-05-01/logic` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/logic/mgmt/2019-05-01/logic/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearning/mgmt/2017-01-01/webservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/machinelearning/mgmt/2017-01-01/webservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2018-11-19/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/machinelearningservices/mgmt/2018-11-19/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2019-05-01/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/machinelearningservices/mgmt/2019-05-01/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2019-06-01/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/machinelearningservices/mgmt/2019-06-01/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2019-11-01/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/machinelearningservices/mgmt/2019-11-01/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2020-01-01/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/machinelearningservices/mgmt/2020-01-01/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2020-03-01/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/machinelearningservices/mgmt/2020-03-01/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2020-04-01/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/machinelearningservices/mgmt/2020-04-01/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/managedservices/mgmt/2019-06-01/managedservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/managedservices/mgmt/2019-06-01/managedservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mariadb/mgmt/2018-06-01/mariadb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/mariadb/mgmt/2018-06-01/mariadb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mariadb/mgmt/2020-01-01/mariadb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/mariadb/mgmt/2020-01-01/mariadb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mediaservices/mgmt/2018-07-01/media` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/mediaservices/mgmt/2018-07-01/media/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mediaservices/mgmt/2020-05-01/media` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/mediaservices/mgmt/2020-05-01/media/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mysql/mgmt/2017-12-01/mysql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/mysql/mgmt/2017-12-01/mysql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mysql/mgmt/2020-01-01/mysql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/mysql/mgmt/2020-01-01/mysql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2019-05-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/netapp/mgmt/2019-05-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2019-06-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/netapp/mgmt/2019-06-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2019-07-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/netapp/mgmt/2019-07-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2019-08-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/netapp/mgmt/2019-08-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2019-10-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/netapp/mgmt/2019-10-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2019-11-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/netapp/mgmt/2019-11-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-02-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/netapp/mgmt/2020-02-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-03-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/netapp/mgmt/2020-03-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-05-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/netapp/mgmt/2020-05-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-06-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/netapp/mgmt/2020-06-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-07-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/netapp/mgmt/2020-07-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-08-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/netapp/mgmt/2020-08-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-09-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/netapp/mgmt/2020-09-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-11-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/netapp/mgmt/2020-11-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-12-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/netapp/mgmt/2020-12-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2015-06-15/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2015-06-15/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2016-03-30/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2016-03-30/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2016-06-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2016-06-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2016-09-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2016-09-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2016-12-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2016-12-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-03-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2017-03-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-06-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2017-06-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-08-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2017-08-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-09-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2017-09-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-10-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2017-10-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-11-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2017-11-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-01-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2018-01-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-02-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2018-02-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-04-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2018-04-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-06-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2018-06-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-07-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2018-07-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-08-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2018-08-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-10-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2018-10-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-11-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2018-11-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-12-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2018-12-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-02-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2019-02-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-04-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2019-04-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2019-06-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-07-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2019-07-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-08-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2019-08-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-09-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2019-09-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-11-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2019-11-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-12-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2019-12-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-03-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2020-03-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-04-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2020-04-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-05-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2020-05-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-06-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2020-06-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-07-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2020-07-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-08-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2020-08-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-11-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/network/mgmt/2020-11-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/notificationhubs/mgmt/2014-09-01/notificationhubs` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/notificationhubs/mgmt/2014-09-01/notificationhubs/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/notificationhubs/mgmt/2016-03-01/notificationhubs` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/notificationhubs/mgmt/2016-03-01/notificationhubs/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/notificationhubs/mgmt/2017-04-01/notificationhubs` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/notificationhubs/mgmt/2017-04-01/notificationhubs/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/operationalinsights/mgmt/2020-08-01/operationalinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/operationalinsights/mgmt/2020-08-01/operationalinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/operationalinsights/mgmt/2020-10-01/operationalinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/operationalinsights/mgmt/2020-10-01/operationalinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/postgresql/mgmt/2017-12-01/postgresql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/postgresql/mgmt/2017-12-01/postgresql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/postgresql/mgmt/2020-01-01/postgresql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/postgresql/mgmt/2020-01-01/postgresql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/powerbidedicated/mgmt/2017-10-01/powerbidedicated` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/powerbidedicated/mgmt/2017-10-01/powerbidedicated/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/powerbiembedded/mgmt/2016-01-29/powerbiembedded` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/powerbiembedded/mgmt/2016-01-29/powerbiembedded/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/addons/mgmt/2017-05-15/addons` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/addons/mgmt/2017-05-15/addons/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/addons/mgmt/2018-03-01/addons` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/addons/mgmt/2018-03-01/addons/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/apimanagement/ctrl/2017-03-01/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/apimanagement/ctrl/2017-03-01/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/apimanagement/mgmt/2018-06-01-preview/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/apimanagement/mgmt/2018-06-01-preview/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/apimanagement/mgmt/2019-12-01-preview/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/apimanagement/mgmt/2019-12-01-preview/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/appconfiguration/mgmt/2019-02-01-preview/appconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/appconfiguration/mgmt/2019-02-01-preview/appconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/appconfiguration/mgmt/2019-11-01-preview/appconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/appconfiguration/mgmt/2019-11-01-preview/appconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/appconfiguration/mgmt/2020-07-01-preview/appconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/appconfiguration/mgmt/2020-07-01-preview/appconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/appplatform/mgmt/2019-05-01-preview/appplatform` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/appplatform/mgmt/2019-05-01-preview/appplatform/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/appplatform/mgmt/2020-11-01-preview/appplatform` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/appplatform/mgmt/2020-11-01-preview/appplatform/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/automanage/mgmt/2020-06-30-preview/automanage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/automanage/mgmt/2020-06-30-preview/automanage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/automation/mgmt/2017-05-15-preview/automation` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/automation/mgmt/2017-05-15-preview/automation/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/automation/mgmt/2018-01-15-preview/automation` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/automation/mgmt/2018-01-15-preview/automation/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/automation/mgmt/2018-06-30-preview/automation` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/automation/mgmt/2018-06-30-preview/automation/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/avs/mgmt/2020-07-17-preview/avs` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/avs/mgmt/2020-07-17-preview/avs/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/baremetalinfrastructure/mgmt/2020-08-06-preview/baremetalinfrastructure` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/baremetalinfrastructure/mgmt/2020-08-06-preview/baremetalinfrastructure/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/batchai/mgmt/2017-09-01-preview/batchai` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/batchai/mgmt/2017-09-01-preview/batchai/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/billing/mgmt/2018-11-01-preview/billing` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/billing/mgmt/2018-11-01-preview/billing/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/billing/mgmt/2020-05-01-preview/billing` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/billing/mgmt/2020-05-01-preview/billing/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/blockchain/mgmt/2018-06-01-preview/blockchain` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/blockchain/mgmt/2018-06-01-preview/blockchain/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/botservice/mgmt/2018-07-12/botservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/botservice/mgmt/2018-07-12/botservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/communication/mgmt/2020-08-20-preview/communication` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/communication/mgmt/2020-08-20-preview/communication/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/compute/mgmt/2016-04-30-preview/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/compute/mgmt/2016-04-30-preview/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/compute/mgmt/2020-10-01-preview/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/compute/mgmt/2020-10-01-preview/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/confluent/mgmt/2020-03-01-preview/confluent` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/confluent/mgmt/2020-03-01-preview/confluent/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerinstance/mgmt/2018-02-01-preview/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/containerinstance/mgmt/2018-02-01-preview/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerregistry/mgmt/2017-06-01-preview/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/containerregistry/mgmt/2017-06-01-preview/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerregistry/mgmt/2018-02-01/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/containerregistry/mgmt/2018-02-01/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerregistry/mgmt/2019-06-01-preview/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/containerregistry/mgmt/2019-06-01-preview/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerregistry/mgmt/2019-12-01-preview/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/containerregistry/mgmt/2019-12-01-preview/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerregistry/mgmt/2020-11-01-preview/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/containerregistry/mgmt/2020-11-01-preview/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerservice/mgmt/2015-11-01-preview/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/containerservice/mgmt/2015-11-01-preview/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerservice/mgmt/2018-08-01-preview/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/containerservice/mgmt/2018-08-01-preview/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerservice/mgmt/2018-09-30-preview/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/containerservice/mgmt/2018-09-30-preview/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerservice/mgmt/2019-02-01-preview/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/containerservice/mgmt/2019-02-01-preview/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerservice/mgmt/2019-09-30-preview/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/containerservice/mgmt/2019-09-30-preview/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerservice/mgmt/2019-10-27-preview/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/containerservice/mgmt/2019-10-27-preview/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2019-08-01-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/cosmos-db/mgmt/2019-08-01-preview/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2019-12-12-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/cosmos-db/mgmt/2019-12-12-preview/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2020-03-01-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/cosmos-db/mgmt/2020-03-01-preview/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2020-04-01-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/cosmos-db/mgmt/2020-04-01-preview/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2020-06-01-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/cosmos-db/mgmt/2020-06-01-preview/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2020-09-01-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/cosmos-db/mgmt/2020-09-01-preview/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2021-03-01-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/cosmos-db/mgmt/2021-03-01-preview/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/customproviders/mgmt/2018-09-01-preview/customproviders` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/customproviders/mgmt/2018-09-01-preview/customproviders/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/databoxedge/mgmt/2020-05-01-preview/databoxedge` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/databoxedge/mgmt/2020-05-01-preview/databoxedge/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datadog/mgmt/2020-02-01-preview/datadog` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/datadog/mgmt/2020-02-01-preview/datadog/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datafactory/mgmt/2017-09-01-preview/datafactory` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/datafactory/mgmt/2017-09-01-preview/datafactory/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datalake/analytics/2017-09-01-preview/job` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/datalake/analytics/2017-09-01-preview/job/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datalake/analytics/mgmt/2015-10-01-preview/account` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/datalake/analytics/mgmt/2015-10-01-preview/account/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datalake/store/mgmt/2015-10-01-preview/account` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/datalake/store/mgmt/2015-10-01-preview/account/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datamigration/mgmt/2017-11-15-preview/datamigration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/datamigration/mgmt/2017-11-15-preview/datamigration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datamigration/mgmt/2018-03-31-preview/datamigration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/datamigration/mgmt/2018-03-31-preview/datamigration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datamigration/mgmt/2018-07-15-preview/datamigration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/datamigration/mgmt/2018-07-15-preview/datamigration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datashare/mgmt/2018-11-01-preview/datashare` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/datashare/mgmt/2018-11-01-preview/datashare/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/delegatednetwork/mgmt/2020-08-08-preview/delegatednetwork` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/delegatednetwork/mgmt/2020-08-08-preview/delegatednetwork/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/deploymentmanager/mgmt/2018-09-01-preview/deploymentmanager` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/deploymentmanager/mgmt/2018-09-01-preview/deploymentmanager/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/deploymentmanager/mgmt/2019-11-01-preview/deploymentmanager` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/deploymentmanager/mgmt/2019-11-01-preview/deploymentmanager/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/deviceupdate/mgmt/2020-03-01-preview/deviceupdate` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/deviceupdate/mgmt/2020-03-01-preview/deviceupdate/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/devops/mgmt/2019-07-01-preview/devops` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/devops/mgmt/2019-07-01-preview/devops/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/devtestlabs/mgmt/2015-05-21-preview/dtl` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/devtestlabs/mgmt/2015-05-21-preview/dtl/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/digitaltwins/mgmt/2020-03-01-preview/digitaltwins` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/digitaltwins/mgmt/2020-03-01-preview/digitaltwins/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/dns/mgmt/2018-03-01-preview/dns` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/dns/mgmt/2018-03-01-preview/dns/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2017-06-15-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/eventgrid/mgmt/2017-06-15-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2017-09-15-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/eventgrid/mgmt/2017-09-15-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2018-05-01-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/eventgrid/mgmt/2018-05-01-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2018-09-15-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/eventgrid/mgmt/2018-09-15-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2019-02-01-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/eventgrid/mgmt/2019-02-01-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2020-01-01-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/eventgrid/mgmt/2020-01-01-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2020-04-01-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/eventgrid/mgmt/2020-04-01-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2020-10-15-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/eventgrid/mgmt/2020-10-15-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventhub/mgmt/2018-01-01-preview/eventhub` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/eventhub/mgmt/2018-01-01-preview/eventhub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/extendedlocation/mgmt/2021-03-15-preview/extendedlocation` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/extendedlocation/mgmt/2021-03-15-preview/extendedlocation/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/frontdoor/mgmt/2018-08-01-preview/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/frontdoor/mgmt/2018-08-01-preview/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/hanaonazure/mgmt/2017-11-03-preview/hanaonazure` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/hanaonazure/mgmt/2017-11-03-preview/hanaonazure/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/hardwaresecuritymodules/mgmt/2018-10-31-preview/hardwaresecuritymodules` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/hardwaresecuritymodules/mgmt/2018-10-31-preview/hardwaresecuritymodules/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/hdinsight/mgmt/2015-03-01-preview/hdinsight` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/hdinsight/mgmt/2015-03-01-preview/hdinsight/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/healthcareapis/mgmt/2018-08-20-preview/healthcareapis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/healthcareapis/mgmt/2018-08-20-preview/healthcareapis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/hybridkubernetes/mgmt/2020-01-01-preview/hybridkubernetes` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/hybridkubernetes/mgmt/2020-01-01-preview/hybridkubernetes/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/hybridnetwork/mgmt/2020-01-01-preview/hybridnetwork` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/hybridnetwork/mgmt/2020-01-01-preview/hybridnetwork/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/iothub/mgmt/2018-12-01-preview/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/iothub/mgmt/2018-12-01-preview/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/iothub/mgmt/2019-03-22-preview/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/iothub/mgmt/2019-03-22-preview/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/iothub/mgmt/2019-07-01-preview/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/iothub/mgmt/2019-07-01-preview/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/iotspaces/mgmt/2017-10-01-preview/iotspaces` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/iotspaces/mgmt/2017-10-01-preview/iotspaces/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/keyvault/mgmt/2020-04-01-preview/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/keyvault/mgmt/2020-04-01-preview/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/keyvault/v7.2-preview/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/keyvault/v7.2-preview/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/kubernetesconfiguration/mgmt/2019-11-01-preview/kubernetesconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/kubernetesconfiguration/mgmt/2019-11-01-preview/kubernetesconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/kubernetesconfiguration/mgmt/2020-07-01-preview/kubernetesconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/kubernetesconfiguration/mgmt/2020-07-01-preview/kubernetesconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/kusto/mgmt/2018-09-07-preview/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/kusto/mgmt/2018-09-07-preview/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/logic/mgmt/2015-02-01-preview/logic` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/logic/mgmt/2015-02-01-preview/logic/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/machinelearning/mgmt/2016-05-01-preview/webservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/machinelearning/mgmt/2016-05-01-preview/webservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/machinelearning/mgmt/2017-08-01-preview/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/machinelearning/mgmt/2017-08-01-preview/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/machinelearningservices/mgmt/2018-03-01-preview/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/machinelearningservices/mgmt/2018-03-01-preview/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/machinelearningservices/mgmt/2020-02-18-preview/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/machinelearningservices/mgmt/2020-02-18-preview/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/managednetwork/mgmt/2019-06-01-preview/managednetwork` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/managednetwork/mgmt/2019-06-01-preview/managednetwork/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/managedservices/mgmt/2019-04-01/managedservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/managedservices/mgmt/2019-04-01/managedservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mediaservices/mgmt/2018-03-30-preview/media` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/mediaservices/mgmt/2018-03-30-preview/media/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mediaservices/mgmt/2018-06-01-preview/media` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/mediaservices/mgmt/2018-06-01-preview/media/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mediaservices/mgmt/2019-05-01-preview/media` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/mediaservices/mgmt/2019-05-01-preview/media/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/monitor/mgmt/2019-11-01-preview/insights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/monitor/mgmt/2019-11-01-preview/insights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mysql/mgmt/2017-12-01-preview/mysql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/mysql/mgmt/2017-12-01-preview/mysql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mysql/mgmt/2020-07-01-preview/mysqlflexibleservers` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/mysql/mgmt/2020-07-01-preview/mysqlflexibleservers/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/netapp/mgmt/2017-08-15/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/netapp/mgmt/2017-08-15/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/network/mgmt/2015-05-01-preview/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/network/mgmt/2015-05-01-preview/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/operationalinsights/mgmt/2015-11-01-preview/operationalinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/operationalinsights/mgmt/2015-11-01-preview/operationalinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/operationalinsights/mgmt/2020-03-01-preview/operationalinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/operationalinsights/mgmt/2020-03-01-preview/operationalinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/operationsmanagement/mgmt/2015-11-01-preview/operationsmanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/operationsmanagement/mgmt/2015-11-01-preview/operationsmanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/policyinsights/mgmt/2019-10-01-preview/policyinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/policyinsights/mgmt/2019-10-01-preview/policyinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/policyinsights/mgmt/2020-07-01-preview/policyinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/policyinsights/mgmt/2020-07-01-preview/policyinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/postgresql/mgmt/2017-12-01-preview/postgresql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/postgresql/mgmt/2017-12-01-preview/postgresql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/postgresql/mgmt/2020-02-14-preview/postgresqlflexibleservers` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/postgresql/mgmt/2020-02-14-preview/postgresqlflexibleservers/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/postgresql/mgmt/2020-11-05-preview/postgresqlflexibleservers` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/postgresql/mgmt/2020-11-05-preview/postgresqlflexibleservers/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/powerplatform/mgmt/2020-10-30/powerplatform` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/powerplatform/mgmt/2020-10-30/powerplatform/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/provisioningservices/mgmt/2017-08-21-preview/iothub` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/provisioningservices/mgmt/2017-08-21-preview/iothub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/purview/mgmt/2020-12-01-preview/purview` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/purview/mgmt/2020-12-01-preview/purview/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/quantum/mgmt/2019-11-04-preview/quantum` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/quantum/mgmt/2019-11-04-preview/quantum/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/redis/mgmt/2019-07-01-preview/redis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/redis/mgmt/2019-07-01-preview/redis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/redisenterprise/mgmt/2020-10-01-preview/redisenterprise` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/redisenterprise/mgmt/2020-10-01-preview/redisenterprise/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/reservations/mgmt/2018-06-01/reservations` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/reservations/mgmt/2018-06-01/reservations/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/reservations/mgmt/2019-04-01/reservations` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/reservations/mgmt/2019-04-01/reservations/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/reservations/mgmt/2019-07-19-preview/reservations` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/reservations/mgmt/2019-07-19-preview/reservations/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/reservations/mgmt/2020-10-25/reservations` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/reservations/mgmt/2020-10-25/reservations/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resourcemover/mgmt/2019-10-01-preview/resourcemover` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/resourcemover/mgmt/2019-10-01-preview/resourcemover/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2016-09-01-preview/managedapplications` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/resources/mgmt/2016-09-01-preview/managedapplications/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2018-01-01-preview/managementgroups` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/resources/mgmt/2018-01-01-preview/managementgroups/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2018-03-01-preview/managementgroups` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/resources/mgmt/2018-03-01-preview/managementgroups/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/security/mgmt/v1.0/security` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/security/mgmt/v1.0/security/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/security/mgmt/v2.0/security` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/security/mgmt/v2.0/security/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/security/mgmt/v3.0/security` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/security/mgmt/v3.0/security/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/servicebus/mgmt/2018-01-01-preview/servicebus` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/servicebus/mgmt/2018-01-01-preview/servicebus/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/servicefabric/mgmt/2017-07-01-preview/servicefabric` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/servicefabric/mgmt/2017-07-01-preview/servicefabric/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/servicefabric/mgmt/2018-02-01-preview/servicefabric` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/servicefabric/mgmt/2018-02-01-preview/servicefabric/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/servicefabric/mgmt/2019-03-01-preview/servicefabric` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/servicefabric/mgmt/2019-03-01-preview/servicefabric/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/signalr/mgmt/2018-03-01-preview/signalr` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/signalr/mgmt/2018-03-01-preview/signalr/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/signalr/mgmt/2020-07-01-preview/signalr` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/signalr/mgmt/2020-07-01-preview/signalr/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sql/mgmt/2015-05-01-preview/sql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/sql/mgmt/2015-05-01-preview/sql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sql/mgmt/2017-03-01-preview/sql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/sql/mgmt/2017-03-01-preview/sql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sql/mgmt/2017-10-01-preview/sql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/sql/mgmt/2017-10-01-preview/sql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sql/mgmt/2018-06-01-preview/sql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/sql/mgmt/2018-06-01-preview/sql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sql/mgmt/v3.0/sql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/sql/mgmt/v3.0/sql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sql/mgmt/v4.0/sql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/sql/mgmt/v4.0/sql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sqlvirtualmachine/mgmt/2017-03-01-preview/sqlvirtualmachine` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/sqlvirtualmachine/mgmt/2017-03-01-preview/sqlvirtualmachine/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/storage/mgmt/2015-05-01-preview/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/storage/mgmt/2015-05-01-preview/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/storage/mgmt/2018-03-01-preview/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/storage/mgmt/2018-03-01-preview/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/storage/mgmt/2018-07-01-preview/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/storage/mgmt/2018-07-01-preview/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/storage/mgmt/2020-08-01-preview/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/storage/mgmt/2020-08-01-preview/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/storagecache/mgmt/2019-08-01-preview/storagecache` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/storagecache/mgmt/2019-08-01-preview/storagecache/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/storagepool/mgmt/2020-03-15-preview/storagepool` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/storagepool/mgmt/2020-03-15-preview/storagepool/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/streamanalytics/mgmt/2020-03-01-preview/streamanalytics` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/streamanalytics/mgmt/2020-03-01-preview/streamanalytics/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/subscription/mgmt/2017-11-01-preview/subscription` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/subscription/mgmt/2017-11-01-preview/subscription/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/subscription/mgmt/2018-03-01-preview/subscription` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/subscription/mgmt/2018-03-01-preview/subscription/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/subscription/mgmt/2019-10-01-preview/subscription` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/subscription/mgmt/2019-10-01-preview/subscription/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/support/mgmt/2019-05-01-preview/support` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/support/mgmt/2019-05-01-preview/support/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/synapse/2019-06-01-preview/artifacts` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/synapse/2019-06-01-preview/artifacts/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/synapse/mgmt/2019-06-01-preview/synapse` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/synapse/mgmt/2019-06-01-preview/synapse/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/timeseriesinsights/mgmt/2017-02-28-preview/timeseriesinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/timeseriesinsights/mgmt/2017-02-28-preview/timeseriesinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/timeseriesinsights/mgmt/2018-08-15-preview/timeseriesinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/timeseriesinsights/mgmt/2018-08-15-preview/timeseriesinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/virtualmachineimagebuilder/mgmt/2018-02-01-preview/virtualmachineimagebuilder` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/virtualmachineimagebuilder/mgmt/2018-02-01-preview/virtualmachineimagebuilder/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/virtualmachineimagebuilder/mgmt/2019-02-01-preview/virtualmachineimagebuilder` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/virtualmachineimagebuilder/mgmt/2019-02-01-preview/virtualmachineimagebuilder/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/virtualmachineimagebuilder/mgmt/2019-05-01-preview/virtualmachineimagebuilder` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/virtualmachineimagebuilder/mgmt/2019-05-01-preview/virtualmachineimagebuilder/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/visualstudio/mgmt/2014-04-01-preview/visualstudio` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/visualstudio/mgmt/2014-04-01-preview/visualstudio/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/web/mgmt/2015-08-01-preview/web` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/web/mgmt/2015-08-01-preview/web/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/windowsesu/2019-09-16-preview/windowsesu` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/preview/windowsesu/2019-09-16-preview/windowsesu/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/privatedns/mgmt/2018-09-01/privatedns` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/privatedns/mgmt/2018-09-01/privatedns/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/provisioningservices/mgmt/2017-11-15/iothub` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/provisioningservices/mgmt/2017-11-15/iothub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/provisioningservices/mgmt/2018-01-22/iothub` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/provisioningservices/mgmt/2018-01-22/iothub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2016-08-10/siterecovery` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/recoveryservices/mgmt/2016-08-10/siterecovery/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2018-01-10/siterecovery` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/recoveryservices/mgmt/2018-01-10/siterecovery/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2018-07-10/siterecovery` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/recoveryservices/mgmt/2018-07-10/siterecovery/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2020-02-02/backup` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/recoveryservices/mgmt/2020-02-02/backup/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redhatopenshift/mgmt/2020-04-30/redhatopenshift` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/redhatopenshift/mgmt/2020-04-30/redhatopenshift/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2016-04-01/redis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/redis/mgmt/2016-04-01/redis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2017-02-01/redis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/redis/mgmt/2017-02-01/redis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2017-10-01/redis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/redis/mgmt/2017-10-01/redis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2018-03-01/redis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/redis/mgmt/2018-03-01/redis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2020-06-01/redis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/redis/mgmt/2020-06-01/redis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redisenterprise/mgmt/2021-03-01/redisenterprise` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/redisenterprise/mgmt/2021-03-01/redisenterprise/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/relay/mgmt/2016-07-01/relay` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/relay/mgmt/2016-07-01/relay/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/relay/mgmt/2017-04-01/relay` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/relay/mgmt/2017-04-01/relay/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/reservations/mgmt/2017-11-01/reservations` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/reservations/mgmt/2017-11-01/reservations/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resourcemover/mgmt/2021-01-01/resourcemover` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/resourcemover/mgmt/2021-01-01/resourcemover/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2015-11-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/resources/mgmt/2015-11-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2016-02-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/resources/mgmt/2016-02-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2016-07-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/resources/mgmt/2016-07-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2016-09-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/resources/mgmt/2016-09-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2017-05-10/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/resources/mgmt/2017-05-10/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2017-09-01/managedapplications` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/resources/mgmt/2017-09-01/managedapplications/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2018-02-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/resources/mgmt/2018-02-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2018-05-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/resources/mgmt/2018-05-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2018-06-01/managedapplications` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/resources/mgmt/2018-06-01/managedapplications/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-03-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/resources/mgmt/2019-03-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-05-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/resources/mgmt/2019-05-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-07-01/features` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/resources/mgmt/2019-07-01/features/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-07-01/managedapplications` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/resources/mgmt/2019-07-01/managedapplications/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-10-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/resources/mgmt/2019-10-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-11-01/managementgroups` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/resources/mgmt/2019-11-01/managementgroups/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2020-02-01/managementgroups` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/resources/mgmt/2020-02-01/managementgroups/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2020-05-01/managementgroups` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/resources/mgmt/2020-05-01/managementgroups/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2020-06-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/resources/mgmt/2020-06-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2020-10-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/resources/mgmt/2020-10-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/scheduler/mgmt/2016-03-01/scheduler` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/scheduler/mgmt/2016-03-01/scheduler/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/search/mgmt/2015-08-19/search` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/search/mgmt/2015-08-19/search/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/search/mgmt/2020-03-13/search` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/search/mgmt/2020-03-13/search/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/search/mgmt/2020-08-01/search` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/search/mgmt/2020-08-01/search/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/servicebus/mgmt/2015-08-01/servicebus` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/servicebus/mgmt/2015-08-01/servicebus/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/servicebus/mgmt/2017-04-01/servicebus` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/servicebus/mgmt/2017-04-01/servicebus/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/servicefabric/mgmt/2016-09-01/servicefabric` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/servicefabric/mgmt/2016-09-01/servicefabric/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/servicefabric/mgmt/2019-03-01/servicefabric` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/servicefabric/mgmt/2019-03-01/servicefabric/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/signalr/mgmt/2018-10-01/signalr` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/signalr/mgmt/2018-10-01/signalr/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/signalr/mgmt/2020-05-01/signalr` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/signalr/mgmt/2020-05-01/signalr/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/sql/mgmt/2014-04-01/sql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/sql/mgmt/2014-04-01/sql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2015-06-15/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/storage/mgmt/2015-06-15/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2016-01-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/storage/mgmt/2016-01-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2016-05-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/storage/mgmt/2016-05-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2016-12-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/storage/mgmt/2016-12-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2017-06-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/storage/mgmt/2017-06-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2017-10-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/storage/mgmt/2017-10-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2018-02-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/storage/mgmt/2018-02-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2018-11-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/storage/mgmt/2018-11-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2019-04-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/storage/mgmt/2019-04-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2019-06-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/storage/mgmt/2019-06-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2021-01-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/storage/mgmt/2021-01-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagecache/mgmt/2019-11-01/storagecache` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/storagecache/mgmt/2019-11-01/storagecache/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagecache/mgmt/2020-03-01/storagecache` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/storagecache/mgmt/2020-03-01/storagecache/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagecache/mgmt/2020-10-01/storagecache` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/storagecache/mgmt/2020-10-01/storagecache/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagecache/mgmt/2021-03-01/storagecache` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/storagecache/mgmt/2021-03-01/storagecache/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2018-04-02/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/storagesync/mgmt/2018-04-02/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2018-07-01/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/storagesync/mgmt/2018-07-01/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2018-10-01/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/storagesync/mgmt/2018-10-01/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2019-02-01/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/storagesync/mgmt/2019-02-01/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2019-06-01/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/storagesync/mgmt/2019-06-01/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2019-10-01/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/storagesync/mgmt/2019-10-01/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2020-03-01/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/storagesync/mgmt/2020-03-01/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storsimple1200series/mgmt/2016-10-01/storsimple` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/storsimple1200series/mgmt/2016-10-01/storsimple/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storsimple8000series/mgmt/2017-06-01/storsimple` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/storsimple8000series/mgmt/2017-06-01/storsimple/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/streamanalytics/mgmt/2016-03-01/streamanalytics` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/streamanalytics/mgmt/2016-03-01/streamanalytics/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/subscription/mgmt/2020-09-01/subscription` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/subscription/mgmt/2020-09-01/subscription/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/support/mgmt/2020-04-01/support` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/support/mgmt/2020-04-01/support/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/synapse/mgmt/2020-12-01/synapse` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/synapse/mgmt/2020-12-01/synapse/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/timeseriesinsights/mgmt/2017-11-15/timeseriesinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/timeseriesinsights/mgmt/2017-11-15/timeseriesinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/timeseriesinsights/mgmt/2020-05-15/timeseriesinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/timeseriesinsights/mgmt/2020-05-15/timeseriesinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/virtualmachineimagebuilder/mgmt/2020-02-01/virtualmachineimagebuilder` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/virtualmachineimagebuilder/mgmt/2020-02-01/virtualmachineimagebuilder/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/vmwarecloudsimple/mgmt/2019-04-01/vmwarecloudsimple` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/vmwarecloudsimple/mgmt/2019-04-01/vmwarecloudsimple/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/web/mgmt/2016-09-01/web` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/web/mgmt/2016-09-01/web/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/web/mgmt/2018-02-01/web` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/web/mgmt/2018-02-01/web/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/web/mgmt/2019-08-01/web` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/web/mgmt/2019-08-01/web/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/web/mgmt/2020-06-01/web` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/web/mgmt/2020-06-01/web/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/web/mgmt/2020-09-01/web` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/web/mgmt/2020-09-01/web/CHANGELOG.md) |

### Breaking Changes

| Package Path | Changelog |
| :--- | :---: |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2020-12-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/compute/mgmt/2020-12-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datafactory/mgmt/2018-06-01/datafactory` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/datafactory/mgmt/2018-06-01/datafactory/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/guestconfiguration/mgmt/2020-06-25/guestconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v53.0.0/services/guestconfiguration/mgmt/2020-06-25/guestconfiguration/CHANGELOG.md) |

## `v52.6.0`

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/changeanalysis/mgmt/2021-04-01/changeanalysis`
- `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2021-03-01/compute`
- `github.com/Azure/azure-sdk-for-go/services/databoxedge/mgmt/2020-12-01/databoxedge`
- `github.com/Azure/azure-sdk-for-go/services/delegatednetwork/mgmt/2021-03-15/delegatednetwork`
- `github.com/Azure/azure-sdk-for-go/services/preview/extendedlocation/mgmt/2021-03-15-preview/extendedlocation`

### Updated Packages

| Package Path | Changelog |
| :--- | :---: |
| `github.com/Azure/azure-sdk-for-go/services/preview/authorization/mgmt/2020-04-01-preview/authorization` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v52.6.0/services/preview/authorization/mgmt/2020-04-01-preview/authorization/CHANGELOG.md) |

### Breaking Changes

| Package Path | Changelog |
| :--- | :---: |
| `github.com/Azure/azure-sdk-for-go/services/preview/deviceupdate/mgmt/2020-03-01-preview/deviceupdate` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v52.6.0/services/preview/deviceupdate/mgmt/2020-03-01-preview/deviceupdate/CHANGELOG.md) |

## `v52.5.0`

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2020-12-01/apimanagement`
- `github.com/Azure/azure-sdk-for-go/services/communication/mgmt/2020-08-20/communication`
- `github.com/Azure/azure-sdk-for-go/services/datadog/mgmt/2021-03-01/datadog`
- `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-12-01/netapp`

## `v52.4.0`

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/digitaltwins/mgmt/2020-12-01/digitaltwins`
- `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-11-01/network`
- `github.com/Azure/azure-sdk-for-go/services/preview/kubernetesconfiguration/mgmt/2020-07-01-preview/kubernetesconfiguration`
- `github.com/Azure/azure-sdk-for-go/services/storagecache/mgmt/2021-03-01/storagecache`

### Breaking Changes

| Package Path | Changelog |
| :--- | :---: |
| `github.com/Azure/azure-sdk-for-go/services/preview/datadog/mgmt/2020-02-01-preview/datadog` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v52.4.0/services/preview/datadog/mgmt/2020-02-01-preview/datadog/CHANGELOG.md) |

## `v52.3.0`

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2021-02-01/containerservice`
- `github.com/Azure/azure-sdk-for-go/services/hybridkubernetes/mgmt/2021-03-01/hybridkubernetes`

### Updated Packages

| Package Path | Changelog |
| :--- | :---: |
| `github.com/Azure/azure-sdk-for-go/services/preview/postgresql/mgmt/2020-02-14-preview/postgresqlflexibleservers` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v52.3.0/services/preview/postgresql/mgmt/2020-02-14-preview/postgresqlflexibleservers/CHANGELOG.md) |

## `v52.2.0`

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/preview/deviceupdate/mgmt/2020-03-01-preview/deviceupdate`

## `v52.1.0`

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2021-03-01-preview/documentdb`

## `v52.0.0`

NOTE: We have switched the uuid package from `github.com/satori/go.uuid` to `github.com/gofrs/uuid` as announced in [this issue](https://github.com/Azure/azure-sdk-for-go/issues/14283)

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/containerinstance/mgmt/2020-11-01/containerinstance`
- `github.com/Azure/azure-sdk-for-go/services/kubernetesconfiguration/mgmt/2021-03-01/kubernetesconfiguration`
- `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-11-01/netapp`
- `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2021-01-01/storage`

### Updated Packages

| Package Path | Changelog |
| :--- | :---: |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2016-07-07/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v52.0.0/services/apimanagement/mgmt/2016-07-07/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2016-10-10/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v52.0.0/services/apimanagement/mgmt/2016-10-10/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/consumption/mgmt/2019-10-01/consumption` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v52.0.0/services/consumption/mgmt/2019-10-01/consumption/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/apimanagement/ctrl/2017-03-01/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v52.0.0/services/preview/apimanagement/ctrl/2017-03-01/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/appplatform/mgmt/2020-11-01-preview/appplatform` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v52.0.0/services/preview/appplatform/mgmt/2020-11-01-preview/appplatform/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/streamanalytics/mgmt/2020-03-01-preview/streamanalytics` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v52.0.0/services/preview/streamanalytics/mgmt/2020-03-01-preview/streamanalytics/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/synapse/mgmt/2019-06-01-preview/synapse` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v52.0.0/services/preview/synapse/mgmt/2019-06-01-preview/synapse/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/synapse/mgmt/2020-12-01/synapse` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v52.0.0/services/synapse/mgmt/2020-12-01/synapse/CHANGELOG.md) |

### Breaking Changes

| Package Path | Changelog |
| :--- | :---: |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2015-06-01/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v52.0.0/services/cdn/mgmt/2015-06-01/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2016-04-02/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v52.0.0/services/cdn/mgmt/2016-04-02/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/logic/mgmt/2016-06-01/logic` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v52.0.0/services/logic/mgmt/2016-06-01/logic/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mediaservices/mgmt/2020-05-01/media` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v52.0.0/services/mediaservices/mgmt/2020-05-01/media/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cognitiveservices/v1.0_preview.1/translatortext` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v52.0.0/services/preview/cognitiveservices/v1.0_preview.1/translatortext/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/logic/mgmt/2018-07-01-preview/logic` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v52.0.0/services/preview/logic/mgmt/2018-07-01-preview/logic/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mixedreality/mgmt/2021-03-01-preview/mixedreality` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v52.0.0/services/preview/mixedreality/mgmt/2021-03-01-preview/mixedreality/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2020-02-01/managementgroups` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v52.0.0/services/resources/mgmt/2020-02-01/managementgroups/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2020-05-01/managementgroups` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v52.0.0/services/resources/mgmt/2020-05-01/managementgroups/CHANGELOG.md) |

## `v51.3.0`

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/cosmos-db/mgmt/2021-01-15/documentdb`
- `github.com/Azure/azure-sdk-for-go/services/monitor/mgmt/2020-10-01/insights`
- `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2020-10-15-preview/eventgrid`
- `github.com/Azure/azure-sdk-for-go/services/preview/mixedreality/mgmt/2021-03-01-preview/mixedreality`
- `github.com/Azure/azure-sdk-for-go/services/web/mgmt/2020-09-01/web`

## `v51.2.0`

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2020-12-01/compute`
- `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2020-09-01-preview/documentdb`
- `github.com/Azure/azure-sdk-for-go/services/preview/sql/mgmt/v4.0/sql`
- `github.com/Azure/azure-sdk-for-go/services/preview/synapse/2020-08-01-preview/accesscontrol`
- `github.com/Azure/azure-sdk-for-go/services/redisenterprise/mgmt/2021-03-01/redisenterprise`
- `github.com/Azure/azure-sdk-for-go/services/resourcemover/mgmt/2021-01-01/resourcemover`

### Updated Packages

| Package Path | Changelog |
| :--- | :---: |
| `github.com/Azure/azure-sdk-for-go/services/datafactory/mgmt/2018-06-01/datafactory` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v51.2.0/services/datafactory/mgmt/2018-06-01/datafactory/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iotcentral/mgmt/2018-09-01/iotcentral` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v51.2.0/services/iotcentral/mgmt/2018-09-01/iotcentral/CHANGELOG.md) |

## `v51.1.0`

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/mixedreality/mgmt/2021-01-01/mixedreality`
- `github.com/Azure/azure-sdk-for-go/services/preview/compute/mgmt/2020-10-01-preview/compute`
- `github.com/Azure/azure-sdk-for-go/services/storagecache/mgmt/2020-10-01/storagecache`

### Breaking Changes

| Package Path | Changelog |
| :--- | :---: |
| `github.com/Azure/azure-sdk-for-go/services/preview/storage/mgmt/2020-08-01-preview/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v51.1.0/services/preview/storage/mgmt/2020-08-01-preview/storage/CHANGELOG.md) |

## `v51.0.0`

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/preview/purview/mgmt/2020-12-01-preview/purview`
- `github.com/Azure/azure-sdk-for-go/services/preview/quantum/mgmt/2019-11-04-preview/quantum`
- `github.com/Azure/azure-sdk-for-go/services/synapse/mgmt/2020-12-01/synapse`

### Breaking Changes

| Package Path | Changelog |
| :--- | :---: |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2020-09-01/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v51.0.0/services/cdn/mgmt/2020-09-01/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/securityinsight/mgmt/2019-01-01-preview/securityinsight` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v51.0.0/services/preview/securityinsight/mgmt/2019-01-01-preview/securityinsight/CHANGELOG.md) |

## `v50.2.0`

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/preview/reservations/mgmt/2020-10-25/reservations`

### Updated Packages

| Package Path | Changelog |
| :--- | :---: |
| `github.com/Azure/azure-sdk-for-go/services/hdinsight/mgmt/2018-06-01/hdinsight` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.2.0/services/hdinsight/mgmt/2018-06-01/hdinsight/CHANGELOG.md) |

## `v50.1.0`

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/confluent/mgmt/2020-03-01/confluent`
- `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-08-01/network`
- `github.com/Azure/azure-sdk-for-go/services/preview/authorization/mgmt/2020-04-01-preview/authorization`

### Breaking Changes

| Package Path | Changelog |
| :--- | :---: |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2020-06-01-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.1.0/services/preview/cosmos-db/mgmt/2020-06-01-preview/documentdb/CHANGELOG.md) |

## `v50.0.0`

NOTE: Due to the changes requested in [this issue](https://github.com/Azure/azure-sdk-for-go/issues/14010), we changed the properties and functions of all future types, which does not affect their functionality and usage, but leads to a very long list of breaking changes. This change requires the latest version of `github.com/Azure/go-autorest/autorest v0.11.15` to work properly.

### Renamed Packages

- `github.com/Azure/azure-sdk-for-go/profiles/2020-09-01/compute` renamed to `github.com/Azure/azure-sdk-for-go/profiles/2020-09-01/compute/mgmt/compute` to align other naming pattern of the profile packages

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/healthbot/mgmt/2020-12-08/healthbot`
- `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2020-10-01/resources`

### Removed Packages

- `github.com/Azure/azure-sdk-for-go/services/preview/avs/mgmt/2019-08-09-preview/avs`

### Breaking Changes

| Package Path | Changelog |
| :--- | :---: |
| `github.com/Azure/azure-sdk-for-go/services/analysisservices/mgmt/2016-05-16/analysisservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/analysisservices/mgmt/2016-05-16/analysisservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/analysisservices/mgmt/2017-07-14/analysisservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/analysisservices/mgmt/2017-07-14/analysisservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/analysisservices/mgmt/2017-08-01/analysisservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/analysisservices/mgmt/2017-08-01/analysisservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2016-07-07/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/apimanagement/mgmt/2016-07-07/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2016-10-10/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/apimanagement/mgmt/2016-10-10/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2017-03-01/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/apimanagement/mgmt/2017-03-01/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2018-01-01/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/apimanagement/mgmt/2018-01-01/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2019-01-01/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/apimanagement/mgmt/2019-01-01/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2019-12-01/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/apimanagement/mgmt/2019-12-01/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/appconfiguration/mgmt/2019-10-01/appconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/appconfiguration/mgmt/2019-10-01/appconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/appconfiguration/mgmt/2020-06-01/appconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/appconfiguration/mgmt/2020-06-01/appconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/appplatform/mgmt/2020-07-01/appplatform` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/appplatform/mgmt/2020-07-01/appplatform/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/automation/mgmt/2015-10-31/automation` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/automation/mgmt/2015-10-31/automation/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/avs/mgmt/2020-03-20/avs` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/avs/mgmt/2020-03-20/avs/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2015-12-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/batch/mgmt/2015-12-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2017-01-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/batch/mgmt/2017-01-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2017-05-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/batch/mgmt/2017-05-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2017-09-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/batch/mgmt/2017-09-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2018-12-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/batch/mgmt/2018-12-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2019-04-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/batch/mgmt/2019-04-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2019-08-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/batch/mgmt/2019-08-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2020-03-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/batch/mgmt/2020-03-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2020-05-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/batch/mgmt/2020-05-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2020-09-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/batch/mgmt/2020-09-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batchai/mgmt/2018-03-01/batchai` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/batchai/mgmt/2018-03-01/batchai/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batchai/mgmt/2018-05-01/batchai` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/batchai/mgmt/2018-05-01/batchai/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2015-06-01/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/cdn/mgmt/2015-06-01/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2016-04-02/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/cdn/mgmt/2016-04-02/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2016-10-02/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/cdn/mgmt/2016-10-02/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2017-04-02/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/cdn/mgmt/2017-04-02/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2017-10-12/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/cdn/mgmt/2017-10-12/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2019-04-15/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/cdn/mgmt/2019-04-15/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2019-06-15/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/cdn/mgmt/2019-06-15/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2020-04-15/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/cdn/mgmt/2020-04-15/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2020-09-01/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/cdn/mgmt/2020-09-01/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2015-06-15/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/compute/mgmt/2015-06-15/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2016-03-30/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/compute/mgmt/2016-03-30/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2017-03-30/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/compute/mgmt/2017-03-30/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2017-12-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/compute/mgmt/2017-12-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2018-04-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/compute/mgmt/2018-04-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2018-06-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/compute/mgmt/2018-06-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2018-10-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/compute/mgmt/2018-10-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-03-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/compute/mgmt/2019-03-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-07-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/compute/mgmt/2019-07-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-12-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/compute/mgmt/2019-12-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2020-06-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/compute/mgmt/2020-06-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2020-06-30/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/compute/mgmt/2020-06-30/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerinstance/mgmt/2018-04-01/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerinstance/mgmt/2018-04-01/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerinstance/mgmt/2018-06-01/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerinstance/mgmt/2018-06-01/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerinstance/mgmt/2018-09-01/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerinstance/mgmt/2018-09-01/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerinstance/mgmt/2018-10-01/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerinstance/mgmt/2018-10-01/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerinstance/mgmt/2019-12-01/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerinstance/mgmt/2019-12-01/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2017-03-01/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerregistry/mgmt/2017-03-01/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2017-10-01/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerregistry/mgmt/2017-10-01/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2018-09-01/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerregistry/mgmt/2018-09-01/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2019-04-01/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerregistry/mgmt/2019-04-01/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2019-05-01/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerregistry/mgmt/2019-05-01/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2016-03-30/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerservice/mgmt/2016-03-30/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2016-09-30/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerservice/mgmt/2016-09-30/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2017-01-31/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerservice/mgmt/2017-01-31/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2017-07-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerservice/mgmt/2017-07-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2017-08-31/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerservice/mgmt/2017-08-31/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2017-09-30/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerservice/mgmt/2017-09-30/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2018-03-31/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerservice/mgmt/2018-03-31/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2019-04-30/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerservice/mgmt/2019-04-30/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2019-06-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerservice/mgmt/2019-06-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2019-08-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerservice/mgmt/2019-08-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2019-10-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerservice/mgmt/2019-10-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2019-11-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerservice/mgmt/2019-11-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-01-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerservice/mgmt/2020-01-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-02-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerservice/mgmt/2020-02-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-03-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerservice/mgmt/2020-03-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-04-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerservice/mgmt/2020-04-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-06-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerservice/mgmt/2020-06-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-07-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerservice/mgmt/2020-07-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-09-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerservice/mgmt/2020-09-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-11-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerservice/mgmt/2020-11-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-12-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/containerservice/mgmt/2020-12-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cosmos-db/mgmt/2015-04-08/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/cosmos-db/mgmt/2015-04-08/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cosmos-db/mgmt/2019-08-01/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/cosmos-db/mgmt/2019-08-01/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/customerinsights/mgmt/2017-01-01/customerinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/customerinsights/mgmt/2017-01-01/customerinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/customerinsights/mgmt/2017-04-26/customerinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/customerinsights/mgmt/2017-04-26/customerinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databox/mgmt/2018-01-01/databox` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/databox/mgmt/2018-01-01/databox/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databox/mgmt/2019-09-01/databox` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/databox/mgmt/2019-09-01/databox/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databox/mgmt/2020-04-01/databox` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/databox/mgmt/2020-04-01/databox/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databox/mgmt/2020-11-01/databox` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/databox/mgmt/2020-11-01/databox/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databoxedge/mgmt/2019-03-01/databoxedge` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/databoxedge/mgmt/2019-03-01/databoxedge/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databoxedge/mgmt/2019-07-01/databoxedge` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/databoxedge/mgmt/2019-07-01/databoxedge/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databoxedge/mgmt/2019-08-01/databoxedge` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/databoxedge/mgmt/2019-08-01/databoxedge/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databricks/mgmt/2018-04-01/databricks` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/databricks/mgmt/2018-04-01/databricks/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datacatalog/mgmt/2016-03-30/datacatalog` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/datacatalog/mgmt/2016-03-30/datacatalog/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datafactory/mgmt/2018-06-01/datafactory` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/datafactory/mgmt/2018-06-01/datafactory/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datalake/analytics/mgmt/2016-11-01/account` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/datalake/analytics/mgmt/2016-11-01/account/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datalake/store/mgmt/2016-11-01/account` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/datalake/store/mgmt/2016-11-01/account/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datamigration/mgmt/2018-04-19/datamigration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/datamigration/mgmt/2018-04-19/datamigration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datashare/mgmt/2019-11-01/datashare` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/datashare/mgmt/2019-11-01/datashare/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/devspaces/mgmt/2019-04-01/devspaces` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/devspaces/mgmt/2019-04-01/devspaces/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/devtestlabs/mgmt/2016-05-15/dtl` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/devtestlabs/mgmt/2016-05-15/dtl/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/devtestlabs/mgmt/2018-09-15/dtl` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/devtestlabs/mgmt/2018-09-15/dtl/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/digitaltwins/mgmt/2020-10-31/digitaltwins` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/digitaltwins/mgmt/2020-10-31/digitaltwins/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/dns/mgmt/2016-04-01/dns` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/dns/mgmt/2016-04-01/dns/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/dns/mgmt/2017-09-01/dns` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/dns/mgmt/2017-09-01/dns/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/dns/mgmt/2017-10-01/dns` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/dns/mgmt/2017-10-01/dns/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/dns/mgmt/2018-05-01/dns` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/dns/mgmt/2018-05-01/dns/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/domainservices/mgmt/2017-01-01/aad` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/domainservices/mgmt/2017-01-01/aad/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/domainservices/mgmt/2017-06-01/aad` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/domainservices/mgmt/2017-06-01/aad/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/domainservices/mgmt/2020-01-01/aad` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/domainservices/mgmt/2020-01-01/aad/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/eventgrid/mgmt/2018-01-01/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/eventgrid/mgmt/2018-01-01/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/eventgrid/mgmt/2019-01-01/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/eventgrid/mgmt/2019-01-01/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/eventgrid/mgmt/2019-06-01/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/eventgrid/mgmt/2019-06-01/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/eventgrid/mgmt/2020-06-01/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/eventgrid/mgmt/2020-06-01/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/eventhub/mgmt/2015-08-01/eventhub` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/eventhub/mgmt/2015-08-01/eventhub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/eventhub/mgmt/2017-04-01/eventhub` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/eventhub/mgmt/2017-04-01/eventhub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2019-04-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/frontdoor/mgmt/2019-04-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2019-05-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/frontdoor/mgmt/2019-05-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2019-10-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/frontdoor/mgmt/2019-10-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2019-11-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/frontdoor/mgmt/2019-11-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2020-01-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/frontdoor/mgmt/2020-01-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2020-04-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/frontdoor/mgmt/2020-04-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2020-05-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/frontdoor/mgmt/2020-05-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/guestconfiguration/mgmt/2020-06-25/guestconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/guestconfiguration/mgmt/2020-06-25/guestconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/hdinsight/mgmt/2018-06-01/hdinsight` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/hdinsight/mgmt/2018-06-01/hdinsight/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/healthcareapis/mgmt/2019-09-16/healthcareapis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/healthcareapis/mgmt/2019-09-16/healthcareapis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/healthcareapis/mgmt/2020-03-15/healthcareapis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/healthcareapis/mgmt/2020-03-15/healthcareapis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/healthcareapis/mgmt/2020-03-30/healthcareapis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/healthcareapis/mgmt/2020-03-30/healthcareapis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/hybridcompute/mgmt/2019-12-12/hybridcompute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/hybridcompute/mgmt/2019-12-12/hybridcompute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/hybridcompute/mgmt/2020-08-02/hybridcompute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/hybridcompute/mgmt/2020-08-02/hybridcompute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/hybriddatamanager/mgmt/2016-06-01/hybriddata` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/hybriddatamanager/mgmt/2016-06-01/hybriddata/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/hybriddatamanager/mgmt/2019-06-01/hybriddata` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/hybriddatamanager/mgmt/2019-06-01/hybriddata/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iotcentral/mgmt/2018-09-01/iotcentral` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/iotcentral/mgmt/2018-09-01/iotcentral/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iothub/mgmt/2016-02-03/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/iothub/mgmt/2016-02-03/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iothub/mgmt/2017-01-19/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/iothub/mgmt/2017-01-19/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iothub/mgmt/2017-07-01/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/iothub/mgmt/2017-07-01/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iothub/mgmt/2018-01-22/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/iothub/mgmt/2018-01-22/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iothub/mgmt/2018-04-01/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/iothub/mgmt/2018-04-01/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iothub/mgmt/2020-03-01/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/iothub/mgmt/2020-03-01/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/keyvault/mgmt/2016-10-01/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/keyvault/mgmt/2016-10-01/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/keyvault/mgmt/2018-02-14/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/keyvault/mgmt/2018-02-14/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/keyvault/mgmt/2019-09-01/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/keyvault/mgmt/2019-09-01/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2019-01-21/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/kusto/mgmt/2019-01-21/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2019-05-15/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/kusto/mgmt/2019-05-15/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2019-09-07/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/kusto/mgmt/2019-09-07/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2019-11-09/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/kusto/mgmt/2019-11-09/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2020-02-15/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/kusto/mgmt/2020-02-15/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2020-06-14/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/kusto/mgmt/2020-06-14/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2020-09-18/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/kusto/mgmt/2020-09-18/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/labservices/mgmt/2018-10-15/labservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/labservices/mgmt/2018-10-15/labservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/logic/mgmt/2019-05-01/logic` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/logic/mgmt/2019-05-01/logic/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearning/mgmt/2017-01-01/webservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/machinelearning/mgmt/2017-01-01/webservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2018-11-19/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/machinelearningservices/mgmt/2018-11-19/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2019-05-01/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/machinelearningservices/mgmt/2019-05-01/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2019-06-01/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/machinelearningservices/mgmt/2019-06-01/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2019-11-01/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/machinelearningservices/mgmt/2019-11-01/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2020-01-01/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/machinelearningservices/mgmt/2020-01-01/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2020-03-01/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/machinelearningservices/mgmt/2020-03-01/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2020-04-01/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/machinelearningservices/mgmt/2020-04-01/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/managedservices/mgmt/2019-06-01/managedservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/managedservices/mgmt/2019-06-01/managedservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mariadb/mgmt/2018-06-01/mariadb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/mariadb/mgmt/2018-06-01/mariadb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mariadb/mgmt/2020-01-01/mariadb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/mariadb/mgmt/2020-01-01/mariadb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mediaservices/mgmt/2018-07-01/media` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/mediaservices/mgmt/2018-07-01/media/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mediaservices/mgmt/2020-05-01/media` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/mediaservices/mgmt/2020-05-01/media/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mysql/mgmt/2017-12-01/mysql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/mysql/mgmt/2017-12-01/mysql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mysql/mgmt/2020-01-01/mysql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/mysql/mgmt/2020-01-01/mysql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2019-05-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/netapp/mgmt/2019-05-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2019-06-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/netapp/mgmt/2019-06-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2019-07-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/netapp/mgmt/2019-07-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2019-08-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/netapp/mgmt/2019-08-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2019-10-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/netapp/mgmt/2019-10-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2019-11-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/netapp/mgmt/2019-11-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-02-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/netapp/mgmt/2020-02-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-03-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/netapp/mgmt/2020-03-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-05-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/netapp/mgmt/2020-05-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-06-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/netapp/mgmt/2020-06-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-07-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/netapp/mgmt/2020-07-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-08-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/netapp/mgmt/2020-08-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-09-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/netapp/mgmt/2020-09-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2015-06-15/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2015-06-15/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2016-03-30/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2016-03-30/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2016-06-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2016-06-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2016-09-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2016-09-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2016-12-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2016-12-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-03-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2017-03-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-06-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2017-06-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-08-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2017-08-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-09-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2017-09-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-10-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2017-10-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-11-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2017-11-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-01-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2018-01-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-02-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2018-02-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-04-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2018-04-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-06-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2018-06-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-07-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2018-07-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-08-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2018-08-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-10-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2018-10-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-11-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2018-11-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-12-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2018-12-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-02-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2019-02-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-04-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2019-04-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2019-06-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-07-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2019-07-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-08-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2019-08-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-09-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2019-09-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-11-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2019-11-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-12-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2019-12-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-03-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2020-03-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-04-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2020-04-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-05-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2020-05-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-06-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2020-06-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-07-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/network/mgmt/2020-07-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/notificationhubs/mgmt/2014-09-01/notificationhubs` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/notificationhubs/mgmt/2014-09-01/notificationhubs/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/notificationhubs/mgmt/2016-03-01/notificationhubs` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/notificationhubs/mgmt/2016-03-01/notificationhubs/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/notificationhubs/mgmt/2017-04-01/notificationhubs` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/notificationhubs/mgmt/2017-04-01/notificationhubs/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/operationalinsights/mgmt/2020-08-01/operationalinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/operationalinsights/mgmt/2020-08-01/operationalinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/operationalinsights/mgmt/2020-10-01/operationalinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/operationalinsights/mgmt/2020-10-01/operationalinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/postgresql/mgmt/2017-12-01/postgresql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/postgresql/mgmt/2017-12-01/postgresql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/postgresql/mgmt/2020-01-01/postgresql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/postgresql/mgmt/2020-01-01/postgresql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/powerbidedicated/mgmt/2017-10-01/powerbidedicated` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/powerbidedicated/mgmt/2017-10-01/powerbidedicated/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/powerbiembedded/mgmt/2016-01-29/powerbiembedded` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/powerbiembedded/mgmt/2016-01-29/powerbiembedded/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/addons/mgmt/2017-05-15/addons` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/addons/mgmt/2017-05-15/addons/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/addons/mgmt/2018-03-01/addons` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/addons/mgmt/2018-03-01/addons/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/alertsmanagement/mgmt/2019-06-01-preview/alertsmanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/alertsmanagement/mgmt/2019-06-01-preview/alertsmanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/apimanagement/ctrl/2017-03-01/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/apimanagement/ctrl/2017-03-01/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/apimanagement/mgmt/2018-06-01-preview/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/apimanagement/mgmt/2018-06-01-preview/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/apimanagement/mgmt/2019-12-01-preview/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/apimanagement/mgmt/2019-12-01-preview/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/appconfiguration/mgmt/2019-02-01-preview/appconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/appconfiguration/mgmt/2019-02-01-preview/appconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/appconfiguration/mgmt/2019-11-01-preview/appconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/appconfiguration/mgmt/2019-11-01-preview/appconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/appconfiguration/mgmt/2020-07-01-preview/appconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/appconfiguration/mgmt/2020-07-01-preview/appconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/appplatform/mgmt/2019-05-01-preview/appplatform` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/appplatform/mgmt/2019-05-01-preview/appplatform/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/appplatform/mgmt/2020-11-01-preview/appplatform` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/appplatform/mgmt/2020-11-01-preview/appplatform/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/automanage/mgmt/2020-06-30-preview/automanage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/automanage/mgmt/2020-06-30-preview/automanage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/automation/mgmt/2017-05-15-preview/automation` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/automation/mgmt/2017-05-15-preview/automation/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/automation/mgmt/2018-01-15-preview/automation` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/automation/mgmt/2018-01-15-preview/automation/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/automation/mgmt/2018-06-30-preview/automation` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/automation/mgmt/2018-06-30-preview/automation/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/avs/mgmt/2020-07-17-preview/avs` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/avs/mgmt/2020-07-17-preview/avs/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/baremetalinfrastructure/mgmt/2020-08-06-preview/baremetalinfrastructure` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/baremetalinfrastructure/mgmt/2020-08-06-preview/baremetalinfrastructure/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/batchai/mgmt/2017-09-01-preview/batchai` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/batchai/mgmt/2017-09-01-preview/batchai/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/billing/mgmt/2018-11-01-preview/billing` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/billing/mgmt/2018-11-01-preview/billing/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/billing/mgmt/2020-05-01-preview/billing` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/billing/mgmt/2020-05-01-preview/billing/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/blockchain/mgmt/2018-06-01-preview/blockchain` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/blockchain/mgmt/2018-06-01-preview/blockchain/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/botservice/mgmt/2018-07-12/botservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/botservice/mgmt/2018-07-12/botservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/communication/mgmt/2020-08-20-preview/communication` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/communication/mgmt/2020-08-20-preview/communication/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/compute/mgmt/2016-04-30-preview/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/compute/mgmt/2016-04-30-preview/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/confluent/mgmt/2020-03-01-preview/confluent` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/confluent/mgmt/2020-03-01-preview/confluent/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerinstance/mgmt/2018-02-01-preview/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/containerinstance/mgmt/2018-02-01-preview/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerregistry/mgmt/2017-06-01-preview/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/containerregistry/mgmt/2017-06-01-preview/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerregistry/mgmt/2018-02-01/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/containerregistry/mgmt/2018-02-01/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerregistry/mgmt/2019-06-01-preview/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/containerregistry/mgmt/2019-06-01-preview/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerregistry/mgmt/2019-12-01-preview/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/containerregistry/mgmt/2019-12-01-preview/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerregistry/mgmt/2020-11-01-preview/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/containerregistry/mgmt/2020-11-01-preview/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerservice/mgmt/2015-11-01-preview/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/containerservice/mgmt/2015-11-01-preview/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerservice/mgmt/2018-08-01-preview/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/containerservice/mgmt/2018-08-01-preview/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerservice/mgmt/2018-09-30-preview/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/containerservice/mgmt/2018-09-30-preview/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerservice/mgmt/2019-02-01-preview/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/containerservice/mgmt/2019-02-01-preview/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerservice/mgmt/2019-09-30-preview/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/containerservice/mgmt/2019-09-30-preview/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerservice/mgmt/2019-10-27-preview/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/containerservice/mgmt/2019-10-27-preview/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2019-08-01-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/cosmos-db/mgmt/2019-08-01-preview/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2019-12-12-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/cosmos-db/mgmt/2019-12-12-preview/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2020-03-01-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/cosmos-db/mgmt/2020-03-01-preview/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2020-04-01-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/cosmos-db/mgmt/2020-04-01-preview/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2020-06-01-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/cosmos-db/mgmt/2020-06-01-preview/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/customproviders/mgmt/2018-09-01-preview/customproviders` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/customproviders/mgmt/2018-09-01-preview/customproviders/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/databoxedge/mgmt/2020-05-01-preview/databoxedge` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/databoxedge/mgmt/2020-05-01-preview/databoxedge/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datadog/mgmt/2020-02-01-preview/datadog` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/datadog/mgmt/2020-02-01-preview/datadog/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datafactory/mgmt/2017-09-01-preview/datafactory` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/datafactory/mgmt/2017-09-01-preview/datafactory/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datalake/analytics/2017-09-01-preview/job` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/datalake/analytics/2017-09-01-preview/job/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datalake/analytics/mgmt/2015-10-01-preview/account` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/datalake/analytics/mgmt/2015-10-01-preview/account/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datalake/store/mgmt/2015-10-01-preview/account` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/datalake/store/mgmt/2015-10-01-preview/account/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datamigration/mgmt/2017-11-15-preview/datamigration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/datamigration/mgmt/2017-11-15-preview/datamigration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datamigration/mgmt/2018-03-31-preview/datamigration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/datamigration/mgmt/2018-03-31-preview/datamigration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datamigration/mgmt/2018-07-15-preview/datamigration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/datamigration/mgmt/2018-07-15-preview/datamigration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datashare/mgmt/2018-11-01-preview/datashare` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/datashare/mgmt/2018-11-01-preview/datashare/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/delegatednetwork/mgmt/2020-08-08-preview/delegatednetwork` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/delegatednetwork/mgmt/2020-08-08-preview/delegatednetwork/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/deploymentmanager/mgmt/2018-09-01-preview/deploymentmanager` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/deploymentmanager/mgmt/2018-09-01-preview/deploymentmanager/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/deploymentmanager/mgmt/2019-11-01-preview/deploymentmanager` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/deploymentmanager/mgmt/2019-11-01-preview/deploymentmanager/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/devops/mgmt/2019-07-01-preview/devops` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/devops/mgmt/2019-07-01-preview/devops/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/devtestlabs/mgmt/2015-05-21-preview/dtl` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/devtestlabs/mgmt/2015-05-21-preview/dtl/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/digitaltwins/mgmt/2020-03-01-preview/digitaltwins` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/digitaltwins/mgmt/2020-03-01-preview/digitaltwins/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/dns/mgmt/2018-03-01-preview/dns` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/dns/mgmt/2018-03-01-preview/dns/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2017-06-15-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/eventgrid/mgmt/2017-06-15-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2017-09-15-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/eventgrid/mgmt/2017-09-15-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2018-05-01-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/eventgrid/mgmt/2018-05-01-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2018-09-15-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/eventgrid/mgmt/2018-09-15-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2019-02-01-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/eventgrid/mgmt/2019-02-01-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2020-01-01-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/eventgrid/mgmt/2020-01-01-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2020-04-01-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/eventgrid/mgmt/2020-04-01-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventhub/mgmt/2018-01-01-preview/eventhub` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/eventhub/mgmt/2018-01-01-preview/eventhub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/frontdoor/mgmt/2018-08-01-preview/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/frontdoor/mgmt/2018-08-01-preview/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/hanaonazure/mgmt/2017-11-03-preview/hanaonazure` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/hanaonazure/mgmt/2017-11-03-preview/hanaonazure/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/hardwaresecuritymodules/mgmt/2018-10-31-preview/hardwaresecuritymodules` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/hardwaresecuritymodules/mgmt/2018-10-31-preview/hardwaresecuritymodules/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/hdinsight/mgmt/2015-03-01-preview/hdinsight` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/hdinsight/mgmt/2015-03-01-preview/hdinsight/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/healthcareapis/mgmt/2018-08-20-preview/healthcareapis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/healthcareapis/mgmt/2018-08-20-preview/healthcareapis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/hybridkubernetes/mgmt/2020-01-01-preview/hybridkubernetes` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/hybridkubernetes/mgmt/2020-01-01-preview/hybridkubernetes/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/hybridnetwork/mgmt/2020-01-01-preview/hybridnetwork` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/hybridnetwork/mgmt/2020-01-01-preview/hybridnetwork/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/iothub/mgmt/2018-12-01-preview/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/iothub/mgmt/2018-12-01-preview/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/iothub/mgmt/2019-03-22-preview/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/iothub/mgmt/2019-03-22-preview/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/iothub/mgmt/2019-07-01-preview/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/iothub/mgmt/2019-07-01-preview/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/iotspaces/mgmt/2017-10-01-preview/iotspaces` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/iotspaces/mgmt/2017-10-01-preview/iotspaces/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/keyvault/mgmt/2020-04-01-preview/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/keyvault/mgmt/2020-04-01-preview/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/keyvault/v7.2-preview/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/keyvault/v7.2-preview/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/kubernetesconfiguration/mgmt/2019-11-01-preview/kubernetesconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/kubernetesconfiguration/mgmt/2019-11-01-preview/kubernetesconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/kusto/mgmt/2018-09-07-preview/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/kusto/mgmt/2018-09-07-preview/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/logic/mgmt/2015-02-01-preview/logic` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/logic/mgmt/2015-02-01-preview/logic/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/machinelearning/mgmt/2016-05-01-preview/webservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/machinelearning/mgmt/2016-05-01-preview/webservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/machinelearning/mgmt/2017-08-01-preview/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/machinelearning/mgmt/2017-08-01-preview/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/machinelearningservices/mgmt/2018-03-01-preview/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/machinelearningservices/mgmt/2018-03-01-preview/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/machinelearningservices/mgmt/2020-02-18-preview/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/machinelearningservices/mgmt/2020-02-18-preview/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/managednetwork/mgmt/2019-06-01-preview/managednetwork` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/managednetwork/mgmt/2019-06-01-preview/managednetwork/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/managedservices/mgmt/2019-04-01/managedservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/managedservices/mgmt/2019-04-01/managedservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mediaservices/mgmt/2018-03-30-preview/media` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/mediaservices/mgmt/2018-03-30-preview/media/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mediaservices/mgmt/2018-06-01-preview/media` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/mediaservices/mgmt/2018-06-01-preview/media/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mediaservices/mgmt/2019-05-01-preview/media` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/mediaservices/mgmt/2019-05-01-preview/media/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/monitor/mgmt/2019-11-01-preview/insights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/monitor/mgmt/2019-11-01-preview/insights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mysql/mgmt/2017-12-01-preview/mysql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/mysql/mgmt/2017-12-01-preview/mysql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mysql/mgmt/2020-07-01-preview/mysqlflexibleservers` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/mysql/mgmt/2020-07-01-preview/mysqlflexibleservers/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/netapp/mgmt/2017-08-15/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/netapp/mgmt/2017-08-15/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/network/mgmt/2015-05-01-preview/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/network/mgmt/2015-05-01-preview/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/operationalinsights/mgmt/2015-11-01-preview/operationalinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/operationalinsights/mgmt/2015-11-01-preview/operationalinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/operationalinsights/mgmt/2020-03-01-preview/operationalinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/operationalinsights/mgmt/2020-03-01-preview/operationalinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/operationsmanagement/mgmt/2015-11-01-preview/operationsmanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/operationsmanagement/mgmt/2015-11-01-preview/operationsmanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/policyinsights/mgmt/2019-10-01-preview/policyinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/policyinsights/mgmt/2019-10-01-preview/policyinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/policyinsights/mgmt/2020-07-01-preview/policyinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/policyinsights/mgmt/2020-07-01-preview/policyinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/postgresql/mgmt/2017-12-01-preview/postgresql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/postgresql/mgmt/2017-12-01-preview/postgresql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/postgresql/mgmt/2020-02-14-preview/postgresqlflexibleservers` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/postgresql/mgmt/2020-02-14-preview/postgresqlflexibleservers/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/postgresql/mgmt/2020-11-05-preview/postgresqlflexibleservers` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/postgresql/mgmt/2020-11-05-preview/postgresqlflexibleservers/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/powerplatform/mgmt/2020-10-30/powerplatform` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/powerplatform/mgmt/2020-10-30/powerplatform/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/provisioningservices/mgmt/2017-08-21-preview/iothub` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/provisioningservices/mgmt/2017-08-21-preview/iothub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/redis/mgmt/2019-07-01-preview/redis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/redis/mgmt/2019-07-01-preview/redis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/redisenterprise/mgmt/2020-10-01-preview/redisenterprise` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/redisenterprise/mgmt/2020-10-01-preview/redisenterprise/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/reservations/mgmt/2018-06-01/reservations` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/reservations/mgmt/2018-06-01/reservations/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/reservations/mgmt/2019-04-01/reservations` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/reservations/mgmt/2019-04-01/reservations/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/reservations/mgmt/2019-07-19-preview/reservations` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/reservations/mgmt/2019-07-19-preview/reservations/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resourcemover/mgmt/2019-10-01-preview/resourcemover` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/resourcemover/mgmt/2019-10-01-preview/resourcemover/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2016-09-01-preview/managedapplications` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/resources/mgmt/2016-09-01-preview/managedapplications/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2018-01-01-preview/managementgroups` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/resources/mgmt/2018-01-01-preview/managementgroups/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2018-03-01-preview/managementgroups` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/resources/mgmt/2018-03-01-preview/managementgroups/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/security/mgmt/v1.0/security` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/security/mgmt/v1.0/security/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/security/mgmt/v2.0/security` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/security/mgmt/v2.0/security/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/security/mgmt/v3.0/security` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/security/mgmt/v3.0/security/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/servicebus/mgmt/2018-01-01-preview/servicebus` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/servicebus/mgmt/2018-01-01-preview/servicebus/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/servicefabric/mgmt/2017-07-01-preview/servicefabric` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/servicefabric/mgmt/2017-07-01-preview/servicefabric/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/servicefabric/mgmt/2018-02-01-preview/servicefabric` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/servicefabric/mgmt/2018-02-01-preview/servicefabric/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/servicefabric/mgmt/2019-03-01-preview/servicefabric` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/servicefabric/mgmt/2019-03-01-preview/servicefabric/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/signalr/mgmt/2018-03-01-preview/signalr` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/signalr/mgmt/2018-03-01-preview/signalr/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/signalr/mgmt/2020-07-01-preview/signalr` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/signalr/mgmt/2020-07-01-preview/signalr/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sql/mgmt/2015-05-01-preview/sql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/sql/mgmt/2015-05-01-preview/sql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sql/mgmt/2017-03-01-preview/sql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/sql/mgmt/2017-03-01-preview/sql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sql/mgmt/2017-10-01-preview/sql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/sql/mgmt/2017-10-01-preview/sql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sql/mgmt/2018-06-01-preview/sql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/sql/mgmt/2018-06-01-preview/sql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sql/mgmt/v3.0/sql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/sql/mgmt/v3.0/sql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sqlvirtualmachine/mgmt/2017-03-01-preview/sqlvirtualmachine` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/sqlvirtualmachine/mgmt/2017-03-01-preview/sqlvirtualmachine/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/storage/mgmt/2015-05-01-preview/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/storage/mgmt/2015-05-01-preview/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/storage/mgmt/2018-03-01-preview/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/storage/mgmt/2018-03-01-preview/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/storage/mgmt/2018-07-01-preview/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/storage/mgmt/2018-07-01-preview/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/storage/mgmt/2020-08-01-preview/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/storage/mgmt/2020-08-01-preview/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/storagecache/mgmt/2019-08-01-preview/storagecache` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/storagecache/mgmt/2019-08-01-preview/storagecache/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/storagepool/mgmt/2020-03-15-preview/storagepool` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/storagepool/mgmt/2020-03-15-preview/storagepool/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/streamanalytics/mgmt/2020-03-01-preview/streamanalytics` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/streamanalytics/mgmt/2020-03-01-preview/streamanalytics/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/subscription/mgmt/2017-11-01-preview/subscription` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/subscription/mgmt/2017-11-01-preview/subscription/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/subscription/mgmt/2018-03-01-preview/subscription` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/subscription/mgmt/2018-03-01-preview/subscription/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/subscription/mgmt/2019-10-01-preview/subscription` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/subscription/mgmt/2019-10-01-preview/subscription/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/support/mgmt/2019-05-01-preview/support` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/support/mgmt/2019-05-01-preview/support/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/synapse/2019-06-01-preview/artifacts` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/synapse/2019-06-01-preview/artifacts/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/synapse/mgmt/2019-06-01-preview/synapse` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/synapse/mgmt/2019-06-01-preview/synapse/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/timeseriesinsights/mgmt/2017-02-28-preview/timeseriesinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/timeseriesinsights/mgmt/2017-02-28-preview/timeseriesinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/timeseriesinsights/mgmt/2018-08-15-preview/timeseriesinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/timeseriesinsights/mgmt/2018-08-15-preview/timeseriesinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/virtualmachineimagebuilder/mgmt/2018-02-01-preview/virtualmachineimagebuilder` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/virtualmachineimagebuilder/mgmt/2018-02-01-preview/virtualmachineimagebuilder/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/virtualmachineimagebuilder/mgmt/2019-02-01-preview/virtualmachineimagebuilder` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/virtualmachineimagebuilder/mgmt/2019-02-01-preview/virtualmachineimagebuilder/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/virtualmachineimagebuilder/mgmt/2019-05-01-preview/virtualmachineimagebuilder` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/virtualmachineimagebuilder/mgmt/2019-05-01-preview/virtualmachineimagebuilder/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/visualstudio/mgmt/2014-04-01-preview/visualstudio` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/visualstudio/mgmt/2014-04-01-preview/visualstudio/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/web/mgmt/2015-08-01-preview/web` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/web/mgmt/2015-08-01-preview/web/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/windowsesu/2019-09-16-preview/windowsesu` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/preview/windowsesu/2019-09-16-preview/windowsesu/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/privatedns/mgmt/2018-09-01/privatedns` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/privatedns/mgmt/2018-09-01/privatedns/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/provisioningservices/mgmt/2017-11-15/iothub` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/provisioningservices/mgmt/2017-11-15/iothub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/provisioningservices/mgmt/2018-01-22/iothub` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/provisioningservices/mgmt/2018-01-22/iothub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2016-08-10/siterecovery` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/recoveryservices/mgmt/2016-08-10/siterecovery/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2018-01-10/siterecovery` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/recoveryservices/mgmt/2018-01-10/siterecovery/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2018-07-10/siterecovery` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/recoveryservices/mgmt/2018-07-10/siterecovery/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2020-02-02/backup` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/recoveryservices/mgmt/2020-02-02/backup/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redhatopenshift/mgmt/2020-04-30/redhatopenshift` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/redhatopenshift/mgmt/2020-04-30/redhatopenshift/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2016-04-01/redis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/redis/mgmt/2016-04-01/redis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2017-02-01/redis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/redis/mgmt/2017-02-01/redis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2017-10-01/redis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/redis/mgmt/2017-10-01/redis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2018-03-01/redis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/redis/mgmt/2018-03-01/redis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2020-06-01/redis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/redis/mgmt/2020-06-01/redis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/relay/mgmt/2016-07-01/relay` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/relay/mgmt/2016-07-01/relay/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/relay/mgmt/2017-04-01/relay` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/relay/mgmt/2017-04-01/relay/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/reservations/mgmt/2017-11-01/reservations` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/reservations/mgmt/2017-11-01/reservations/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2015-11-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/resources/mgmt/2015-11-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2016-02-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/resources/mgmt/2016-02-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2016-07-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/resources/mgmt/2016-07-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2016-09-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/resources/mgmt/2016-09-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2017-05-10/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/resources/mgmt/2017-05-10/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2017-09-01/managedapplications` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/resources/mgmt/2017-09-01/managedapplications/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2018-02-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/resources/mgmt/2018-02-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2018-05-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/resources/mgmt/2018-05-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2018-06-01/managedapplications` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/resources/mgmt/2018-06-01/managedapplications/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-03-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/resources/mgmt/2019-03-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-05-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/resources/mgmt/2019-05-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-07-01/features` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/resources/mgmt/2019-07-01/features/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-07-01/managedapplications` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/resources/mgmt/2019-07-01/managedapplications/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-10-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/resources/mgmt/2019-10-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-11-01/managementgroups` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/resources/mgmt/2019-11-01/managementgroups/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2020-02-01/managementgroups` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/resources/mgmt/2020-02-01/managementgroups/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2020-05-01/managementgroups` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/resources/mgmt/2020-05-01/managementgroups/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2020-06-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/resources/mgmt/2020-06-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/scheduler/mgmt/2016-03-01/scheduler` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/scheduler/mgmt/2016-03-01/scheduler/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/search/mgmt/2015-08-19/search` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/search/mgmt/2015-08-19/search/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/search/mgmt/2020-03-13/search` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/search/mgmt/2020-03-13/search/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/search/mgmt/2020-08-01/search` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/search/mgmt/2020-08-01/search/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/servicebus/mgmt/2015-08-01/servicebus` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/servicebus/mgmt/2015-08-01/servicebus/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/servicebus/mgmt/2017-04-01/servicebus` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/servicebus/mgmt/2017-04-01/servicebus/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/servicefabric/mgmt/2016-09-01/servicefabric` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/servicefabric/mgmt/2016-09-01/servicefabric/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/servicefabric/mgmt/2019-03-01/servicefabric` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/servicefabric/mgmt/2019-03-01/servicefabric/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/signalr/mgmt/2018-10-01/signalr` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/signalr/mgmt/2018-10-01/signalr/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/signalr/mgmt/2020-05-01/signalr` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/signalr/mgmt/2020-05-01/signalr/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/sql/mgmt/2014-04-01/sql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/sql/mgmt/2014-04-01/sql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2015-06-15/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/storage/mgmt/2015-06-15/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2016-01-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/storage/mgmt/2016-01-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2016-05-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/storage/mgmt/2016-05-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2016-12-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/storage/mgmt/2016-12-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2017-06-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/storage/mgmt/2017-06-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2017-10-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/storage/mgmt/2017-10-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2018-02-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/storage/mgmt/2018-02-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2018-11-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/storage/mgmt/2018-11-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2019-04-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/storage/mgmt/2019-04-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2019-06-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/storage/mgmt/2019-06-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagecache/mgmt/2019-11-01/storagecache` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/storagecache/mgmt/2019-11-01/storagecache/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagecache/mgmt/2020-03-01/storagecache` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/storagecache/mgmt/2020-03-01/storagecache/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2018-04-02/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/storagesync/mgmt/2018-04-02/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2018-07-01/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/storagesync/mgmt/2018-07-01/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2018-10-01/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/storagesync/mgmt/2018-10-01/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2019-02-01/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/storagesync/mgmt/2019-02-01/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2019-06-01/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/storagesync/mgmt/2019-06-01/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2019-10-01/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/storagesync/mgmt/2019-10-01/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2020-03-01/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/storagesync/mgmt/2020-03-01/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storsimple1200series/mgmt/2016-10-01/storsimple` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/storsimple1200series/mgmt/2016-10-01/storsimple/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storsimple8000series/mgmt/2017-06-01/storsimple` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/storsimple8000series/mgmt/2017-06-01/storsimple/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/streamanalytics/mgmt/2016-03-01/streamanalytics` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/streamanalytics/mgmt/2016-03-01/streamanalytics/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/subscription/mgmt/2020-09-01/subscription` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/subscription/mgmt/2020-09-01/subscription/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/support/mgmt/2020-04-01/support` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/support/mgmt/2020-04-01/support/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/timeseriesinsights/mgmt/2017-11-15/timeseriesinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/timeseriesinsights/mgmt/2017-11-15/timeseriesinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/timeseriesinsights/mgmt/2020-05-15/timeseriesinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/timeseriesinsights/mgmt/2020-05-15/timeseriesinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/virtualmachineimagebuilder/mgmt/2020-02-01/virtualmachineimagebuilder` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/virtualmachineimagebuilder/mgmt/2020-02-01/virtualmachineimagebuilder/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/vmwarecloudsimple/mgmt/2019-04-01/vmwarecloudsimple` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/vmwarecloudsimple/mgmt/2019-04-01/vmwarecloudsimple/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/web/mgmt/2016-09-01/web` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/web/mgmt/2016-09-01/web/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/web/mgmt/2018-02-01/web` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/web/mgmt/2018-02-01/web/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/web/mgmt/2019-08-01/web` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/web/mgmt/2019-08-01/web/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/web/mgmt/2020-06-01/web` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v50.0.0/services/web/mgmt/2020-06-01/web/CHANGELOG.md) |

## `v49.2.0`

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-12-01/containerservice`
- `github.com/Azure/azure-sdk-for-go/services/guestconfiguration/mgmt/2020-06-25/guestconfiguration`
- `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-09-01/netapp`
- `github.com/Azure/azure-sdk-for-go/services/preview/appplatform/mgmt/2020-11-01-preview/appplatform`
- `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2019-06-01-preview/templatespecs`

### Updated Packages

| Package Path | Changelog |
| :--- | :---: |
| `github.com/Azure/azure-sdk-for-go/services/hdinsight/mgmt/2018-06-01/hdinsight` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.2.0/services/hdinsight/mgmt/2018-06-01/hdinsight/CHANGELOG.md) |

## `v49.1.0`

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2020-09-01/cdn`
- `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-07-01/network`
- `github.com/Azure/azure-sdk-for-go/services/preview/azureadb2c/mgmt/2020-05-01-preview/azureadb2c`
- `github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2020-06-01/redis`

### Updated Packages

| Package Path | Changelog |
| :--- | :---: |
| `github.com/Azure/azure-sdk-for-go/services/datafactory/mgmt/2018-06-01/datafactory` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.1.0/services/datafactory/mgmt/2018-06-01/datafactory/CHANGELOG.md) |

### Breaking Changes

| Package Path | Changelog |
| :--- | :---: |
| `github.com/Azure/azure-sdk-for-go/services/preview/security/mgmt/v3.0/security` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.1.0/services/preview/security/mgmt/v3.0/security/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/synapse/mgmt/2019-06-01-preview/synapse` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.1.0/services/preview/synapse/mgmt/2019-06-01-preview/synapse/CHANGELOG.md) |

## `v49.0.0`

**NOTE**: Due to the changes requested in [this issue](https://github.com/Azure/azure-sdk-for-go/issues/12834), we changed the constructor of all the pager structs, which leads to a very long list of breaking changes. Please check the details of the breaking changes by the link in the table.

### New Packages

- `github.com/Azure/azure-sdk-for-go/services/attestation/2018-09-01/attestation`
- `github.com/Azure/azure-sdk-for-go/services/attestation/2020-10-01/attestation`
- `github.com/Azure/azure-sdk-for-go/services/attestation/mgmt/2018-09-01/attestation`
- `github.com/Azure/azure-sdk-for-go/services/batch/2020-09-01.12.0/batch`
- `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2020-05-01/batch`
- `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2020-09-01/batch`
- `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2019-06-15/cdn`
- `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v3.1/computervision`
- `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v3.1/customvision/prediction`
- `github.com/Azure/azure-sdk-for-go/services/consumption/mgmt/2019-10-01/consumption`
- `github.com/Azure/azure-sdk-for-go/services/databox/mgmt/2020-04-01/databox`
- `github.com/Azure/azure-sdk-for-go/services/databox/mgmt/2020-11-01/databox`
- `github.com/Azure/azure-sdk-for-go/services/domainservices/mgmt/2020-01-01/aad`
- `github.com/Azure/azure-sdk-for-go/services/migrate/mgmt/2020-01-01/migrate`
- `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-03-01/netapp`
- `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-05-01/netapp`
- `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-08-01/netapp`
- `github.com/Azure/azure-sdk-for-go/services/operationalinsights/mgmt/2020-10-01/operationalinsights`
- `github.com/Azure/azure-sdk-for-go/services/peering/mgmt/2020-10-01/peering`
- `github.com/Azure/azure-sdk-for-go/services/personalizer/v1.0/personalizer`
- `github.com/Azure/azure-sdk-for-go/services/preview/alertsmanagement/mgmt/2019-05-05-preview/alertsmanagement`
- `github.com/Azure/azure-sdk-for-go/services/preview/appconfiguration/mgmt/2020-07-01-preview/appconfiguration`
- `github.com/Azure/azure-sdk-for-go/services/preview/avs/mgmt/2020-07-17-preview/avs`
- `github.com/Azure/azure-sdk-for-go/services/preview/baremetalinfrastructure/mgmt/2020-08-06-preview/baremetalinfrastructure`
- `github.com/Azure/azure-sdk-for-go/services/preview/cognitiveservices/v1.0_preview.1/translatortext`
- `github.com/Azure/azure-sdk-for-go/services/preview/cognitiveservices/v3.4-preview/customvision/training`
- `github.com/Azure/azure-sdk-for-go/services/preview/containerregistry/mgmt/2019-06-01-preview/containerregistry`
- `github.com/Azure/azure-sdk-for-go/services/preview/containerregistry/mgmt/2020-11-01-preview/containerregistry`
- `github.com/Azure/azure-sdk-for-go/services/preview/databoxedge/mgmt/2020-05-01-preview/databoxedge`
- `github.com/Azure/azure-sdk-for-go/services/preview/delegatednetwork/mgmt/2020-08-08-preview/delegatednetwork`
- `github.com/Azure/azure-sdk-for-go/services/preview/desktopvirtualization/mgmt/2020-09-21-preview/desktopvirtualization`
- `github.com/Azure/azure-sdk-for-go/services/preview/desktopvirtualization/mgmt/2020-10-19-preview/desktopvirtualization`
- `github.com/Azure/azure-sdk-for-go/services/preview/desktopvirtualization/mgmt/2020-11-02-preview/desktopvirtualization`
- `github.com/Azure/azure-sdk-for-go/services/preview/digitaltwins/2020-05-31-preview/digitaltwins`
- `github.com/Azure/azure-sdk-for-go/services/preview/hybridnetwork/mgmt/2020-01-01-preview/hybridnetwork`
- `github.com/Azure/azure-sdk-for-go/services/preview/iothub/mgmt/2019-07-01-preview/devices`
- `github.com/Azure/azure-sdk-for-go/services/preview/monitor/mgmt/2019-11-01-preview/insights`
- `github.com/Azure/azure-sdk-for-go/services/preview/policyinsights/mgmt/2020-07-01-preview/policyinsights`
- `github.com/Azure/azure-sdk-for-go/services/preview/postgresql/mgmt/2020-11-05-preview/postgresqlflexibleservers`
- `github.com/Azure/azure-sdk-for-go/services/preview/powerplatform/mgmt/2020-10-30/powerplatform`
- `github.com/Azure/azure-sdk-for-go/services/preview/qnamaker/cognitiveservices/v5.0-preview.1/qnamaker`
- `github.com/Azure/azure-sdk-for-go/services/preview/redis/mgmt/2019-07-01-preview/redis`
- `github.com/Azure/azure-sdk-for-go/services/preview/storagepool/mgmt/2020-03-15-preview/storagepool`
- `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-11-01/managementgroups`
- `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2020-02-01/managementgroups`
- `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2020-05-01/managementgroups`
- `github.com/Azure/azure-sdk-for-go/services/search/mgmt/2020-08-01/search`
- `github.com/Azure/azure-sdk-for-go/services/servicefabric/7.2/servicefabric`

### Removed Packages

- `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v1.0_preview.1/translatortext`
- `github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2019-05-01-preview/containerregistry`
- `github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2019-06-01-preview/containerregistry`
- `github.com/Azure/azure-sdk-for-go/services/preview/alertsmanagement/mgmt/2019-05-05/alertsmanagement`
- `github.com/Azure/azure-sdk-for-go/services/preview/attestation/mgmt/2018-09-01-preview/attestation`
- `github.com/Azure/azure-sdk-for-go/services/preview/cdn/mgmt/2019-06-15-preview/cdn`
- `github.com/Azure/azure-sdk-for-go/services/preview/cognitiveservices/v3.0-preview/computervision`
- `github.com/Azure/azure-sdk-for-go/services/preview/costmanagement/mgmt/2019-10-01/costmanagement`
- `github.com/Azure/azure-sdk-for-go/services/preview/digitaltwins/2020-05-31/digitaltwins`
- `github.com/Azure/azure-sdk-for-go/services/preview/hybridcompute/mgmt/2019-03-18-preview/hybridcompute`
- `github.com/Azure/azure-sdk-for-go/services/preview/management/2020-01-01-preview/hybridnetwork`
- `github.com/Azure/azure-sdk-for-go/services/preview/migrate/mgmt/2020-01-01/migrate`
- `github.com/Azure/azure-sdk-for-go/services/preview/personalizer/v1.0/personalizer`
- `github.com/Azure/azure-sdk-for-go/services/preview/policyinsights/mgmt/2017-08-09-preview/policyinsights`
- `github.com/Azure/azure-sdk-for-go/services/preview/policyinsights/mgmt/2017-10-17-preview/policyinsights`
- `github.com/Azure/azure-sdk-for-go/services/preview/policyinsights/mgmt/2017-12-12-preview/policyinsights`
- `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2019-11-01/managementgroups`
- `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2020-02-01/managementgroups`
- `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2020-05-01/managementgroups`
- `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2017-07-01/backup`

### Updated Packages

| Package Path | Changelog |
| :--- | :---: |
| `github.com/Azure/azure-sdk-for-go/services/eventgrid/2018-01-01/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/eventgrid/2018-01-01/eventgrid/CHANGELOG.md) |

### Breaking Changes

| Package Path | Changelog |
| :--- | :---: |
| `github.com/Azure/azure-sdk-for-go/services/adhybridhealthservice/mgmt/2014-01-01/adhybridhealthservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/adhybridhealthservice/mgmt/2014-01-01/adhybridhealthservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/advisor/mgmt/2017-03-31/advisor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/advisor/mgmt/2017-03-31/advisor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/advisor/mgmt/2017-04-19/advisor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/advisor/mgmt/2017-04-19/advisor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/advisor/mgmt/2020-01-01/advisor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/advisor/mgmt/2020-01-01/advisor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/alertsmanagement/mgmt/2018-05-05/alertsmanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/alertsmanagement/mgmt/2018-05-05/alertsmanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/alertsmanagement/mgmt/2019-03-01/alertsmanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/alertsmanagement/mgmt/2019-03-01/alertsmanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/analysisservices/mgmt/2017-08-01/analysisservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/analysisservices/mgmt/2017-08-01/analysisservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2016-07-07/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/apimanagement/mgmt/2016-07-07/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2016-10-10/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/apimanagement/mgmt/2016-10-10/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2017-03-01/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/apimanagement/mgmt/2017-03-01/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2018-01-01/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/apimanagement/mgmt/2018-01-01/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2019-01-01/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/apimanagement/mgmt/2019-01-01/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/apimanagement/mgmt/2019-12-01/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/apimanagement/mgmt/2019-12-01/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/appconfiguration/mgmt/2019-10-01/appconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/appconfiguration/mgmt/2019-10-01/appconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/appconfiguration/mgmt/2020-06-01/appconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/appconfiguration/mgmt/2020-06-01/appconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/appinsights/mgmt/2015-05-01/insights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/appinsights/mgmt/2015-05-01/insights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/appplatform/mgmt/2020-07-01/appplatform` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/appplatform/mgmt/2020-07-01/appplatform/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/authorization/mgmt/2015-07-01/authorization` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/authorization/mgmt/2015-07-01/authorization/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/automation/mgmt/2015-10-31/automation` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/automation/mgmt/2015-10-31/automation/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/avs/mgmt/2020-03-20/avs` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/avs/mgmt/2020-03-20/avs/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/azurestack/mgmt/2017-06-01/azurestack` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/azurestack/mgmt/2017-06-01/azurestack/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/azurestackhci/mgmt/2020-10-01/azurestackhci` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/azurestackhci/mgmt/2020-10-01/azurestackhci/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/2017-05-01.5.0/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/batch/2017-05-01.5.0/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/2018-03-01.6.1/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/batch/2018-03-01.6.1/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/2018-08-01.7.0/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/batch/2018-08-01.7.0/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/2018-12-01.8.0/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/batch/2018-12-01.8.0/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/2019-06-01.9.0/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/batch/2019-06-01.9.0/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/2019-08-01.10.0/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/batch/2019-08-01.10.0/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/2020-03-01.11.0/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/batch/2020-03-01.11.0/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2015-12-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/batch/mgmt/2015-12-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2017-01-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/batch/mgmt/2017-01-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2017-05-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/batch/mgmt/2017-05-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2017-09-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/batch/mgmt/2017-09-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2018-12-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/batch/mgmt/2018-12-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2019-04-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/batch/mgmt/2019-04-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2019-08-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/batch/mgmt/2019-08-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batch/mgmt/2020-03-01/batch` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/batch/mgmt/2020-03-01/batch/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batchai/mgmt/2018-03-01/batchai` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/batchai/mgmt/2018-03-01/batchai/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/batchai/mgmt/2018-05-01/batchai` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/batchai/mgmt/2018-05-01/batchai/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2016-10-02/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/cdn/mgmt/2016-10-02/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2017-04-02/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/cdn/mgmt/2017-04-02/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2017-10-12/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/cdn/mgmt/2017-10-12/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2019-04-15/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/cdn/mgmt/2019-04-15/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2020-04-15/cdn` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/cdn/mgmt/2020-04-15/cdn/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/mgmt/2017-04-18/cognitiveservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/cognitiveservices/mgmt/2017-04-18/cognitiveservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v4.0/qnamaker` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/cognitiveservices/v4.0/qnamaker/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2015-06-15/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/compute/mgmt/2015-06-15/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2016-03-30/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/compute/mgmt/2016-03-30/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2017-03-30/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/compute/mgmt/2017-03-30/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2017-09-01/skus` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/compute/mgmt/2017-09-01/skus/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2017-12-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/compute/mgmt/2017-12-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2018-04-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/compute/mgmt/2018-04-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2018-06-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/compute/mgmt/2018-06-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2018-10-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/compute/mgmt/2018-10-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-03-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/compute/mgmt/2019-03-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-07-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/compute/mgmt/2019-07-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-12-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/compute/mgmt/2019-12-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2020-06-01/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/compute/mgmt/2020-06-01/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2020-06-30/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/compute/mgmt/2020-06-30/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/consumption/mgmt/2017-11-30/consumption` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/consumption/mgmt/2017-11-30/consumption/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/consumption/mgmt/2018-01-31/consumption` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/consumption/mgmt/2018-01-31/consumption/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/consumption/mgmt/2018-03-31/consumption` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/consumption/mgmt/2018-03-31/consumption/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/consumption/mgmt/2018-05-31/consumption` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/consumption/mgmt/2018-05-31/consumption/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/consumption/mgmt/2018-06-30/consumption` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/consumption/mgmt/2018-06-30/consumption/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/consumption/mgmt/2018-08-31/consumption` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/consumption/mgmt/2018-08-31/consumption/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/consumption/mgmt/2018-10-01/consumption` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/consumption/mgmt/2018-10-01/consumption/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/consumption/mgmt/2019-01-01/consumption` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/consumption/mgmt/2019-01-01/consumption/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerinstance/mgmt/2018-04-01/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerinstance/mgmt/2018-04-01/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerinstance/mgmt/2018-06-01/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerinstance/mgmt/2018-06-01/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerinstance/mgmt/2018-09-01/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerinstance/mgmt/2018-09-01/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerinstance/mgmt/2018-10-01/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerinstance/mgmt/2018-10-01/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerinstance/mgmt/2019-12-01/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerinstance/mgmt/2019-12-01/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2017-03-01/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerregistry/mgmt/2017-03-01/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2017-10-01/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerregistry/mgmt/2017-10-01/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2018-09-01/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerregistry/mgmt/2018-09-01/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2019-04-01/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerregistry/mgmt/2019-04-01/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2019-05-01/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerregistry/mgmt/2019-05-01/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2016-09-30/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerservice/mgmt/2016-09-30/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2017-01-31/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerservice/mgmt/2017-01-31/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2017-07-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerservice/mgmt/2017-07-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2017-08-31/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerservice/mgmt/2017-08-31/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2017-09-30/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerservice/mgmt/2017-09-30/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2018-03-31/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerservice/mgmt/2018-03-31/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2019-04-30/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerservice/mgmt/2019-04-30/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2019-06-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerservice/mgmt/2019-06-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2019-08-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerservice/mgmt/2019-08-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2019-10-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerservice/mgmt/2019-10-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2019-11-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerservice/mgmt/2019-11-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-01-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerservice/mgmt/2020-01-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-02-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerservice/mgmt/2020-02-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-03-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerservice/mgmt/2020-03-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-04-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerservice/mgmt/2020-04-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-06-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerservice/mgmt/2020-06-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-07-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerservice/mgmt/2020-07-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-09-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerservice/mgmt/2020-09-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-11-01/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/containerservice/mgmt/2020-11-01/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cosmos-db/mgmt/2015-04-08/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/cosmos-db/mgmt/2015-04-08/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/cosmos-db/mgmt/2019-08-01/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/cosmos-db/mgmt/2019-08-01/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/costmanagement/mgmt/2018-05-31/costmanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/costmanagement/mgmt/2018-05-31/costmanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/costmanagement/mgmt/2019-01-01/costmanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/costmanagement/mgmt/2019-01-01/costmanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/costmanagement/mgmt/2019-10-01/costmanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/costmanagement/mgmt/2019-10-01/costmanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/customerinsights/mgmt/2017-01-01/customerinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/customerinsights/mgmt/2017-01-01/customerinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/customerinsights/mgmt/2017-04-26/customerinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/customerinsights/mgmt/2017-04-26/customerinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databox/mgmt/2018-01-01/databox` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/databox/mgmt/2018-01-01/databox/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databox/mgmt/2019-09-01/databox` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/databox/mgmt/2019-09-01/databox/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databoxedge/mgmt/2019-03-01/databoxedge` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/databoxedge/mgmt/2019-03-01/databoxedge/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databoxedge/mgmt/2019-07-01/databoxedge` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/databoxedge/mgmt/2019-07-01/databoxedge/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databoxedge/mgmt/2019-08-01/databoxedge` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/databoxedge/mgmt/2019-08-01/databoxedge/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/databricks/mgmt/2018-04-01/databricks` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/databricks/mgmt/2018-04-01/databricks/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datafactory/mgmt/2018-06-01/datafactory` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/datafactory/mgmt/2018-06-01/datafactory/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datalake/analytics/2016-11-01-preview/catalog` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/datalake/analytics/2016-11-01-preview/catalog/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datalake/analytics/2016-11-01/job` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/datalake/analytics/2016-11-01/job/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datalake/analytics/mgmt/2016-11-01/account` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/datalake/analytics/mgmt/2016-11-01/account/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datalake/store/mgmt/2016-11-01/account` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/datalake/store/mgmt/2016-11-01/account/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datamigration/mgmt/2018-04-19/datamigration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/datamigration/mgmt/2018-04-19/datamigration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/datashare/mgmt/2019-11-01/datashare` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/datashare/mgmt/2019-11-01/datashare/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/devspaces/mgmt/2019-04-01/devspaces` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/devspaces/mgmt/2019-04-01/devspaces/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/devtestlabs/mgmt/2016-05-15/dtl` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/devtestlabs/mgmt/2016-05-15/dtl/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/devtestlabs/mgmt/2018-09-15/dtl` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/devtestlabs/mgmt/2018-09-15/dtl/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/digitaltwins/mgmt/2020-10-31/digitaltwins` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/digitaltwins/mgmt/2020-10-31/digitaltwins/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/dns/mgmt/2016-04-01/dns` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/dns/mgmt/2016-04-01/dns/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/dns/mgmt/2017-09-01/dns` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/dns/mgmt/2017-09-01/dns/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/dns/mgmt/2017-10-01/dns` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/dns/mgmt/2017-10-01/dns/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/dns/mgmt/2018-05-01/dns` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/dns/mgmt/2018-05-01/dns/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/domainservices/mgmt/2017-01-01/aad` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/domainservices/mgmt/2017-01-01/aad/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/domainservices/mgmt/2017-06-01/aad` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/domainservices/mgmt/2017-06-01/aad/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/eventgrid/mgmt/2019-06-01/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/eventgrid/mgmt/2019-06-01/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/eventgrid/mgmt/2020-06-01/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/eventgrid/mgmt/2020-06-01/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/eventhub/mgmt/2015-08-01/eventhub` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/eventhub/mgmt/2015-08-01/eventhub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/eventhub/mgmt/2017-04-01/eventhub` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/eventhub/mgmt/2017-04-01/eventhub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2019-04-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/frontdoor/mgmt/2019-04-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2019-05-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/frontdoor/mgmt/2019-05-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2019-10-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/frontdoor/mgmt/2019-10-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2019-11-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/frontdoor/mgmt/2019-11-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2020-01-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/frontdoor/mgmt/2020-01-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2020-04-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/frontdoor/mgmt/2020-04-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/frontdoor/mgmt/2020-05-01/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/frontdoor/mgmt/2020-05-01/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/graphrbac/1.6/graphrbac` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/graphrbac/1.6/graphrbac/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/hdinsight/mgmt/2018-06-01/hdinsight` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/hdinsight/mgmt/2018-06-01/hdinsight/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/healthcareapis/mgmt/2019-09-16/healthcareapis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/healthcareapis/mgmt/2019-09-16/healthcareapis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/healthcareapis/mgmt/2020-03-15/healthcareapis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/healthcareapis/mgmt/2020-03-15/healthcareapis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/healthcareapis/mgmt/2020-03-30/healthcareapis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/healthcareapis/mgmt/2020-03-30/healthcareapis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/hybridcompute/mgmt/2019-12-12/hybridcompute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/hybridcompute/mgmt/2019-12-12/hybridcompute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/hybridcompute/mgmt/2020-08-02/hybridcompute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/hybridcompute/mgmt/2020-08-02/hybridcompute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/hybriddatamanager/mgmt/2016-06-01/hybriddata` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/hybriddatamanager/mgmt/2016-06-01/hybriddata/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/hybriddatamanager/mgmt/2019-06-01/hybriddata` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/hybriddatamanager/mgmt/2019-06-01/hybriddata/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iotcentral/mgmt/2018-09-01/iotcentral` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/iotcentral/mgmt/2018-09-01/iotcentral/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iothub/mgmt/2016-02-03/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/iothub/mgmt/2016-02-03/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iothub/mgmt/2017-01-19/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/iothub/mgmt/2017-01-19/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iothub/mgmt/2017-07-01/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/iothub/mgmt/2017-07-01/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iothub/mgmt/2018-01-22/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/iothub/mgmt/2018-01-22/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iothub/mgmt/2018-04-01/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/iothub/mgmt/2018-04-01/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/iothub/mgmt/2020-03-01/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/iothub/mgmt/2020-03-01/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/keyvault/2015-06-01/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/keyvault/2015-06-01/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/keyvault/2016-10-01/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/keyvault/2016-10-01/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/keyvault/mgmt/2015-06-01/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/keyvault/mgmt/2015-06-01/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/keyvault/mgmt/2016-10-01/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/keyvault/mgmt/2016-10-01/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/keyvault/mgmt/2018-02-14/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/keyvault/mgmt/2018-02-14/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/keyvault/mgmt/2019-09-01/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/keyvault/mgmt/2019-09-01/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/keyvault/v7.0/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/keyvault/v7.0/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/keyvault/v7.1/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/keyvault/v7.1/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2019-01-21/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/kusto/mgmt/2019-01-21/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2019-05-15/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/kusto/mgmt/2019-05-15/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2019-09-07/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/kusto/mgmt/2019-09-07/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2019-11-09/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/kusto/mgmt/2019-11-09/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2020-02-15/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/kusto/mgmt/2020-02-15/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2020-06-14/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/kusto/mgmt/2020-06-14/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/kusto/mgmt/2020-09-18/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/kusto/mgmt/2020-09-18/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/labservices/mgmt/2018-10-15/labservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/labservices/mgmt/2018-10-15/labservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/logic/mgmt/2016-06-01/logic` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/logic/mgmt/2016-06-01/logic/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/logic/mgmt/2019-05-01/logic` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/logic/mgmt/2019-05-01/logic/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearning/mgmt/2016-04-01/workspaces` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/machinelearning/mgmt/2016-04-01/workspaces/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearning/mgmt/2017-01-01/webservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/machinelearning/mgmt/2017-01-01/webservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearning/mgmt/2019-10-01/workspaces` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/machinelearning/mgmt/2019-10-01/workspaces/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2018-11-19/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/machinelearningservices/mgmt/2018-11-19/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2019-05-01/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/machinelearningservices/mgmt/2019-05-01/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2019-06-01/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/machinelearningservices/mgmt/2019-06-01/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2019-11-01/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/machinelearningservices/mgmt/2019-11-01/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2020-01-01/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/machinelearningservices/mgmt/2020-01-01/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2020-03-01/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/machinelearningservices/mgmt/2020-03-01/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/machinelearningservices/mgmt/2020-04-01/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/machinelearningservices/mgmt/2020-04-01/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/managedservices/mgmt/2019-06-01/managedservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/managedservices/mgmt/2019-06-01/managedservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mariadb/mgmt/2018-06-01/mariadb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/mariadb/mgmt/2018-06-01/mariadb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mariadb/mgmt/2020-01-01/mariadb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/mariadb/mgmt/2020-01-01/mariadb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/marketplaceordering/mgmt/2015-06-01/marketplaceordering` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/marketplaceordering/mgmt/2015-06-01/marketplaceordering/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mediaservices/mgmt/2018-07-01/media` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/mediaservices/mgmt/2018-07-01/media/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mediaservices/mgmt/2020-05-01/media` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/mediaservices/mgmt/2020-05-01/media/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/msi/mgmt/2018-11-30/msi` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/msi/mgmt/2018-11-30/msi/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mysql/mgmt/2017-12-01/mysql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/mysql/mgmt/2017-12-01/mysql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/mysql/mgmt/2020-01-01/mysql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/mysql/mgmt/2020-01-01/mysql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2019-10-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/netapp/mgmt/2019-10-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2019-11-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/netapp/mgmt/2019-11-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-02-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/netapp/mgmt/2020-02-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/netapp/mgmt/2020-06-01/netapp` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/netapp/mgmt/2020-06-01/netapp/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2015-06-15/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2015-06-15/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2016-03-30/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2016-03-30/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2016-06-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2016-06-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2016-09-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2016-09-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2016-12-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2016-12-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-03-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2017-03-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-06-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2017-06-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-08-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2017-08-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-09-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2017-09-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-10-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2017-10-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-11-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2017-11-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-01-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2018-01-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-02-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2018-02-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-04-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2018-04-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-06-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2018-06-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-07-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2018-07-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-08-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2018-08-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-10-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2018-10-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-11-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2018-11-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2018-12-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2018-12-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-02-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2019-02-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-04-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2019-04-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2019-06-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-07-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2019-07-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-08-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2019-08-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-09-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2019-09-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-11-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2019-11-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-12-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2019-12-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-03-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2020-03-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-04-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2020-04-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-05-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2020-05-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/network/mgmt/2020-06-01/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/network/mgmt/2020-06-01/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/notificationhubs/mgmt/2014-09-01/notificationhubs` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/notificationhubs/mgmt/2014-09-01/notificationhubs/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/notificationhubs/mgmt/2016-03-01/notificationhubs` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/notificationhubs/mgmt/2016-03-01/notificationhubs/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/notificationhubs/mgmt/2017-04-01/notificationhubs` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/notificationhubs/mgmt/2017-04-01/notificationhubs/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/operationalinsights/mgmt/2015-03-20/operationalinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/operationalinsights/mgmt/2015-03-20/operationalinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/operationalinsights/mgmt/2020-08-01/operationalinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/operationalinsights/mgmt/2020-08-01/operationalinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/peering/mgmt/2020-04-01/peering` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/peering/mgmt/2020-04-01/peering/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/postgresql/mgmt/2017-12-01/postgresql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/postgresql/mgmt/2017-12-01/postgresql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/postgresql/mgmt/2020-01-01/postgresql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/postgresql/mgmt/2020-01-01/postgresql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/powerbidedicated/mgmt/2017-10-01/powerbidedicated` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/powerbidedicated/mgmt/2017-10-01/powerbidedicated/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/advisor/mgmt/2016-07-12-preview/advisor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/advisor/mgmt/2016-07-12-preview/advisor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/alertsmanagement/mgmt/2018-05-05-preview/alertsmanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/alertsmanagement/mgmt/2018-05-05-preview/alertsmanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/alertsmanagement/mgmt/2019-06-01-preview/alertsmanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/alertsmanagement/mgmt/2019-06-01-preview/alertsmanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/apimanagement/ctrl/2017-03-01/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/apimanagement/ctrl/2017-03-01/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/apimanagement/mgmt/2018-06-01-preview/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/apimanagement/mgmt/2018-06-01-preview/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/apimanagement/mgmt/2019-12-01-preview/apimanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/apimanagement/mgmt/2019-12-01-preview/apimanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/appconfiguration/mgmt/2019-02-01-preview/appconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/appconfiguration/mgmt/2019-02-01-preview/appconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/appconfiguration/mgmt/2019-11-01-preview/appconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/appconfiguration/mgmt/2019-11-01-preview/appconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/appplatform/mgmt/2019-05-01-preview/appplatform` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/appplatform/mgmt/2019-05-01-preview/appplatform/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/authorization/mgmt/2017-10-01-preview/authorization` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/authorization/mgmt/2017-10-01-preview/authorization/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/authorization/mgmt/2018-01-01-preview/authorization` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/authorization/mgmt/2018-01-01-preview/authorization/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/authorization/mgmt/2018-07-01-preview/authorization` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/authorization/mgmt/2018-07-01-preview/authorization/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/authorization/mgmt/2018-09-01-preview/authorization` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/authorization/mgmt/2018-09-01-preview/authorization/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/automanage/mgmt/2020-06-30-preview/automanage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/automanage/mgmt/2020-06-30-preview/automanage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/automation/mgmt/2017-05-15-preview/automation` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/automation/mgmt/2017-05-15-preview/automation/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/automation/mgmt/2018-01-15-preview/automation` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/automation/mgmt/2018-01-15-preview/automation/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/automation/mgmt/2018-06-30-preview/automation` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/automation/mgmt/2018-06-30-preview/automation/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/avs/mgmt/2019-08-09-preview/avs` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/avs/mgmt/2019-08-09-preview/avs/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/azuredata/mgmt/2017-03-01-preview/azuredata` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/azuredata/mgmt/2017-03-01-preview/azuredata/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/azuredata/mgmt/2019-07-24-preview/azuredata` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/azuredata/mgmt/2019-07-24-preview/azuredata/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/azurestackhci/mgmt/2020-03-01-preview/azurestackhci` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/azurestackhci/mgmt/2020-03-01-preview/azurestackhci/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/batchai/mgmt/2017-09-01-preview/batchai` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/batchai/mgmt/2017-09-01-preview/batchai/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/billing/mgmt/2017-02-27-preview/billing` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/billing/mgmt/2017-02-27-preview/billing/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/billing/mgmt/2017-04-24-preview/billing` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/billing/mgmt/2017-04-24-preview/billing/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/billing/mgmt/2018-03-01-preview/billing` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/billing/mgmt/2018-03-01-preview/billing/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/billing/mgmt/2018-11-01-preview/billing` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/billing/mgmt/2018-11-01-preview/billing/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/billing/mgmt/2020-05-01-preview/billing` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/billing/mgmt/2020-05-01-preview/billing/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/blockchain/mgmt/2018-06-01-preview/blockchain` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/blockchain/mgmt/2018-06-01-preview/blockchain/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/blueprint/mgmt/2018-11-01-preview/blueprint` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/blueprint/mgmt/2018-11-01-preview/blueprint/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/botservice/mgmt/2017-12-01/botservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/botservice/mgmt/2017-12-01/botservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/botservice/mgmt/2018-07-12/botservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/botservice/mgmt/2018-07-12/botservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cognitiveservices/v1.0/anomalydetector` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/cognitiveservices/v1.0/anomalydetector/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/commerce/mgmt/2015-06-01-preview/commerce` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/commerce/mgmt/2015-06-01-preview/commerce/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/communication/mgmt/2020-08-20-preview/communication` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/communication/mgmt/2020-08-20-preview/communication/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/compute/mgmt/2016-04-30-preview/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/compute/mgmt/2016-04-30-preview/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/confluent/mgmt/2020-03-01-preview/confluent` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/confluent/mgmt/2020-03-01-preview/confluent/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/consumption/mgmt/2017-04-24-preview/consumption` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/consumption/mgmt/2017-04-24-preview/consumption/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/consumption/mgmt/2017-12-30-preview/consumption` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/consumption/mgmt/2017-12-30-preview/consumption/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerinstance/mgmt/2017-08-01-preview/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/containerinstance/mgmt/2017-08-01-preview/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerinstance/mgmt/2017-10-01-preview/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/containerinstance/mgmt/2017-10-01-preview/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerinstance/mgmt/2017-12-01-preview/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/containerinstance/mgmt/2017-12-01-preview/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerinstance/mgmt/2018-02-01-preview/containerinstance` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/containerinstance/mgmt/2018-02-01-preview/containerinstance/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerregistry/mgmt/2016-06-27-preview/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/containerregistry/mgmt/2016-06-27-preview/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerregistry/mgmt/2017-06-01-preview/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/containerregistry/mgmt/2017-06-01-preview/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerregistry/mgmt/2018-02-01/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/containerregistry/mgmt/2018-02-01/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerregistry/mgmt/2019-12-01-preview/containerregistry` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/containerregistry/mgmt/2019-12-01-preview/containerregistry/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerservice/mgmt/2018-08-01-preview/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/containerservice/mgmt/2018-08-01-preview/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerservice/mgmt/2018-09-30-preview/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/containerservice/mgmt/2018-09-30-preview/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerservice/mgmt/2019-02-01-preview/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/containerservice/mgmt/2019-02-01-preview/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerservice/mgmt/2019-09-30-preview/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/containerservice/mgmt/2019-09-30-preview/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/containerservice/mgmt/2019-10-27-preview/containerservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/containerservice/mgmt/2019-10-27-preview/containerservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2019-08-01-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/cosmos-db/mgmt/2019-08-01-preview/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2019-12-12-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/cosmos-db/mgmt/2019-12-12-preview/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2020-03-01-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/cosmos-db/mgmt/2020-03-01-preview/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2020-04-01-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/cosmos-db/mgmt/2020-04-01-preview/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2020-06-01-preview/documentdb` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/cosmos-db/mgmt/2020-06-01-preview/documentdb/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/costmanagement/mgmt/2018-08-01-preview/costmanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/costmanagement/mgmt/2018-08-01-preview/costmanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/costmanagement/mgmt/2019-03-01/costmanagement` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/costmanagement/mgmt/2019-03-01/costmanagement/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/customerlockbox/mgmt/2018-02-28-preview/customerlockbox` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/customerlockbox/mgmt/2018-02-28-preview/customerlockbox/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/customproviders/mgmt/2018-09-01-preview/customproviders` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/customproviders/mgmt/2018-09-01-preview/customproviders/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datadog/mgmt/2020-02-01-preview/datadog` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/datadog/mgmt/2020-02-01-preview/datadog/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datafactory/mgmt/2017-09-01-preview/datafactory` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/datafactory/mgmt/2017-09-01-preview/datafactory/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datalake/analytics/2015-10-01-preview/catalog` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/datalake/analytics/2015-10-01-preview/catalog/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datalake/analytics/2015-11-01-preview/job` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/datalake/analytics/2015-11-01-preview/job/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datalake/analytics/2016-03-20-preview/job` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/datalake/analytics/2016-03-20-preview/job/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datalake/analytics/2017-09-01-preview/job` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/datalake/analytics/2017-09-01-preview/job/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datalake/analytics/mgmt/2015-10-01-preview/account` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/datalake/analytics/mgmt/2015-10-01-preview/account/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datalake/store/mgmt/2015-10-01-preview/account` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/datalake/store/mgmt/2015-10-01-preview/account/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datamigration/mgmt/2017-11-15-preview/datamigration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/datamigration/mgmt/2017-11-15-preview/datamigration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datamigration/mgmt/2018-03-31-preview/datamigration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/datamigration/mgmt/2018-03-31-preview/datamigration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datamigration/mgmt/2018-07-15-preview/datamigration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/datamigration/mgmt/2018-07-15-preview/datamigration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/datashare/mgmt/2018-11-01-preview/datashare` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/datashare/mgmt/2018-11-01-preview/datashare/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/desktopvirtualization/mgmt/2019-01-23-preview/desktopvirtualization` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/desktopvirtualization/mgmt/2019-01-23-preview/desktopvirtualization/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/desktopvirtualization/mgmt/2019-09-24-preview/desktopvirtualization` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/desktopvirtualization/mgmt/2019-09-24-preview/desktopvirtualization/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/desktopvirtualization/mgmt/2019-12-10-preview/desktopvirtualization` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/desktopvirtualization/mgmt/2019-12-10-preview/desktopvirtualization/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/devops/mgmt/2019-07-01-preview/devops` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/devops/mgmt/2019-07-01-preview/devops/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/devtestlabs/mgmt/2015-05-21-preview/dtl` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/devtestlabs/mgmt/2015-05-21-preview/dtl/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/digitaltwins/mgmt/2020-03-01-preview/digitaltwins` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/digitaltwins/mgmt/2020-03-01-preview/digitaltwins/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/dns/mgmt/2015-05-04-preview/dns` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/dns/mgmt/2015-05-04-preview/dns/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/dns/mgmt/2018-03-01-preview/dns` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/dns/mgmt/2018-03-01-preview/dns/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/enterpriseknowledgegraphservice/2018-12-03/enterpriseknowledgegraphservice` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/enterpriseknowledgegraphservice/2018-12-03/enterpriseknowledgegraphservice/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2019-02-01-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/eventgrid/mgmt/2019-02-01-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2020-01-01-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/eventgrid/mgmt/2020-01-01-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventgrid/mgmt/2020-04-01-preview/eventgrid` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/eventgrid/mgmt/2020-04-01-preview/eventgrid/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/eventhub/mgmt/2018-01-01-preview/eventhub` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/eventhub/mgmt/2018-01-01-preview/eventhub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/frontdoor/mgmt/2018-08-01-preview/frontdoor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/frontdoor/mgmt/2018-08-01-preview/frontdoor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/hanaonazure/mgmt/2017-11-03-preview/hanaonazure` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/hanaonazure/mgmt/2017-11-03-preview/hanaonazure/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/hardwaresecuritymodules/mgmt/2018-10-31-preview/hardwaresecuritymodules` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/hardwaresecuritymodules/mgmt/2018-10-31-preview/hardwaresecuritymodules/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/hdinsight/2018-11-01-preview/hdinsight` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/hdinsight/2018-11-01-preview/hdinsight/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/hdinsight/mgmt/2015-03-01-preview/hdinsight` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/hdinsight/mgmt/2015-03-01-preview/hdinsight/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/healthcareapis/mgmt/2018-08-20-preview/healthcareapis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/healthcareapis/mgmt/2018-08-20-preview/healthcareapis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/hybridkubernetes/mgmt/2020-01-01-preview/hybridkubernetes` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/hybridkubernetes/mgmt/2020-01-01-preview/hybridkubernetes/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/iothub/mgmt/2018-12-01-preview/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/iothub/mgmt/2018-12-01-preview/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/iothub/mgmt/2019-03-22-preview/devices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/iothub/mgmt/2019-03-22-preview/devices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/iotspaces/mgmt/2017-10-01-preview/iotspaces` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/iotspaces/mgmt/2017-10-01-preview/iotspaces/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/keyvault/mgmt/2020-04-01-preview/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/keyvault/mgmt/2020-04-01-preview/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/keyvault/v7.2-preview/keyvault` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/keyvault/v7.2-preview/keyvault/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/kubernetesconfiguration/mgmt/2019-11-01-preview/kubernetesconfiguration` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/kubernetesconfiguration/mgmt/2019-11-01-preview/kubernetesconfiguration/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/kusto/mgmt/2018-09-07-preview/kusto` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/kusto/mgmt/2018-09-07-preview/kusto/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/logic/mgmt/2015-02-01-preview/logic` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/logic/mgmt/2015-02-01-preview/logic/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/logic/mgmt/2015-08-01-preview/logic` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/logic/mgmt/2015-08-01-preview/logic/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/logic/mgmt/2018-07-01-preview/logic` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/logic/mgmt/2018-07-01-preview/logic/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/machinelearning/mgmt/2016-05-01-preview/commitmentplans` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/machinelearning/mgmt/2016-05-01-preview/commitmentplans/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/machinelearning/mgmt/2016-05-01-preview/webservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/machinelearning/mgmt/2016-05-01-preview/webservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/machinelearning/mgmt/2017-05-01-preview/experimentation` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/machinelearning/mgmt/2017-05-01-preview/experimentation/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/machinelearning/mgmt/2017-08-01-preview/compute` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/machinelearning/mgmt/2017-08-01-preview/compute/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/machinelearningservices/mgmt/2018-03-01-preview/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/machinelearningservices/mgmt/2018-03-01-preview/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/machinelearningservices/mgmt/2020-02-18-preview/machinelearningservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/machinelearningservices/mgmt/2020-02-18-preview/machinelearningservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/managednetwork/mgmt/2019-06-01-preview/managednetwork` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/managednetwork/mgmt/2019-06-01-preview/managednetwork/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/managedservices/mgmt/2018-06-01/managedservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/managedservices/mgmt/2018-06-01/managedservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/managedservices/mgmt/2019-04-01/managedservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/managedservices/mgmt/2019-04-01/managedservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/managementpartner/mgmt/2018-02-01/managementpartner` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/managementpartner/mgmt/2018-02-01/managementpartner/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/maps/mgmt/2020-02-01-preview/maps` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/maps/mgmt/2020-02-01-preview/maps/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mediaservices/mgmt/2018-03-30-preview/media` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/mediaservices/mgmt/2018-03-30-preview/media/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mediaservices/mgmt/2018-06-01-preview/media` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/mediaservices/mgmt/2018-06-01-preview/media/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mediaservices/mgmt/2019-05-01-preview/media` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/mediaservices/mgmt/2019-05-01-preview/media/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mixedreality/mgmt/2019-02-28/mixedreality` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/mixedreality/mgmt/2019-02-28/mixedreality/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mixedreality/mgmt/2020-05-01-preview/mixedreality` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/mixedreality/mgmt/2020-05-01-preview/mixedreality/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/monitor/mgmt/2017-05-01-preview/insights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/monitor/mgmt/2017-05-01-preview/insights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/monitor/mgmt/2018-03-01/insights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/monitor/mgmt/2018-03-01/insights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/monitor/mgmt/2018-09-01/insights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/monitor/mgmt/2018-09-01/insights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/monitor/mgmt/2018-11-01-preview/insights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/monitor/mgmt/2018-11-01-preview/insights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/monitor/mgmt/2019-03-01/insights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/monitor/mgmt/2019-03-01/insights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/monitor/mgmt/2019-06-01/insights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/monitor/mgmt/2019-06-01/insights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/msi/mgmt/2015-08-31-preview/msi` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/msi/mgmt/2015-08-31-preview/msi/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mysql/mgmt/2017-12-01-preview/mysql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/mysql/mgmt/2017-12-01-preview/mysql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/mysql/mgmt/2020-07-01-preview/mysqlflexibleservers` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/mysql/mgmt/2020-07-01-preview/mysqlflexibleservers/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/network/mgmt/2015-05-01-preview/network` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/network/mgmt/2015-05-01-preview/network/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/operationalinsights/mgmt/2015-11-01-preview/operationalinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/operationalinsights/mgmt/2015-11-01-preview/operationalinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/operationalinsights/mgmt/2015-11-01-preview/servicemap` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/operationalinsights/mgmt/2015-11-01-preview/servicemap/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/operationalinsights/mgmt/2020-03-01-preview/operationalinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/operationalinsights/mgmt/2020-03-01-preview/operationalinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/peering/mgmt/2019-08-01-preview/peering` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/peering/mgmt/2019-08-01-preview/peering/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/peering/mgmt/2019-09-01-preview/peering` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/peering/mgmt/2019-09-01-preview/peering/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/peering/mgmt/2020-01-01-preview/peering` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/peering/mgmt/2020-01-01-preview/peering/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/policyinsights/mgmt/2018-07-01-preview/policyinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/policyinsights/mgmt/2018-07-01-preview/policyinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/policyinsights/mgmt/2019-10-01-preview/policyinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/policyinsights/mgmt/2019-10-01-preview/policyinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/portal/mgmt/2015-08-01-preview/portal` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/portal/mgmt/2015-08-01-preview/portal/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/portal/mgmt/2018-10-01-preview/portal` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/portal/mgmt/2018-10-01-preview/portal/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/portal/mgmt/2019-01-01-preview/portal` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/portal/mgmt/2019-01-01-preview/portal/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/postgresql/mgmt/2017-12-01-preview/postgresql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/postgresql/mgmt/2017-12-01-preview/postgresql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/postgresql/mgmt/2020-02-14-preview/postgresqlflexibleservers` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/postgresql/mgmt/2020-02-14-preview/postgresqlflexibleservers/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/provisioningservices/mgmt/2017-08-21-preview/iothub` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/provisioningservices/mgmt/2017-08-21-preview/iothub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/redisenterprise/mgmt/2020-10-01-preview/redisenterprise` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/redisenterprise/mgmt/2020-10-01-preview/redisenterprise/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/reservations/mgmt/2018-06-01/reservations` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/reservations/mgmt/2018-06-01/reservations/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/reservations/mgmt/2019-04-01/reservations` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/reservations/mgmt/2019-04-01/reservations/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/reservations/mgmt/2019-07-19-preview/reservations` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/reservations/mgmt/2019-07-19-preview/reservations/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resourcegraph/mgmt/2018-09-01/resourcegraph` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/resourcegraph/mgmt/2018-09-01/resourcegraph/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resourcemover/mgmt/2019-10-01-preview/resourcemover` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/resourcemover/mgmt/2019-10-01-preview/resourcemover/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2015-10-01-preview/policy` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/resources/mgmt/2015-10-01-preview/policy/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2016-09-01-preview/managedapplications` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/resources/mgmt/2016-09-01-preview/managedapplications/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2017-06-01-preview/policy` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/resources/mgmt/2017-06-01-preview/policy/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2017-08-31-preview/managementgroups` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/resources/mgmt/2017-08-31-preview/managementgroups/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2017-11-01-preview/managementgroups` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/resources/mgmt/2017-11-01-preview/managementgroups/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2018-01-01-preview/managementgroups` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/resources/mgmt/2018-01-01-preview/managementgroups/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2018-03-01-preview/managementgroups` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/resources/mgmt/2018-03-01-preview/managementgroups/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/resources/mgmt/2020-03-01-preview/policy` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/resources/mgmt/2020-03-01-preview/policy/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/scheduler/mgmt/2014-08-01-preview/scheduler` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/scheduler/mgmt/2014-08-01-preview/scheduler/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/security/mgmt/v1.0/security` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/security/mgmt/v1.0/security/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/security/mgmt/v2.0/security` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/security/mgmt/v2.0/security/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/security/mgmt/v3.0/security` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/security/mgmt/v3.0/security/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/securityinsight/mgmt/2019-01-01-preview/securityinsight` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/securityinsight/mgmt/2019-01-01-preview/securityinsight/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/servicebus/mgmt/2018-01-01-preview/servicebus` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/servicebus/mgmt/2018-01-01-preview/servicebus/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/servicefabric/mgmt/2017-07-01-preview/servicefabric` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/servicefabric/mgmt/2017-07-01-preview/servicefabric/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/servicefabric/mgmt/2018-02-01-preview/servicefabric` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/servicefabric/mgmt/2018-02-01-preview/servicefabric/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/servicefabric/mgmt/2019-03-01-preview/servicefabric` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/servicefabric/mgmt/2019-03-01-preview/servicefabric/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/servicefabricmesh/mgmt/2018-07-01-preview/servicefabricmesh` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/servicefabricmesh/mgmt/2018-07-01-preview/servicefabricmesh/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/servicefabricmesh/mgmt/2018-09-01-preview/servicefabricmesh` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/servicefabricmesh/mgmt/2018-09-01-preview/servicefabricmesh/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/signalr/mgmt/2018-03-01-preview/signalr` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/signalr/mgmt/2018-03-01-preview/signalr/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/signalr/mgmt/2020-07-01-preview/signalr` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/signalr/mgmt/2020-07-01-preview/signalr/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/softwareplan/mgmt/2019-06-01-preview/softwareplan` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/softwareplan/mgmt/2019-06-01-preview/softwareplan/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sql/mgmt/2015-05-01-preview/sql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/sql/mgmt/2015-05-01-preview/sql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sql/mgmt/2017-03-01-preview/sql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/sql/mgmt/2017-03-01-preview/sql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sql/mgmt/2017-10-01-preview/sql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/sql/mgmt/2017-10-01-preview/sql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sql/mgmt/2018-06-01-preview/sql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/sql/mgmt/2018-06-01-preview/sql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sql/mgmt/v3.0/sql` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/sql/mgmt/v3.0/sql/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/sqlvirtualmachine/mgmt/2017-03-01-preview/sqlvirtualmachine` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/sqlvirtualmachine/mgmt/2017-03-01-preview/sqlvirtualmachine/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/storage/mgmt/2015-05-01-preview/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/storage/mgmt/2015-05-01-preview/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/storage/mgmt/2020-08-01-preview/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/storage/mgmt/2020-08-01-preview/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/storagecache/mgmt/2019-08-01-preview/storagecache` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/storagecache/mgmt/2019-08-01-preview/storagecache/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/streamanalytics/mgmt/2020-03-01-preview/streamanalytics` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/streamanalytics/mgmt/2020-03-01-preview/streamanalytics/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/subscription/mgmt/2017-11-01-preview/subscription` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/subscription/mgmt/2017-11-01-preview/subscription/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/subscription/mgmt/2018-03-01-preview/subscription` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/subscription/mgmt/2018-03-01-preview/subscription/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/subscription/mgmt/2019-10-01-preview/subscription` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/subscription/mgmt/2019-10-01-preview/subscription/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/support/mgmt/2019-05-01-preview/support` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/support/mgmt/2019-05-01-preview/support/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/synapse/2019-06-01-preview/artifacts` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/synapse/2019-06-01-preview/artifacts/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/synapse/2019-06-01-preview/managedvirtualnetwork` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/synapse/2019-06-01-preview/managedvirtualnetwork/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/synapse/2020-02-01-preview/accesscontrol` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/synapse/2020-02-01-preview/accesscontrol/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/synapse/mgmt/2019-06-01-preview/synapse` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/synapse/mgmt/2019-06-01-preview/synapse/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/timeseriesinsights/mgmt/2017-02-28-preview/timeseriesinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/timeseriesinsights/mgmt/2017-02-28-preview/timeseriesinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/timeseriesinsights/mgmt/2018-08-15-preview/timeseriesinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/timeseriesinsights/mgmt/2018-08-15-preview/timeseriesinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/virtualmachineimagebuilder/mgmt/2018-02-01-preview/virtualmachineimagebuilder` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/virtualmachineimagebuilder/mgmt/2018-02-01-preview/virtualmachineimagebuilder/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/virtualmachineimagebuilder/mgmt/2019-02-01-preview/virtualmachineimagebuilder` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/virtualmachineimagebuilder/mgmt/2019-02-01-preview/virtualmachineimagebuilder/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/virtualmachineimagebuilder/mgmt/2019-05-01-preview/virtualmachineimagebuilder` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/virtualmachineimagebuilder/mgmt/2019-05-01-preview/virtualmachineimagebuilder/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/web/mgmt/2015-08-01-preview/web` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/web/mgmt/2015-08-01-preview/web/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/windowsesu/2019-09-16-preview/windowsesu` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/windowsesu/2019-09-16-preview/windowsesu/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/workloadmonitor/mgmt/2018-08-31-preview/workloadmonitor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/workloadmonitor/mgmt/2018-08-31-preview/workloadmonitor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/preview/workloadmonitor/mgmt/2020-01-13-preview/workloadmonitor` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/preview/workloadmonitor/mgmt/2020-01-13-preview/workloadmonitor/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/privatedns/mgmt/2018-09-01/privatedns` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/privatedns/mgmt/2018-09-01/privatedns/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/provisioningservices/mgmt/2017-11-15/iothub` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/provisioningservices/mgmt/2017-11-15/iothub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/provisioningservices/mgmt/2018-01-22/iothub` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/provisioningservices/mgmt/2018-01-22/iothub/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2016-06-01/backup` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/recoveryservices/mgmt/2016-06-01/backup/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2016-06-01/recoveryservices` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/recoveryservices/mgmt/2016-06-01/recoveryservices/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2016-08-10/siterecovery` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/recoveryservices/mgmt/2016-08-10/siterecovery/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2016-12-01/backup` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/recoveryservices/mgmt/2016-12-01/backup/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2018-01-10/siterecovery` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/recoveryservices/mgmt/2018-01-10/siterecovery/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2018-07-10/siterecovery` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/recoveryservices/mgmt/2018-07-10/siterecovery/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2019-05-13/backup` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/recoveryservices/mgmt/2019-05-13/backup/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2019-06-15/backup` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/recoveryservices/mgmt/2019-06-15/backup/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/recoveryservices/mgmt/2020-02-02/backup` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/recoveryservices/mgmt/2020-02-02/backup/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redhatopenshift/mgmt/2020-04-30/redhatopenshift` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/redhatopenshift/mgmt/2020-04-30/redhatopenshift/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2015-08-01/redis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/redis/mgmt/2015-08-01/redis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2016-04-01/redis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/redis/mgmt/2016-04-01/redis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2017-02-01/redis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/redis/mgmt/2017-02-01/redis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2017-10-01/redis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/redis/mgmt/2017-10-01/redis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2018-03-01/redis` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/redis/mgmt/2018-03-01/redis/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/relay/mgmt/2016-07-01/relay` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/relay/mgmt/2016-07-01/relay/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/relay/mgmt/2017-04-01/relay` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/relay/mgmt/2017-04-01/relay/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/reservations/mgmt/2017-11-01/reservations` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/reservations/mgmt/2017-11-01/reservations/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resourcehealth/mgmt/2015-01-01/resourcehealth` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resourcehealth/mgmt/2015-01-01/resourcehealth/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resourcehealth/mgmt/2017-07-01/resourcehealth` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resourcehealth/mgmt/2017-07-01/resourcehealth/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2015-01-01/locks` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2015-01-01/locks/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2015-11-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2015-11-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2015-11-01/subscriptions` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2015-11-01/subscriptions/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2015-12-01/features` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2015-12-01/features/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2016-02-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2016-02-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2016-04-01/policy` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2016-04-01/policy/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2016-06-01/subscriptions` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2016-06-01/subscriptions/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2016-07-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2016-07-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2016-09-01/links` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2016-09-01/links/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2016-09-01/locks` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2016-09-01/locks/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2016-09-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2016-09-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2016-12-01/policy` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2016-12-01/policy/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2017-05-10/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2017-05-10/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2017-09-01/managedapplications` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2017-09-01/managedapplications/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2018-02-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2018-02-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2018-03-01/policy` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2018-03-01/policy/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2018-05-01/policy` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2018-05-01/policy/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2018-05-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2018-05-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2018-06-01/managedapplications` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2018-06-01/managedapplications/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2018-06-01/subscriptions` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2018-06-01/subscriptions/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-01-01/policy` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2019-01-01/policy/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-03-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2019-03-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-05-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2019-05-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-06-01/policy` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2019-06-01/policy/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-06-01/subscriptions` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2019-06-01/subscriptions/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-07-01/features` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2019-07-01/features/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-07-01/managedapplications` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2019-07-01/managedapplications/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-09-01/policy` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2019-09-01/policy/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-10-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2019-10-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2019-11-01/subscriptions` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2019-11-01/subscriptions/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2020-06-01/resources` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/resources/mgmt/2020-06-01/resources/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/scheduler/mgmt/2016-01-01/scheduler` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/scheduler/mgmt/2016-01-01/scheduler/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/scheduler/mgmt/2016-03-01/scheduler` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/scheduler/mgmt/2016-03-01/scheduler/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/search/mgmt/2020-03-13/search` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/search/mgmt/2020-03-13/search/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/securityinsight/mgmt/v1.0/securityinsight` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/securityinsight/mgmt/v1.0/securityinsight/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/servicebus/mgmt/2015-08-01/servicebus` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/servicebus/mgmt/2015-08-01/servicebus/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/servicebus/mgmt/2017-04-01/servicebus` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/servicebus/mgmt/2017-04-01/servicebus/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/servicefabric/mgmt/2016-09-01/servicefabric` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/servicefabric/mgmt/2016-09-01/servicefabric/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/servicefabric/mgmt/2019-03-01/servicefabric` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/servicefabric/mgmt/2019-03-01/servicefabric/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/signalr/mgmt/2018-10-01/signalr` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/signalr/mgmt/2018-10-01/signalr/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/signalr/mgmt/2020-05-01/signalr` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/signalr/mgmt/2020-05-01/signalr/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2019-04-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/storage/mgmt/2019-04-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2019-06-01/storage` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/storage/mgmt/2019-06-01/storage/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagecache/mgmt/2019-11-01/storagecache` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/storagecache/mgmt/2019-11-01/storagecache/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagecache/mgmt/2020-03-01/storagecache` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/storagecache/mgmt/2020-03-01/storagecache/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storageimportexport/mgmt/2016-11-01/storageimportexport` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/storageimportexport/mgmt/2016-11-01/storageimportexport/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storageimportexport/mgmt/2020-08-01/storageimportexport` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/storageimportexport/mgmt/2020-08-01/storageimportexport/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2018-04-02/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/storagesync/mgmt/2018-04-02/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2018-07-01/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/storagesync/mgmt/2018-07-01/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2018-10-01/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/storagesync/mgmt/2018-10-01/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2019-02-01/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/storagesync/mgmt/2019-02-01/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2019-06-01/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/storagesync/mgmt/2019-06-01/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2019-10-01/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/storagesync/mgmt/2019-10-01/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storagesync/mgmt/2020-03-01/storagesync` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/storagesync/mgmt/2020-03-01/storagesync/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storsimple1200series/mgmt/2016-10-01/storsimple` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/storsimple1200series/mgmt/2016-10-01/storsimple/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/storsimple8000series/mgmt/2017-06-01/storsimple` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/storsimple8000series/mgmt/2017-06-01/storsimple/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/streamanalytics/mgmt/2016-03-01/streamanalytics` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/streamanalytics/mgmt/2016-03-01/streamanalytics/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/subscription/mgmt/2020-09-01/subscription` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/subscription/mgmt/2020-09-01/subscription/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/support/mgmt/2020-04-01/support` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/support/mgmt/2020-04-01/support/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/timeseriesinsights/mgmt/2017-11-15/timeseriesinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/timeseriesinsights/mgmt/2017-11-15/timeseriesinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/timeseriesinsights/mgmt/2020-05-15/timeseriesinsights` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/timeseriesinsights/mgmt/2020-05-15/timeseriesinsights/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/virtualmachineimagebuilder/mgmt/2020-02-01/virtualmachineimagebuilder` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/virtualmachineimagebuilder/mgmt/2020-02-01/virtualmachineimagebuilder/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/vmwarecloudsimple/mgmt/2019-04-01/vmwarecloudsimple` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/vmwarecloudsimple/mgmt/2019-04-01/vmwarecloudsimple/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/web/mgmt/2016-09-01/web` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/web/mgmt/2016-09-01/web/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/web/mgmt/2018-02-01/web` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/web/mgmt/2018-02-01/web/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/web/mgmt/2019-08-01/web` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/web/mgmt/2019-08-01/web/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/web/mgmt/2020-06-01/web` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/web/mgmt/2020-06-01/web/CHANGELOG.md) |
| `github.com/Azure/azure-sdk-for-go/services/windowsiot/mgmt/2019-06-01/windowsiot` | [details](https://github.com/Azure/azure-sdk-for-go/blob/v49.0.0/services/windowsiot/mgmt/2019-06-01/windowsiot/CHANGELOG.md) |

## `v48.2.0`

### New Packages

| Package Name | API Version |
| ---: | :---: |
| attestation | 2020-10-01 |
| containerservice | 2020-11-01 |
| redisenterprise | 2020-10-01-preview |

Generated from https://github.com/Azure/azure-rest-api-specs/tree/48e7415267518cc2b92f6e6002466f676bdef1a5

## `v48.1.0`

### New Packages

| Package Name | API Version |
| ---: | :---: |
| storage | 2020-08-01-preview |

Generated from https://github.com/Azure/azure-rest-api-specs/tree/4c1dfe438cfb73300281d05583324aee2fc96015

## `v48.0.0`

### New Packages

| Package Name | API Version |
| ---: | :---: |
| confluent | 2020-03-01-preview |
| healthcareapis | 2020-03-30 |
| media | 2020-05-01 |
| mysqlflexibleservers | 2020-07-01-preview |
| netapp | 2020-07-01 |
| policy | 2020-03-01-preview |
| securityinsight | v1.0 |

### Breaking Changes

| Package Name | API Version |
| ---: | :---: |
| digitaltwins | 2020-10-31 |
| hdinsight | 2018-06-01 |
| workloadmonitor | 2020-01-13-preview |

### Package Renames

- package `github.com/Azure/azure-sdk-for-go/services/preview/securityinsight/mgmt/v1.0/securityinsight` has been renamed to `github.com/Azure/azure-sdk-for-go/services/securityinsight/mgmt/v1.0/securityinsight`

Generated from https://github.com/Azure/azure-rest-api-specs/tree/c174d654d4cd9f45e929a31480e19ada7baa6924

## `v47.1.0`

### New Packages

| Package Name | API Version |
| ---: | :---: |
| alertsmanagement | 2019-06-01-preview |
| azurestackhci | 2020-10-01 |
| compute | 2020-06-30 |
| healthcareapis | 2020-03-15 |
| keyvault | 2020-04-01-preview |
| kusto | 2020-09-18 |
| workloadmonitor | 2020-01-13-preview |

### Updated Packages

| Package Name | API Version |
| ---: | :---: |
| appplatform | 2019-05-01-preview |

### Breaking Changes

| Package Name | API Version |
| ---: | :---: |
| operationalinsights | 2020-03-01-preview |
| synapse | 2019-06-01-preview |

Generated from https://github.com/Azure/azure-rest-api-specs/tree/c0471d8a805d8ae7f55946ea7020dd56282b23f4
Generated from https://github.com/Azure/azure-rest-api-specs/tree/6e52d5c8e77a02fb333a991de8f1ca630ba9ff3e (synapse)

## `v47.0.0`

### New Packages

| Package Name | API Version |
| ---: | :---: |
| datadog | 2020-02-01-preview |

### Updated Packages

| Package Name | API Version |
| ---: | :---: |
| cognitiveservices | 2017-04-18 |

### Breaking Changes

| Package Name | API Version |
| ---: | :---: |
| containerservice | 2020-09-01 |
| datafactory | 2018-06-01 |
| security | v1.0<br/>v2.0<br/>v3.0 |

### Package Renames

- package `github.com/Azure/azure-sdk-for-go/services/preview/regionmove/mgmt/2019-10-01-preview/regionmove` has been renamed to `github.com/Azure/azure-sdk-for-go/services/preview/resourcemover/mgmt/2019-10-01-preview/resourcemover`

- package `github.com/Azure/azure-sdk-for-go/services/preview/appconfiguration/mgmt/2020-06-01/appconfiguration` has been renamed to `github.com/Azure/azure-sdk-for-go/services/appconfiguration/mgmt/2020-06-01/appconfiguration`

- package `github.com/Azure/azure-sdk-for-go/services/billing/mgmt/2020-05-01/billing` has been renamed to `github.com/Azure/azure-sdk-for-go/services/preview/billing/mgmt/2020-05-01-preview/billing`

- package `github.com/Azure/azure-sdk-for-go/services/preview/cdn/mgmt/2020-04-15/cdn` has been renamed to `github.com/Azure/azure-sdk-for-go/services/cdn/mgmt/2020-04-15/cdn`

- package `github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2019-02-01/containerservice` has been renamed to `github.com/Azure/azure-sdk-for-go/services/preview/containerservice/mgmt/2019-02-01-preview/containerservice`

- package `github.com/Azure/azure-sdk-for-go/services/cosmos-db/mgmt/2019-12-12/documentdb` has been renamed to `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2019-12-12-preview/documentdb`

- package `github.com/Azure/azure-sdk-for-go/services/cosmos-db/mgmt/2020-03-01/documentdb` has been renamed to `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2020-03-01-preview/documentdb`

- package `github.com/Azure/azure-sdk-for-go/services/cosmos-db/mgmt/2020-04-01/documentdb` has been renamed to `github.com/Azure/azure-sdk-for-go/services/preview/cosmos-db/mgmt/2020-04-01-preview/documentdb`

- package `github.com/Azure/azure-sdk-for-go/services/preview/hdinsight/mgmt/2018-06-01-preview/hdinsight` has been renamed to `github.com/Azure/azure-sdk-for-go/services/hdinsight/mgmt/2018-06-01/hdinsight`

- package `github.com/Azure/azure-sdk-for-go/services/preview/keyvault/v7.1-preview/keyvault` has been renamed to `github.com/Azure/azure-sdk-for-go/services/keyvault/v7.1/keyvault`

- package `github.com/Azure/azure-sdk-for-go/services/preview/operationalinsights/mgmt/2020-08-01/operationalinsights` has been renamed to `github.com/Azure/azure-sdk-for-go/services/operationalinsights/mgmt/2020-08-01/operationalinsights`

- package `github.com/Azure/azure-sdk-for-go/services/policyinsights/mgmt/2019-10-01/policyinsights` has been renamed to `github.com/Azure/azure-sdk-for-go/services/preview/policyinsights/mgmt/2019-10-01-preview/policyinsights`

- package `github.com/Azure/azure-sdk-for-go/services/servicefabric/mgmt/2018-02-01/servicefabric` has been renamed to `github.com/Azure/azure-sdk-for-go/services/preview/servicefabric/mgmt/2018-02-01-preview/servicefabric`

- package `github.com/Azure/azure-sdk-for-go/services/preview/hybridcompute/mgmt/2019-12-12/hybridcompute` has been renamed to `github.com/Azure/azure-sdk-for-go/services/hybridcompute/mgmt/2019-12-12/hybridcompute`

- package `github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2018-07-01/storage` has been renamed to `github.com/Azure/azure-sdk-for-go/services/preview/storage/mgmt/2018-07-01-preview/storage`

- package `github.com/Azure/azure-sdk-for-go/services/trafficmanager/mgmt/2018-02-01/trafficmanager` has been renamed to `github.com/Azure/azure-sdk-for-go/services/preview/trafficmanager/mgmt/2018-02-01-preview/trafficmanager`

Generated from https://github.com/Azure/azure-rest-api-specs/tree/93106ff722d481c30a9a4ae089564635c10e9401

## `v46.4.0`

### New Packages

| Package Name | API Version |
| ---: | :---: |
| postgresqlflexibleservers | 2020-02-14-preview |

### Updated Packages

| Package Name | API Version |
| ---: | :---: |
| compute | 2020-06-01 |
| subscription | 2020-09-01 |

### Breaking Changes

| Package Name | API Version |
| ---: | :---: |
| synapse | 2019-06-01-preview |

Generated from https://github.com/Azure/azure-rest-api-specs/tree/2a4dc288266c1f1829d6c1d5ef877b120fa2abd7

## `v46.3.0`

### New Packages

| Package Name | API Version |
| ---: | :---: |
| communication | 2020-08-20-preview |

Generated from https://github.com/Azure/azure-rest-api-specs/tree/be39f5abd3dc4cf6db384f688e0dd18dd907d04b

## `v46.2.0`

### New Packages

| Package Name | API Version |
| ---: | :---: |
| containerservice | 2020-09-01 |
| digitaltwins | 2020-10-31 |
| streamanalytics | 2020-03-01-preview |

### Breaking Changes

| Package Name | API Version |
| ---: | :---: |
| hybridkubernetes | 2020-01-01-preview |

Generated from https://github.com/Azure/azure-rest-api-specs/tree/1b04f5fce19cd330dcc0eec11a98e499c0cda50d

## `v46.1.0`

### New Packages

| Package Name | API Version |
| ---: | :---: |
| hybridcompute | 2020-08-02 |
| mariadb | 2020-01-01 |
| netapp | 2020-06-01 |
| operationalinsights | 2020-08-01 |
| subscription | 2020-09-01 |

### Updated Packages

| Package Name | API Version |
| ---: | :---: |
| datafactory | 2018-06-01 |
| mysql | 2020-01-01 |
| network | 2020-05-01 |

## `v46.0.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| appplatform | 2020-07-01 |
| automanage | 2020-06-30-preview |
| containerservice | 2020-07-01 |
| network | 2020-06-01 |
| regionmove | 2019-10-01-preview |
| resources | 2020-06-01 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| appconfiguration | 2020-06-01 |
| documentdb | 2020-04-01 |
| hdinsight | 2018-06-01-preview |
| network | 2020-05-01 |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| mysql | 2020-01-01 |
| postgresql | 2020-01-01 |
| servicebus | 2018-01-01-preview |
| storagecache | 2019-11-01<br/>2020-03-01 |

## `v45.1.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| resources | 2019-10-01 |
| web | 2020-06-01 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| appplatform | 2019-05-01-preview |
| cdn | 2020-04-15 |

## `v45.0.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| hybridnetwork | 2020-01-01-preview |
| managementgroups | 2020-05-01 |
| monitoring | 2019-11-01-preview |
| netapp | 2020-02-01 |
| signalr | 2020-07-01-preview |
| storageimportexport | 2020-08-01 |
| storagetables | 2019-02-02-preview |
| timeseriesinsights | 2020-05-15 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| aad | 2017-01-01<br/>2017-06-01 |
| account | 2015-10-01-preview<br/>2015-10-01-preview<br/>2016-11-01 |
| adhybridhealthservice | 2014-01-01 |
| alertsmanagement | 2018-05-05<br/>2018-05-05-preview<br/>2019-03-01<br/>2019-05-05 |
| analysisservices | 2016-05-16<br/>2017-07-14<br/>2017-08-01 |
| apimanagement | 2016-07-07<br/>2016-10-10<br/>2017-03-01<br/>2017-03-01<br/>2018-01-01<br/>2018-06-01-preview<br/>2019-12-01-preview |
| appconfiguration | 2019-02-01-preview<br/>2019-10-01<br/>2019-11-01-preview<br/>2020-06-01 |
| appplatform | 2019-05-01-preview |
| artifacts | 2019-06-01-preview |
| authorization | 2015-07-01 |
| automation | 2015-10-31<br/>2017-05-15-preview<br/>2018-01-15-preview<br/>2018-06-30-preview |
| avs | 2019-08-09-preview<br/>2020-03-20 |
| azurestack | 2017-06-01 |
| azurestackhci | 2020-03-01-preview |
| backup | 2016-12-01<br/>2017-07-01<br/>2019-05-13 |
| batch | 2015-12-01<br/>2017-09-01<br/>2018-03-01.6.1<br/>2018-08-01.7.0<br/>2018-12-01<br/>2019-04-01<br/>2019-08-01<br/>2020-03-01 |
| batchai | 2017-09-01-preview<br/>2018-03-01<br/>2018-05-01 |
| billing | 2017-02-27-preview<br/>2017-04-24-preview<br/>2018-03-01-preview<br/>2018-11-01-preview |
| blockchain | 2018-06-01-preview |
| blueprint | 2018-11-01-preview |
| botservice | 2017-12-01<br/>2018-07-12 |
| catalog | 2015-10-01-preview<br/>2016-11-01-preview |
| cdn | 2015-06-01<br/>2016-04-02<br/>2016-10-02<br/>2017-04-02<br/>2017-10-12<br/>2019-04-15 |
| cognitiveservices | 2016-02-01-preview<br/>2017-04-18 |
| commitmentplans | 2016-05-01-preview |
| compute | 2015-06-15<br/>2016-03-30<br/>2016-04-30-preview<br/>2017-03-30<br/>2017-08-01-preview<br/>2017-12-01<br/>2018-04-01<br/>2018-06-01<br/>2018-10-01<br/>2019-03-01<br/>2019-07-01<br/>2019-12-01<br/>2020-06-01 |
| computervision | v2.0<br/>v2.1<br/>v3.0 |
| consumption | 2017-04-24-preview<br/>2017-11-30<br/>2017-12-30-preview<br/>2018-01-31<br/>2018-03-31<br/>2018-05-31<br/>2018-06-30<br/>2018-08-31<br/>2018-10-01<br/>2019-01-01 |
| containerinstance | 2017-08-01-preview<br/>2017-10-01-preview<br/>2017-12-01-preview<br/>2018-02-01-preview<br/>2018-04-01<br/>2018-06-01<br/>2018-09-01<br/>2018-10-01<br/>2019-12-01 |
| containerregistry | 2016-06-27-preview<br/>2017-03-01<br/>2017-06-01-preview<br/>2017-10-01<br/>2018-02-01<br/>2018-09-01<br/>2019-04-01<br/>2019-05-01<br/>2019-05-01-preview |
| containerservice | 2015-11-01-preview<br/>2016-03-30<br/>2016-09-30<br/>2017-01-31<br/>2017-07-01<br/>2017-08-31<br/>2017-09-30<br/>2018-03-31<br/>2018-08-01-preview<br/>2018-09-30-preview<br/>2019-02-01<br/>2019-04-30<br/>2019-06-01<br/>2019-08-01<br/>2019-09-30-preview<br/>2019-10-01<br/>2019-10-27-preview<br/>2019-11-01<br/>2020-01-01<br/>2020-02-01<br/>2020-03-01<br/>2020-04-01<br/>2020-06-01 |
| costmanagement | 2018-05-31<br/>2018-08-01-preview<br/>2019-01-01<br/>2019-03-01<br/>2019-10-01 |
| customerinsights | 2017-01-01<br/>2017-04-26 |
| customerlockbox | 2018-02-28-preview |
| customimagesearch | v1.0 |
| customproviders | 2018-09-01-preview |
| customsearch | v1.0 |
| databox | 2018-01-01<br/>2019-09-01 |
| databoxedge | 2019-03-01<br/>2019-07-01<br/>2019-08-01 |
| databricks | 2018-04-01 |
| datafactory | 2017-09-01-preview |
| datamigration | 2017-11-15-preview<br/>2018-03-31-preview<br/>2018-04-19<br/>2018-07-15-preview |
| deploymentmanager | 2018-09-01-preview<br/>2019-11-01-preview |
| desktopvirtualization | 2019-01-23-preview<br/>2019-09-24-preview<br/>2019-12-10-preview |
| devices | 2016-02-03<br/>2017-01-19<br/>2017-07-01<br/>2018-01-22<br/>2018-04-01<br/>2018-12-01-preview<br/>2019-03-22-preview<br/>2020-03-01 |
| devops | 2019-07-01-preview |
| devspaces | 2019-04-01 |
| digitaltwins | 2020-03-01-preview<br/>2020-05-31 |
| dns | 2016-04-01<br/>2017-09-01<br/>2017-10-01<br/>2018-03-01-preview<br/>2018-05-01 |
| documentdb | 2015-04-08<br/>2019-08-01<br/>2019-08-01-preview<br/>2019-12-12<br/>2020-03-01<br/>2020-04-01 |
| dtl | 2016-05-15<br/>2018-09-15 |
| enterpriseknowledgegraphservice | 2018-12-03 |
| entitysearch | v1.0 |
| eventgrid | 2017-06-15-preview<br/>2017-09-15-preview<br/>2018-01-01<br/>2018-01-01<br/>2018-05-01-preview<br/>2018-09-15-preview<br/>2019-01-01<br/>2019-02-01-preview<br/>2019-06-01<br/>2020-01-01-preview<br/>2020-04-01-preview<br/>2020-06-01 |
| eventhub | 2015-08-01<br/>2017-04-01<br/>2018-01-01-preview |
| experimentation | 2017-05-01-preview |
| features | 2019-07-01 |
| filesystem | 2015-10-01-preview<br/>2016-11-01 |
| frontdoor | 2018-08-01-preview<br/>2019-04-01<br/>2019-05-01<br/>2019-10-01<br/>2019-11-01<br/>2020-01-01<br/>2020-04-01<br/>2020-05-01 |
| hanaonazure | 2017-11-03-preview |
| hardwaresecuritymodules | 2018-10-31-preview |
| hdinsight | 2015-03-01-preview<br/>2018-06-01-preview<br/>2018-11-01-preview |
| healthcareapis | 2018-08-20-preview<br/>2019-09-16 |
| hybridcompute | 2019-12-12 |
| hybridkubernetes | 2020-01-01-preview |
| imagesearch | v1.0 |
| insights | 2015-05-01<br/>2017-05-01-preview<br/>2018-03-01<br/>2018-09-01<br/>2018-11-01-preview<br/>2019-03-01<br/>2019-06-01 |
| iotcentral | 2018-09-01 |
| iothub | 2017-08-21-preview<br/>2017-11-15<br/>2018-01-22 |
| iotspaces | 2017-10-01-preview |
| job | 2015-11-01-preview<br/>2016-03-20-preview<br/>2016-11-01<br/>2017-09-01-preview |
| kubernetesconfiguration | 2019-11-01-preview |
| kusto | 2018-09-07-preview<br/>2019-01-21<br/>2019-05-15<br/>2019-09-07<br/>2019-11-09<br/>2020-02-15<br/>2020-06-14 |
| labservices | 2018-10-15 |
| links | 2016-09-01 |
| logic | 2015-02-01-preview<br/>2015-08-01-preview<br/>2016-06-01<br/>2018-07-01-preview<br/>2019-05-01 |
| machinelearningservices | 2018-03-01-preview<br/>2018-11-19<br/>2019-05-01<br/>2019-06-01<br/>2019-11-01<br/>2020-01-01<br/>2020-02-18-preview<br/>2020-03-01<br/>2020-04-01 |
| managedapplications | 2016-09-01-preview<br/>2017-09-01<br/>2018-06-01<br/>2019-07-01 |
| managednetwork | 2019-06-01-preview |
| managedservices | 2018-06-01<br/>2019-04-01<br/>2019-06-01 |
| managedvirtualnetwork | 2019-06-01-preview |
| managementgroups | 2017-08-31-preview<br/>2017-11-01-preview<br/>2018-01-01-preview<br/>2018-03-01-preview<br/>2019-11-01<br/>2020-02-01 |
| maps | 2017-01-01-preview<br/>2018-05-01<br/>2020-02-01-preview |
| mariadb | 2018-06-01 |
| marketplaceordering | 2015-06-01 |
| media | 2015-10-01<br/>2018-03-30-preview<br/>2018-06-01-preview<br/>2018-07-01<br/>2019-05-01-preview |
| migrate | 2018-02-02<br/>2018-09-01-preview<br/>2020-01-01 |
| mixedreality | 2020-05-01-preview |
| mysql | 2017-12-01<br/>2017-12-01-preview |
| netapp | 2017-08-15<br/>2019-05-01<br/>2019-06-01<br/>2019-07-01<br/>2019-08-01<br/>2019-10-01<br/>2019-11-01 |
| network | 2015-05-01-preview<br/>2015-06-15<br/>2016-03-30<br/>2016-06-01<br/>2016-09-01<br/>2016-12-01<br/>2017-03-01<br/>2017-06-01<br/>2017-08-01<br/>2017-09-01<br/>2017-10-01<br/>2017-11-01<br/>2018-01-01<br/>2018-02-01<br/>2018-04-01<br/>2018-06-01<br/>2018-07-01<br/>2018-08-01<br/>2018-10-01<br/>2018-11-01<br/>2018-12-01<br/>2019-02-01<br/>2019-04-01<br/>2019-06-01<br/>2019-07-01<br/>2019-08-01<br/>2019-09-01<br/>2019-11-01<br/>2019-12-01 |
| newssearch | v1.0 |
| notificationhubs | 2017-04-01 |
| operationalinsights | 2015-03-20<br/>2015-11-01-preview<br/>2020-03-01-preview |
| operationsmanagement | 2015-11-01-preview |
| peering | 2019-08-01-preview<br/>2019-09-01-preview<br/>2020-01-01-preview<br/>2020-04-01 |
| personalizer | v1.0 |
| policy | 2018-05-01<br/>2019-01-01<br/>2019-06-01<br/>2019-09-01 |
| policyinsights | 2018-07-01-preview<br/>2019-10-01 |
| postgresql | 2017-12-01<br/>2017-12-01-preview |
| powerbidedicated | 2017-10-01 |
| privatedns | 2018-09-01 |
| qnamakerruntime | v4.0 |
| recoveryservices | 2016-06-01 |
| redis | 2017-02-01<br/>2017-10-01<br/>2018-03-01 |
| relay | 2016-07-01<br/>2017-04-01 |
| reservations | 2017-11-01<br/>2018-06-01<br/>2019-04-01<br/>2019-07-19-preview |
| resourcegraph | 2018-09-01 |
| resources | 2015-11-01<br/>2016-02-01<br/>2016-07-01<br/>2016-09-01<br/>2017-05-10<br/>2018-02-01<br/>2018-05-01<br/>2019-03-01<br/>2019-05-01 |
| scheduler | 2014-08-01-preview<br/>2016-01-01<br/>2016-03-01 |
| search | 2015-02-28<br/>2015-08-19<br/>2020-03-13 |
| security | v1.0<br/>v2.0 |
| securityinsight | 2019-01-01-preview |
| servicebus | 2015-08-01<br/>2017-04-01<br/>2018-01-01-preview |
| servicefabric | 2016-09-01<br/>2017-07-01-preview<br/>2018-02-01<br/>2019-03-01<br/>2019-03-01-preview<br/>6.3<br/>6.4<br/>6.5<br/>7.0 |
| servicefabricmesh | 2018-07-01-preview<br/>2018-09-01-preview |
| servicemap | 2015-11-01-preview |
| signalr | 2018-03-01-preview<br/>2018-10-01<br/>2020-05-01 |
| siterecovery | 2016-08-10<br/>2018-01-10<br/>2018-07-10 |
| spellcheck | v1.0 |
| sql | 2014-04-01<br/>2015-05-01-preview<br/>2017-03-01-preview<br/>2017-10-01-preview<br/>2018-06-01-preview |
| sqlvirtualmachine | 2017-03-01-preview |
| storage | 2016-01-01<br/>2016-05-01<br/>2016-12-01<br/>2017-06-01<br/>2017-10-01<br/>2018-02-01<br/>2018-03-01-preview<br/>2018-07-01<br/>2018-11-01<br/>2019-04-01<br/>2019-06-01 |
| storagecache | 2019-08-01-preview<br/>2020-03-01 |
| storageimportexport | 2016-11-01 |
| storagesync | 2018-04-02<br/>2018-07-01<br/>2018-10-01<br/>2019-02-01<br/>2019-06-01<br/>2019-10-01<br/>2020-03-01 |
| storsimple | 2016-10-01<br/>2017-06-01 |
| streamanalytics | 2016-03-01 |
| subscription | 2017-11-01-preview<br/>2018-03-01-preview<br/>2019-10-01-preview |
| subscriptions | 2015-11-01<br/>2016-06-01<br/>2018-06-01<br/>2019-06-01<br/>2019-11-01 |
| support | 2019-05-01-preview<br/>2020-04-01 |
| synapse | 2019-06-01-preview |
| textanalytics | v2.0<br/>v2.1 |
| timeseriesinsights | 2017-02-28-preview<br/>2017-11-15<br/>2018-08-15-preview |
| trafficmanager | 2017-05-01<br/>2017-09-01-preview<br/>2018-02-01<br/>2018-03-01<br/>2018-04-01 |
| training | customvision<br/>customvision<br/>customvision<br/>customvision<br/>customvision<br/>customvision<br/>customvision |
| translatortext | v1.0_preview.1 |
| videosearch | v1.0 |
| virtualmachineimagebuilder | 2018-02-01-preview<br/>2019-02-01-preview<br/>2019-05-01-preview<br/>2020-02-01 |
| visualsearch | v1.0 |
| vmwarecloudsimple | 2019-04-01 |
| web | 2015-08-01-preview<br/>2016-09-01<br/>2018-02-01<br/>2019-08-01 |
| websearch | v1.0 |
| webservices | 2017-01-01 |
| windowsesu | 2019-09-16-preview |
| windowsiot | 2019-06-01 |
| workloadmonitor | 2018-08-31-preview |
| workspaces | 2016-04-01<br/>2019-10-01 |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| apimanagement | 2019-01-01<br/>2019-12-01 |
| azuredata | 2019-07-24-preview |
| backup | 2019-06-15<br/>2020-02-02 |
| billing | 2020-05-01 |
| cdn | 2020-04-15 |
| containerregistry | 2019-06-01-preview<br/>2019-12-01-preview |
| datafactory | 2018-06-01 |
| datashare | 2018-11-01-preview<br/>2019-11-01 |
| documentdb | 2020-06-01-preview |
| mysql | 2020-01-01 |
| network | 2020-03-01<br/>2020-04-01<br/>2020-05-01 |
| postgresql | 2020-01-01 |
| security | v3.0 |
| securityinsight | v1.0 |
| sql | v3.0 |
| storagecache | 2019-11-01 |

### NOTE

- All the enums in a package will be in a separated `enums.go` file.
- Paginated result will automatically skip empty pages.
- All models that have READ-ONLY fields will have their own custom marshaler.

## `v44.2.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| azurestackhci | 2020-03-01-preview |
| managedvirtualnetwork | 2019-06-01-preview |
| monitoring | 2019-11-01-preview |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| hdinsight | 2018-06-01-preview |

## `v44.1.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| avs | 2020-03-20 |
| kusto | 2020-06-14 |
| storagesync | 2020-03-01 |

## `v44.0.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| appconfiguration | 2020-06-01 |
| avs | 2019-08-09-preview |
| billing | 2020-05-01 |
| compute | 2020-06-01 |
| containerinstance | 2019-12-01 |
| containerservice | 2020-06-01 |
| digitaltwins | 2020-05-31 |
| documentdb | 2020-06-01-preview |
| hardwaresecuritymodules | 2018-10-31-preview |
| search | 2020-03-13 |
| training | customvision |
| translatortext | v1.0_preview.1 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| attestation | 2018-09-01-preview |
| azuredata | 2019-07-24-preview |
| backup | 2019-06-15<br/>2020-02-02 |
| cdn | 2020-04-15 |
| containerregistry | 2019-06-01-preview<br/>2019-12-01-preview |
| face | v1.0 |
| frontdoor | 2020-05-01 |
| hdinsight | 2015-03-01-preview<br/>2018-06-01-preview |
| insights | 2015-05-01 |
| iotcentral | 2018-09-01 |
| siterecovery | 2018-07-10 |
| sql | v3.0 |
| storage | 2019-04-01<br/>2019-06-01 |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| accesscontrol | 2020-02-01-preview |
| alertsmanagement | 2019-03-01 |
| apimanagement | 2019-12-01 |
| azurestack | 2017-06-01 |
| blueprint | 2018-11-01-preview |
| compute | 2019-07-01 |
| databricks | 2018-04-01 |
| datafactory | 2018-06-01 |
| desktopvirtualization | 2019-01-23-preview<br/>2019-09-24-preview<br/>2019-12-10-preview |
| eventhub | 2018-01-01-preview |
| formrecognizer | v1.0 |
| hdinsight | 2018-11-01-preview |
| machinelearningservices | 2018-03-01-preview<br/>2018-11-19<br/>2019-05-01<br/>2019-06-01<br/>2019-11-01<br/>2020-01-01<br/>2020-02-18-preview<br/>2020-03-01<br/>2020-04-01 |
| migrate | 2018-09-01-preview |
| network | 2019-04-01<br/>2019-06-01<br/>2019-07-01<br/>2019-08-01<br/>2019-09-01<br/>2019-11-01<br/>2019-12-01<br/>2020-03-01<br/>2020-04-01<br/>2020-05-01 |
| operationalinsights | 2020-03-01-preview |
| reservations | 2019-07-19-preview |
| security | v1.0<br/>v2.0<br/>v3.0 |
| sql | 2014-04-01<br/>2015-05-01-preview<br/>2017-03-01-preview |

### Removed Packages

| Package Name | API Version |
| -----------: | :---------: |
| billing | 2020-05-01-preview |
| vmware | 2019-08-09-preview |

## `v43.3.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| accesscontrol | 2020-02-01-preview |
| artifacts | 2019-06-01-preview |
| spark | 2019-11-01-preview |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| synapse | 2019-06-01-preview |

## `v43.2.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| documentdb | 2020-04-01 |
| eventgrid | 2020-06-01 |
| network | 2020-05-01 |
| vmware | 2019-08-09-preview |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| hdinsight | 2018-06-01-preview |

## `v43.1.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| keyvault | 2019-09-01 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| resourcegraph | 2018-09-01 |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| digitaltwins | 2020-03-01-preview |

## `v43.0.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| azuredata | 2019-07-24-preview |
| billing | 2020-05-01-preview |
| cdn | 2020-04-15 |
| computervision | v3.0 |
| containerservice | 2020-04-01 |
| desktopvirtualization | 2019-09-24-preview<br/>2019-12-10-preview |
| devices | 2020-03-01 |
| frontdoor | 2020-04-01<br/>2020-05-01 |
| machinelearningservices | 2020-04-01 |
| mixedreality | 2020-05-01-preview |
| operationalinsights | 2020-03-01-preview |
| peering | 2020-04-01 |
| signalr | 2020-05-01 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| alertsmanagement | 2019-05-05 |
| anomalydetector | v1.0 |
| appconfiguration | 2019-11-01-preview |
| appplatform | 2019-05-01-preview |
| cognitiveservices | 2017-04-18 |
| containerregistry | 2019-12-01-preview |
| datafactory | 2018-06-01 |
| desktopvirtualization | 2019-01-23-preview |
| eventgrid | 2018-01-01 |
| hdinsight | 2018-06-01-preview |
| machinelearningservices | 2020-03-01 |
| media | 2018-07-01 |
| migrate | 2020-01-01 |
| sql | 2015-05-01-preview<br/>2017-03-01-preview<br/>2018-06-01-preview<br/>v3.0 |
| storagecache | 2020-03-01 |
| synapse | 2019-06-01-preview |
| web | 2019-08-01 |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| authoring | luis |
| digitaltwins | 2020-03-01-preview |
| documentdb | 2019-08-01<br/>2020-03-01 |
| eventhub | 2018-01-01-preview |
| insights | v1 |
| kubernetesconfiguration | 2019-11-01-preview |
| kusto | 2020-02-15 |
| netapp | 2019-11-01 |
| operationalinsights | v1 |
| policyinsights | 2018-07-01-preview<br/>2019-10-01 |
| reservations | 2019-07-19-preview |
| security | v1.0<br/>v2.0<br/>v3.0 |
| securityinsight | 2019-01-01-preview |
| siterecovery | 2018-07-10 |
| storage | 2016-12-01<br/>2019-06-01 |
| storagesync | 2019-10-01 |
| timeseriesinsights | 2018-08-15-preview |

## `v42.3.0`

| Package Name | API Version |
| -----------: | :---------: |
| subscription | 2019-10-01-preview |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| resourcegraph | 2018-09-01 |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| eventgrid | 2020-04-01-preview |

## `v42.2.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| mysql | 2020-01-01 |
| network | 2020-04-01 |
| postgresql | 2020-01-01 |
| virtualmachineimagebuilder | 2020-02-01 |

## `v42.1.0`

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| databricks | 2018-04-01 |

## `v42.0.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| desktopvirtualization | 2019-01-23-preview |
| documentdb | 2020-03-01 |
| hybriddata | 2019-06-01 |
| hybridkubernetes | 2020-01-01-preview |
| maintenance | 2020-04-01 |
| search | 2020-03-13 |
| securityinsight | 2019-01-01-preview<br/>v1.0 |
| storagecache | 2020-03-01 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| adhybridhealthservice | 2014-01-01 |
| batch | 2020-03-01 |
| containerregistry | 2019-06-01-preview<br/>2019-12-01-preview |
| costmanagement | 2019-03-01 |
| datafactory | 2018-06-01 |
| documentdb | 2019-08-01-preview<br/>2019-12-12 |
| healthcareapis | 2019-09-16 |
| insights | 2018-03-01<br/>2018-09-01<br/>2018-11-01-preview<br/>2019-03-01<br/>2019-06-01 |
| siterecovery | 2018-07-10 |
| sql | 2014-04-01<br/>2015-05-01-preview<br/>2017-03-01-preview<br/>v3.0 |
| web | 2019-08-01 |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| appplatform | 2019-05-01-preview |
| authoring | luis |
| compute | 2019-12-01 |
| containerservice | 2020-03-01 |
| databricks | 2018-04-01 |
| digitaltwins | 2020-03-01-preview |
| eventhub | 2018-01-01-preview |
| graphrbac | 1.6 |
| logic | 2019-05-01 |
| managedapplications | 2019-07-01 |
| netapp | 2019-05-01<br/>2019-06-01<br/>2019-07-01<br/>2019-08-01<br/>2019-10-01<br/>2019-11-01 |
| policyinsights | 2019-10-01 |
| portal | 2015-08-01-preview<br/>2018-10-01-preview<br/>2019-01-01-preview |
| recoveryservices | 2016-06-01 |
| security | v1.0<br/>v2.0<br/>v3.0 |
| storage | 2019-06-01 |

### Removed Packages

| Package Name | API Version |
| -----------: | :---------: |
| securityinsight | 2017-08-01-preview |

## `v41.3.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| kubernetesconfiguration | 2019-11-01-preview |
| network | 2020-03-01 |
| training | customvision |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| storage | 2019-06-01 |

## `v41.2.0`

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| synapse | 2019-06-01-preview |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| eventgrid | 2020-04-01-preview |

## `v41.1.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| backup | 2020-02-02 |


## `v41.0.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| batch | 2020-03-01<br/>2020-03-01.11.0 |
| computervision | v3.0-preview |
| containerservice | 2020-03-01 |
| keyvault | v7.1-preview<br/>v7.2-preview |
| machinelearningservices | 2020-02-18-preview<br/>2020-03-01 |
| managementgroups | 2020-02-01 |
| migrate | 2020-01-01 |
| redhatopenshift | 2020-04-30 |
| storagesync | 2019-10-01 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| azurestack | 2017-06-01 |
| backup | 2019-06-15 |
| blueprint | 2018-11-01-preview |
| compute | 2019-12-01 |
| documentdb | 2019-08-01<br/>2019-08-01-preview |
| features | 2015-12-01 |
| frontdoor | 2019-04-01<br/>2019-05-01<br/>2019-10-01<br/>2019-11-01 |
| insights | 2015-05-01<br/>v1 |
| operationalinsights | v1 |
| siterecovery | 2018-01-10 |
| sql | 2018-06-01-preview<br/>v3.0 |
| storage | 2019-06-01 |
| web | 2019-08-01 |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| apimanagement | 2018-06-01-preview<br/>2019-01-01<br/>2019-12-01-preview |
| attestation | 2018-09-01-preview |
| cdn | 2019-06-15-preview |
| containerregistry | 2019-06-01-preview<br/>2019-12-01-preview |
| datafactory | 2018-06-01 |
| documentdb | 2019-12-12 |
| eventgrid | 2018-01-01 |
| eventhub | 2015-08-01<br/>2017-04-01 |
| frontdoor | 2020-01-01 |
| insights | 2018-03-01<br/>2018-09-01<br/>2018-11-01-preview<br/>2019-03-01<br/>2019-06-01 |
| iotcentral | 2018-09-01 |
| mariadb | 2018-06-01 |
| msi | 2015-08-31-preview<br/>2018-11-30 |
| mysql | 2017-12-01<br/>2017-12-01-preview |
| netapp | 2019-11-01 |
| network | 2016-09-01<br/>2016-12-01<br/>2019-11-01 |
| notificationhubs | 2014-09-01<br/>2016-03-01<br/>2017-04-01 |
| operationalinsights | 2015-11-01-preview |
| postgresql | 2017-12-01<br/>2017-12-01-preview |
| relay | 2016-07-01<br/>2017-04-01 |
| reservations | 2019-07-19-preview |
| security | v1.0<br/>v2.0<br/>v3.0 |
| servicebus | 2015-08-01<br/>2017-04-01 |
| siterecovery | 2018-07-10 |
| synapse | 2019-06-01-preview |

## `v40.6.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| apimanagement | 2019-12-01 |
| network | 2019-12-01 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| cognitiveservices | 2017-04-18 |

### Updated `./storage` package, which allows users to use azure-sdk-for-go to connect to and use cosmosdb table storage.

## `v40.5.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| kusto | 2020-02-15 |
| subscriptions | 2019-11-01 |
| support | 2020-04-01 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| policyinsights | 2019-10-01 |

## `v40.4.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| advisor | 2020-01-01 |

## `v40.3.0`

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| eventgrid | 2020-04-01-preview |

## `v40.2.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| compute | 2019-12-01 |

## `v40.1.0`

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| managementpartner | 2018-02-01 |

**NOTE:** `services/preview/managementpartner/mgmt/2018-02-01/managementpartner` is a preview package.

## `v40.0.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| logic | 2019-05-01 |
| maps | 2020-02-01-preview |
| peering | 2020-01-01-preview |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| devspaces | 2019-04-01 |
| machinelearningservices | 2020-01-01 |
| msi | 2015-08-31-preview |
| operationalinsights | 2015-03-20<br/>2015-11-01-preview |
| policyinsights | 2019-10-01 |
| security | v1.0<br/>v2.0 |
| sql | 2015-05-01-preview<br/>2017-03-01-preview<br/>2018-06-01-preview |
| subscriptions | 2018-06-01<br/>2019-06-01 |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| backup | 2017-07-01 |
| containerregistry | 2019-05-01-preview<br/>2019-06-01-preview<br/>2019-12-01-preview |
| datafactory | 2018-06-01 |
| datamigration | 2018-07-15-preview |
| documentdb | 2019-12-12 |
| features | 2019-07-01 |
| frontdoor | 2020-01-01 |
| graphrbac | 1.6 |
| hanaonazure | 2017-11-03-preview |
| msi | 2018-11-30 |
| mysql | 2017-12-01 |
| postgresql | 2017-12-01 |
| prediction | customvision |
| resources | 2015-11-01<br/>2016-02-01<br/>2016-07-01<br/>2016-09-01<br/>2017-05-10<br/>2018-02-01<br/>2018-05-01<br/>2019-03-01<br/>2019-05-01 |
| security | v3.0 |
| securityinsight | 2017-08-01-preview |
| sql | v3.0 |
| storage | 2019-06-01 |

### Removed Packages

| Package Name | API Version |
| -----------: | :---------: |
| logic | 2019-05-01 |

**NOTE:** `services/preview/logic/mgmt/2019-05-01/logic` was moved to `services/logic/mgmt/2019-05-01/logic`, since it is a stable package and was placed in `preview` directory by mistake.

## `v39.3.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| containerservice | 2019-10-27-preview |
| synapse | 2019-06-01-preview |

## `v39.2.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| containerservice | 2020-02-01 |

## `v39.1.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| apimanagement | 2019-12-01-preview |
| reservations | 2019-07-19-preview |

## `v39.0.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| appconfiguration | 2019-11-01-preview |
| backup | 2019-06-15 |
| containerregistry | 2019-08-15-preview |
| containerservice | 2020-01-01 |
| digitaltwins | 2020-03-01-preview |
| documentdb | 2019-12-12 |
| frontdoor | 2020-01-01 |
| hybridcompute | 2019-12-12 |
| kusto | 2019-11-09 |
| netapp | 2019-11-01 |
| network | 2019-11-01 |
| support | 2019-05-01-preview |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| blueprint | 2018-11-01-preview |
| compute | 2019-07-01 |
| containerservice | 2019-11-01 |
| eventgrid | 2020-04-01-preview |
| hdinsight | 2015-03-01-preview<br/>2018-06-01-preview |
| healthcareapis | 2019-09-16 |
| keyvault | 2018-02-14 |
| operationalinsights | 2015-11-01-preview |
| policy | 2019-09-01 |
| qnamaker | v4.0 |
| search | 2015-08-19 |
| siterecovery | 2018-07-10 |
| sql | 2017-03-01-preview<br/>2018-06-01-preview<br/>v3.0 |
| web | 2019-08-01 |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| apimanagement | 2019-01-01 |
| containerservice | 2017-07-01<br/>2017-08-31<br/>2017-09-30<br/>2018-03-31 |
| insights | 2018-03-01<br/>2018-09-01<br/>2018-11-01-preview<br/>2019-03-01<br/>2019-06-01 |
| machinelearningservices | 2018-11-19<br/>2019-05-01<br/>2019-06-01<br/>2019-11-01<br/>2020-01-01 |
| managedapplications | 2019-07-01 |
| mariadb | 2018-06-01 |
| netapp | 2019-05-01<br/>2019-06-01<br/>2019-07-01<br/>2019-08-01<br/>2019-10-01 |
| network | 2019-09-01 |
| security | v1.0<br/>v2.0<br/>v3.0 |
| securityinsight | 2017-08-01-preview |
| siterecovery | 2018-01-10 |
| storage | 2019-06-01 |

## `v38.2.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| attestation | 2018-09-01-preview |
| eventgrid | 2020-04-01-preview |
| storagesync | 2019-06-01 |

## `v38.1.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| frontdoor | 2019-10-01<br/>2019-11-01 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| storage | 2019-06-01 |

## `v38.0.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| containerregistry | 2019-12-01-preview |
| databoxedge | 2019-08-01 |
| machinelearningservices | 2020-01-01 |
| netapp | 2019-10-01 |
| windowsesu | 2019-09-16-preview |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| anomalydetector | v1.0 |
| hdinsight | 2015-03-01-preview<br/>2018-06-01-preview |
| insights | 2015-05-01 |
| media | 2018-07-01 |
| personalizer | v1.0 |
| resourcehealth | 2017-07-01 |
| servicebus | 2018-01-01-preview |
| siterecovery | 2018-07-10 |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| authorization | 2015-07-01<br/>2017-10-01-preview |
| blueprint | 2018-11-01-preview |
| compute | 2018-10-01<br/>2019-03-01<br/>2019-07-01 |
| containerservice | 2019-11-01 |
| customerlockbox | 2018-02-28-preview |
| databricks | 2018-04-01 |
| datafactory | 2018-06-01 |
| features | 2019-07-01 |
| managedservices | 2018-06-01<br/>2019-04-01 |
| resources | 2015-11-01<br/>2016-02-01<br/>2016-07-01<br/>2016-09-01<br/>2017-05-10<br/>2018-02-01<br/>2018-05-01<br/>2019-03-01<br/>2019-05-01 |
| security | v1.0<br/>v2.0<br/>v3.0 |
| servicefabric | 2017-07-01-preview<br/>2018-02-01<br/>2019-03-01<br/>2019-03-01-preview |
| sql | v3.0 |
| textanalytics | v2.1 |

## `v37.2.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| containerservice | 2019-11-01 |
| deploymentmanager | 2019-11-01-preview |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| containerservice | 2017-07-01<br/>2017-08-31<br/>2017-09-30<br/>2018-03-31 |
| machinelearningservices | 2019-11-01 |

## `v37.1.0`

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| iotcentral | 2018-09-01 |

## `v37.0.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| azuredata | 2017-03-01-preview |
| backup | 2019-05-13 |
| customerlockbox | 2018-02-28-preview |
| managedapplications | 2019-07-01 |
| servicefabric | 7.0 |
| siterecovery | 2018-07-10 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| azurestack | 2017-06-01 |
| compute | 2018-04-01<br/>2018-06-01<br/>2018-10-01<br/>2019-03-01 |
| containerregistry | 2019-06-01-preview |
| containerservice | 2019-10-01 |
| datashare | 2018-11-01-preview<br/>2019-11-01 |
| hanaonazure | 2017-11-03-preview |
| mysql | 2017-12-01<br/>2017-12-01-preview |
| network | 2019-09-01 |
| policyinsights | 2019-10-01 |
| postgresql | 2017-12-01-preview |
| qnamaker | v4.0 |
| securityinsight | 2017-08-01-preview |
| sql | 2015-05-01-preview<br/>2018-06-01-preview |
| sqlvirtualmachine | 2017-03-01-preview |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| appplatform | 2019-05-01-preview |
| backup | 2017-07-01 |
| cdn | 2019-06-15-preview |
| compute | 2019-07-01 |
| datafactory | 2018-06-01 |
| documentdb | 2019-08-01<br/>2019-08-01-preview |
| network | 2019-08-01 |
| resourcegraph | 2018-09-01 |
| security | v1.0<br/>v2.0<br/>v3.0 |
| sql | 2017-03-01-preview<br/>v3.0 |
| storage | 2019-06-01 |
| virtualmachineimagebuilder | 2019-05-01-preview |

## `v36.2.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| media | 2019-05-01-preview |
| netapp | 2019-08-01 |
| sql | v3.0 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| containerservice | 2019-06-01 |

## `v36.1.0`

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| eventgrid | 2020-01-01-preview |

## `v36.0.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| documentdb | 2019-08-01-preview |
| machinelearningservices | 2019-11-01 |
| managementgroups | 2019-11-01 |
| policy | 2019-09-01 |
| workspaces | 2019-10-01 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| datafactory | 2018-06-01 |
| hanaonazure | 2017-11-03-preview |
| securityinsight | 2017-08-01-preview |
| storage | 2019-06-01 |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| authoring | luis |
| network | 2019-04-01<br/>2019-06-01<br/>2019-07-01<br/>2019-08-01<br/>2019-09-01 |
| serialconsole | 2018-05-01 |
| signalr | 2018-10-01 |
| sql | 2018-06-01-preview |
| timeseriesinsights | 2018-08-15-preview |

### Removed Packages

| Package Name | API Version |
| -----------: | :---------: |
| documentdb | 2019-08-01-preview |

## `v35.1.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| documentdb | 2019-08-01<br/>2019-08-01-preview |

## `v35.0.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| appconfiguration | 2019-10-01 |
| authoring | luis |
| containerservice | 2019-09-30-preview<br/>2019-10-01 |
| costmanagement | 2019-10-01<br/>2019-10-01 |
| datashare | 2019-11-01 |
| hybridcompute | 2019-03-18-preview |
| peering | 2019-09-01-preview |
| policyinsights | 2019-10-01 |
| storagecache | 2019-11-01 |
| training | customvision |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| advisor | 2017-03-31<br/>2017-04-19 |
| apimanagement | 2019-01-01 |
| compute | 2018-10-01 |
| containerservice | 2019-08-01 |
| datafactory | 2017-09-01-preview |
| eventgrid | 2018-01-01 |
| eventhub | 2018-01-01-preview |
| maps | 2017-01-01-preview |
| mysql | 2017-12-01-preview |
| postgresql | 2017-12-01-preview |
| qnamakerruntime | v4.0 |
| sqlvirtualmachine | 2017-03-01-preview |
| web | 2018-02-01 |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| aad | 2017-01-01<br/>2017-06-01 |
| appplatform | 2019-05-01-preview |
| backup | 2016-12-01<br/>2017-07-01 |
| cdn | 2019-04-15 |
| cognitiveservices | 2017-04-18 |
| compute | 2019-03-01<br/>2019-07-01 |
| containerregistry | 2017-10-01<br/>2018-02-01<br/>2018-09-01 |
| datafactory | 2018-06-01 |
| datashare | 2018-11-01-preview |
| hdinsight | 2015-03-01-preview<br/>2018-06-01-preview |
| operationalinsights | 2015-03-20 |
| reservations | 2019-04-01 |
| security | v3.0 |
| servicebus | 2017-04-01 |
| sql | 2014-04-01 |
| storage | 2019-04-01 |
| vmwarecloudsimple | 2019-04-01 |
| web | 2019-08-01 |

### Removed Packages

| Package Name | API Version |
| -----------: | :---------: |
| sql | 2018-06-15-preview |

## `v34.4.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| eventgrid | 2020-01-01-preview |
| sql | 2018-06-15-preview |

## `v34.3.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| network | 2019-09-01 |
| runtime | luis |
| storage | 2019-06-01 |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| datamigration | 2018-07-15-preview |

### Removed Packages

| Package Name | API Version |
| -----------: | :---------: |
| runtime | luis |

## `v34.2.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| appplatform | 2019-05-01-preview |
| web | 2019-08-01 |

## `v34.1.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| features | 2019-07-01 |
| network | 2019-08-01 |
| affinitygroup | management |
| auth | keyvault |
| hostedservice | management |
| location | management |
| mongodb | cosmos-db |
| networksecuritygroup | management |
| osimage | management |
| programmatic | luis |
| sql | management |
| storageservice | management |
| testutils | management |
| virtualmachine | management |
| virtualmachinedisk | management |
| virtualmachineimage | management |
| virtualnetwork | management |
| vmutils | management |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| compute | 2019-03-01<br/>2019-07-01 |

Revert deletion of packages in `classic\management` and `keyvault\auth`

## `v34.0.0`

### New Packages
| Package Name | API Version |
| -----------: | :---------: |
| databox | 2019-09-01 |
| databoxedge | 2019-03-01<br/>2019-07-01 |
| frontdoor | 2019-04-01<br/>2019-05-01 |
| healthcareapis | 2019-09-16 |
| kusto | 2019-09-07 |
| logic | 2019-05-01 |
| maintenance | 2018-06-01-preview |
| storagedatalake | 2019-10-31 |
| subscriptions | 2019-06-01 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| appconfiguration | 2019-02-01-preview |
| datashare | 2018-11-01-preview |
| eventgrid | 2018-01-01 |
| eventhub | 2017-04-01 |
| kusto | 2019-05-15 |
| network | 2018-07-01<br/>2018-08-01<br/>2018-10-01<br/>2018-11-01<br/>2018-12-01<br/>2019-02-01 |
| servicebus | 2017-04-01 |
| sql | 2015-05-01-preview<br/>2017-03-01-preview |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| apimanagement | 2019-01-01 |
| compute | 2016-03-30<br/>2017-03-30<br/>2017-12-01<br/>2018-04-01<br/>2018-06-01<br/>2018-10-01<br/>2019-03-01<br/>2019-07-01 |
| containerregistry | 2019-05-01-preview<br/>2019-06-01-preview |
| containerservice | 2019-06-01<br/>2019-08-01 |
| datafactory | 2018-06-01 |
| datamigration | 2018-03-31-preview<br/>2018-04-19<br/>2018-07-15-preview |
| documentdb | 2015-04-08 |
| frontdoor | 2018-08-01-preview |
| machinelearningservices | 2019-06-01 |
| managednetwork | 2019-06-01-preview |
| network | 2019-04-01 |
| reservations | 2019-04-01 |
| security | v1.0<br/>v2.0<br/>v3.0 |
| securityinsight | 2017-08-01-preview |
| storage | 2019-04-01 |

### Removed Packages

| Package Name | API Version |
| -----------: | :---------: |
| affinitygroup | management |
| anomalyfinder | v2.0 |
| auth | keyvault |
| batch | 2015-12-01.2.2<br/>2016-02-01.3.0<br/>2016-07-01.3.1<br/>2017-01-01.4.0<br/>2017-06-01.5.1<br/>2017-09-01.6.0 |
| computervision | v1.0 |
| devspaces | 2018-06-01-preview<br/>2019-01-01-preview |
| edgegateway | 2019-03-01 |
| frontdoor | preview/2019-04-01<br/>preview/2019-05-01 |
| hostedservice | management |
| insights | v1 |
| location | management |
| mobileengagement | 2014-12-01 |
| mongodb | cosmos-db |
| networksecuritygroup | management |
| osimage | management |
| peering | 2019-03-01-preview |
| portal | 2015-11-01-preview |
| programmatic | luis |
| servicefabric | 2019-03-01 |
| services | 2018-03-01-preview |
| sql | management |
| storageservice | management |
| testutils | management |
| virtualmachine | management |
| virtualmachinedisk | management |
| virtualmachineimage | management |
| virtualnetwork | management |
| vmutils | management |

## `v33.4.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| netapp | 2019-07-01 |

## `v33.3.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| network | 2019-07-01 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| network | 2019-06-01 |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| sql | 2017-10-01-preview<br/>2018-06-01-preview |

## `v33.2.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| sql | 2018-06-01-preview |

## `v33.1.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| storagecache | 2019-08-01-preview |

## `v33.0.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| batch | 2019-08-01<br/>2019-08-01.10.0 |
| computervision | v2.1 |
| containerregistry | 2019-07 |
| containerservice | 2019-08-01 |
| frontdoor | 2019-05-01 |
| machinelearningservices | 2019-06-01 |
| managednetwork | 2019-06-01-preview |
| peering | 2019-08-01-preview |
| policy | 2019-06-01 |
| portal | 2018-10-01-preview |
| servicefabric | 2019-03-01-preview |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| backup | 2016-12-01<br/>2017-07-01 |
| frontdoor | 2019-04-01 |
| logic | 2016-06-01<br/>2018-07-01-preview |
| network | 2018-07-01<br/>2018-08-01<br/>2018-10-01<br/>2018-11-01<br/>2018-12-01 |
| resources | 2015-11-01<br/>2016-02-01<br/>2016-07-01<br/>2016-09-01<br/>2017-05-10<br/>2018-02-01<br/>2018-05-01<br/>2019-03-01<br/>2019-05-01 |
| security | v2.0 |
| sql | 2015-05-01-preview<br/>2017-03-01-preview<br/>2017-10-01-preview |
| storage | 2019-04-01 |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| billing | 2018-11-01-preview |
| compute | 2019-03-01<br/>2019-07-01 |
| datafactory | 2018-06-01 |
| datamigration | 2017-11-15-preview<br/>2018-03-31-preview<br/>2018-04-19<br/>2018-07-15-preview |
| hanaonazure | 2017-11-03-preview |
| healthcareapis | 2018-08-20-preview |
| inkrecognizer | v1.0 |
| insights | 2015-05-01 |
| kusto | 2019-01-21 |
| network | 2019-02-01<br/>2019-04-01<br/>2019-06-01 |
| qnamaker | v4.0 |
| reservations | 2019-04-01 |
| security | v3.0 |
| securityinsight | 2017-08-01-preview |
| servicefabric | 2019-03-01 |

## `v32.6.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| kusto | 2019-05-15 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| datalake | 2018-06-17 |

## `v32.5.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| servicebus | 2018-01-01-preview |

## `v32.4.0`

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| datamigration | 2018-07-15-preview |

## `v32.3.0`

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| subscription | 2018-03-01-preview |

## `v32.2.0`

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| hdinsight | 2015-03-01-preview<br/>2018-06-01-preview |

## `v32.1.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| qnamakerruntime | v4.0 |

### Fixed a bug with the table query continuation token in the ./storage package.

## `v32.0.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| aad | 2017-04-01 |
| compute | 2019-07-01 |
| datashare | 2018-11-01-preview |
| devops | 2019-07-01-preview |
| enterpriseknowledgegraphservice | 2018-12-03 |
| managedservices | 2019-06-01 |
| migrate | 2018-09-01-preview |
| mysql | 2017-12-01-preview |
| network | 2019-06-01 |
| policy | 2019-01-01 |
| portal | 2015-08-01-preview<br/>2019-01-01-preview |
| postgresql | 2017-12-01-preview |
| windowsiot | 2019-06-01 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| alertsmanagement | 2019-05-05 |
| authoring | luis |
| cdn | 2019-04-15 |
| datafactory | 2017-09-01-preview |
| datamigration | 2018-07-15-preview |
| devices | 2019-03-22-preview |
| hanaonazure | 2017-11-03-preview |
| signalr | 2018-10-01 |
| subscriptions | 2018-06-01 |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| compute | 2019-03-01 |
| contentmoderator | v1.0 |
| datafactory | 2018-06-01 |
| documentdb | 2015-04-08 |
| dtl | 2018-09-15 |
| healthcareapis | 2018-08-20-preview |
| insights | 2018-03-01<br/>2018-09-01<br/>2018-11-01-preview<br/>2019-03-01<br/>2019-06-01 |
| machinelearningservices | 2019-05-01 |
| managedservices | 2018-06-01 |
| network | 2019-04-01 |
| reservations | 2019-04-01 |
| security | v1.0<br/>v2.0<br/>v3.0 |
| securityinsight | 2017-08-01-preview |
| storage | 2019-04-01 |

## `v31.2.0`

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| apimanagement | 2019-01-01 |

## `v31.1.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| alertsmanagement | 2019-03-01 |
| authorization | 2018-07-01-preview |
| batch | 2019-04-01 |
| containerregistry | 2019-06-01-preview |
| netapp | 2019-06-01 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| alertsmanagement | 2018-05-05-preview |
| hdinsight | 2015-03-01-preview<br/>2018-06-01-preview |
| sqlvirtualmachine | 2017-03-01-preview |

## `v31.0.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| alertsmanagement | 2019-05-05 |
| appconfiguration | 2019-02-01-preview |
| cdn | 2019-06-15-preview |
| containerservice | 2019-06-01 |
| insights | 2019-06-01 |
| machinelearningservices | 2018-03-01-preview<br/>2018-11-19<br/>2019-05-01 |
| network | 2019-04-01 |
| resources | 2019-05-01 |
| servicefabric | 6.5 |
| softwareplan | 2019-06-01-preview |
| vmwarecloudsimple | 2019-04-01 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| cognitiveservices | 2017-04-18 |
| compute | 2019-03-01 |
| containerregistry | 2019-04-01<br/>2019-05-01 |
| hanaonazure | 2017-11-03-preview |
| hdinsight | 2015-03-01-preview<br/>2018-06-01-preview |
| insights | 2017-05-01-preview<br/>2018-03-01<br/>2018-09-01<br/>2018-11-01-preview<br/>2019-03-01 |
| managementgroups | 2018-03-01-preview |
| media | 2018-07-01 |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| alertsmanagement | 2018-05-05 |
| authorization | 2015-07-01 |
| billing | 2018-11-01-preview |
| blueprint | 2018-11-01-preview |
| computervision | v2.0 |
| datafactory | 2017-09-01-preview<br/>2018-06-01 |
| eventgrid | 2018-01-01 |
| eventhub | 2015-08-01<br/>2018-01-01-preview |
| face | v1.0 |
| netapp | 2019-05-01 |
| network | 2015-06-15<br/>2016-09-01<br/>2016-12-01<br/>2017-03-01<br/>2017-06-01<br/>2017-08-01<br/>2017-09-01<br/>2017-10-01<br/>2017-11-01<br/>2018-01-01<br/>2018-02-01<br/>2018-04-01<br/>2018-06-01<br/>2018-07-01<br/>2018-08-01<br/>2018-10-01<br/>2018-11-01<br/>2018-12-01<br/>2019-02-01 |
| reservations | 2019-04-01 |
| resourcegraph | 2019-04-01 |
| securityinsight | 2017-08-01-preview |
| sql | 2015-05-01-preview<br/>2017-03-01-preview |
| storage | 2019-04-01 |

## `v30.1.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| batch | 2019-06-01.9.0 |
| managedservices | 2019-04-01 |
| personalizer | v1.0 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| authorization | 2017-10-01-preview<br/>2018-01-01-preview<br/>2018-09-01-preview |
| automation | 2015-10-31<br/>2017-05-15-preview<br/>2018-01-15-preview<br/>2018-06-30-preview |
| datafactory | 2018-06-01 |
| datamigration | 2018-07-15-preview |
| documentdb | 2015-04-08 |
| formrecognizer | v1.0 |
| hanaonazure | 2017-11-03-preview |
| hdinsight | 2018-06-01-preview |
| postgresql | 2017-12-01 |
| qnamaker | v4.0 |
| signalr | 2018-10-01 |

## `v30.0.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| blockchain | 2018-06-01-preview |
| containerregistry | 2019-05-01<br/>2019-05-01-preview |
| costmanagement | 2019-03-01 |
| devices | 2019-03-22-preview |
| devspaces | 2019-04-01 |
| dns | 2018-05-01 |
| eventgrid | 2019-06-01 |
| security | v3.0 |
| servicefabric | 2019-03-01<br/>2019-03-01 |

### Updated Packages

| account | 2016-11-01 |
| advisor | 2017-04-19 |
| billing | 2018-11-01-preview |
| cdn | 2019-04-15 |
| cognitiveservices | 2017-04-18 |
| customproviders | 2018-09-01-preview |
| datafactory | 2018-06-01 |
| devices | 2018-12-01-preview |
| eventgrid | 2018-01-01 |
| hanaonazure | 2017-11-03-preview |
| kusto | 2019-01-21 |
| managementpartner | 2018-02-01 |
| mariadb | 2018-06-01 |
| mysql | 2017-12-01 |
| network | 2019-02-01 |
| operationsmanagement | 2015-11-01-preview |
| postgresql | 2017-12-01 |
| servicefabric | 2016-09-01 |
| web | 2018-02-01 |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| analysisservices | 2017-08-01 |
| authoring | luis |
| automation | 2018-01-15-preview<br/>2018-06-30-preview |
| blueprint | 2018-11-01-preview |
| compute | 2017-12-01 |
| computervision | v2.0 |
| contentmoderator | v1.0 |
| documentdb | 2015-04-08 |
| insights | 2015-05-01 |
| netapp | 2019-05-01 |
| resources | 2018-05-01 |
| security | v1.0<br/>v2.0 |
| servicefabric | 2017-07-01-preview<br/>2018-02-01 |
| spellcheck | v1.0 |
| subscriptions | 2016-06-01 |

## `v29.0.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| cdn | 2019-04-15 |
| customproviders | 2018-09-01-preview |
| formrecognizer | v1.0 |
| inkrecognizer | v1.0 |
| portal | 2015-11-01-preview |
| runtime | luis |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| insights | 2018-03-01<br/>2018-09-01<br/>2018-11-01-preview<br/>2019-03-01 |
| locks | 2016-09-01 |
| sql | 2014-04-01<br/>2017-10-01-preview |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| apimanagement | 2019-01-01 |
| containerservice | 2019-04-30 |
| graphrbac | 1.6 |
| sql | 2015-05-01-preview<br/>2017-03-01-preview |
| storage | 2019-04-01 |

## `v28.1.0`

Fixed build issue in legacy storage package affecting some consumers.

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| resourcegraph | 2018-09-01 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| eventhub | 2018-01-01-preview |

## `v28.0.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| containerregistry | 2019-04-01 |
| containerservice | 2019-04-30 |
| hybriddata | 2016-06-01 |
| netapp | 2019-05-01 |
| network | 2019-02-01 |
| resources | 2019-03-01 |
| serialconsole | 2018-05-01 |
| storage | 2019-04-01 |
| subscriptions | 2018-06-01 |
| virtualmachineimagebuilder | 2019-05-01-preview |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| cognitiveservices | 2017-04-18 |
| consumption | 2019-01-01 |
| costmanagement | 2019-01-01 |
| datafactory | 2018-06-01 |
| eventgrid | 2018-01-01 |
| iotcentral | 2018-09-01 |
| qnamaker | v4.0 |
| sql | 2017-10-01-preview |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| apimanagement | 2018-06-01-preview<br/>2019-01-01 |
| billing | 2018-11-01-preview |
| compute | 2019-03-01 |
| cosmos-db | 2015-04-08 |
| documentdb | 2015-04-08 |
| hanaonazure | 2017-11-03-preview |
| insights | 2018-03-01<br/>2018-09-01<br/>2018-11-01-preview<br/>2019-03-01 |
| mysql | 2017-12-01 |
| network | 2017-06-01<br/>2017-08-01<br/>2017-09-01<br/>2017-10-01<br/>2017-11-01<br/>2018-01-01<br/>2018-02-01<br/>2018-04-01<br/>2018-06-01<br/>2018-07-01<br/>2018-08-01<br/>2018-10-01<br/>2018-11-01<br/>2018-12-01 |
| operationalinsights | 2015-11-01-preview |
| policyinsights | 2018-07-01-preview |
| postgresql | 2017-12-01 |
| resources | 2018-05-01 |
| runtime | luis |
| sql | 2015-05-01-preview<br/>2017-03-01-preview |
| storagedatalake | 2018-11-09 |
| subscriptions | 2016-06-01 |
| virtualmachineimagebuilder | 2019-02-01-preview |
| web | 2018-02-01 |

### Removed Packages (duplicates)

| Package Name | API Version |
| -----------: | :---------: |
| automation | 2017-05-15-preview |
| compute | 2017-06-01-preview |
| devices | 2018-12-01-preview |
| fabric | 2016-05-01 |
| infrastructureinsights | 2016-05-01 |
| mariadb | 2018-06-01-preview |
| postgresql | 2017-04-30-preview<br/>2017-12-01-preview |
| reservations | 2018-06-01 |
| storagesync | 2018-10-01 |

## `v27.3.0`

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| anomalydetector | v1.0 |

### Breaking Changes - Preview Only

| Package Name | API Version |
| -----------: | :---------: |
| hanaonazure | 2017-11-03-preview |

## `v27.2.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| apimanagement | 2019-01-01 |
| reservations | 2019-04-01 |

## `v27.1.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| compute | 2019-03-01 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| cdn | 2017-10-12 |
| compute | 2018-10-01 |
| containerservice | 2018-08-01-preview<br/>2018-09-30-preview<br/>2019-02-01 |
| datafactory | 2018-06-01 |
| hdinsight | 2018-06-01-preview |
| postgresql | 2017-12-01 |
| recoveryservices | 2016-06-01 |
| security | v1.0<br/>v2.0 |
| securityinsight | 2017-08-01-preview |
| storage | 2018-02-01<br/>2018-03-01-preview<br/>2018-07-01<br/>2018-11-01 |

### Breaking Changes - Preview Only

| Package Name | API Version |
| -----------: | :---------: |
| hdinsight | 2015-03-01-preview |

## `v27.0.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| billing | 2018-11-01-preview |
| frontdoor | 2019-04-01 |
| healthcareapis | 2018-08-20-preview |
| managedservices | 2018-06-01 |
| peering | 2019-03-01-preview |
| resourcegraph | 2019-04-01 |
| storagesync | 2019-02-01 |
| virtualmachineimagebuilder | 2018-02-01-preview<br/>2019-02-01-preview |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| adhybridhealthservice | 2014-01-01 |
| catalog | 2016-11-01-preview |
| containerregistry | 2017-10-01<br/>2018-02-01<br/>2018-09-01 |
| eventgrid | 2018-01-01 |
| eventhub | 2017-04-01 |
| hanaonazure | 2017-11-03-preview |
| mysql | 2017-12-01 |
| postgresql | 2017-12-01 |
| security | v1.0<br/>v2.0 |
| servicebus | 2017-04-01 |
| signalr | 2018-10-01 |
| sql | 2017-03-01-preview |
| storage | 2018-11-01 |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| apimanagement | 2018-06-01-preview |
| authoring | luis |
| blueprint | 2018-11-01-preview |
| cdn | 2017-10-12 |
| compute | 2018-10-01 |
| computervision | v2.0 |
| datafactory | 2017-09-01-preview<br/>2018-06-01 |
| devices | 2018-12-01-preview |
| edgegateway | 2019-03-01 |
| face | v1.0 |
| graphrbac | 1.6 |
| insights | 2018-03-01 |
| postgresqlapi | postgresql |
| search | 2015-08-19 |
| web | 2018-02-01 |
| webservices | 2017-01-01 |

## `v26.7.0`

| Package Name | API Version |
| -----------: | :---------: |
| training | v3.0 |

## `v26.6.0`

## New Packages

| Package Name | API Version |
| -----------: | :---------: |
| prediction | v3.0 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| marketplaceordering | 2015-06-01 |
| media | 2018-07-01 |

### Breaking Changes - Preview

| Package Name | API Version |
| -----------: | :---------: |
| netapp | 2017-08-15 |

## `v26.5.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| anomalydetector | v1.0 |
| containerservice | 2019-02-01 |
| storage | 2018-11-01 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| network | 2018-12-01 |

## `v26.4.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| insights | 2019-03-01 |

## `v26.3.1`

Remove committed vendored content.

## `v26.3.0`

| Package Name | API Version |
| -----------: | :---------: |
| eventgrid | 2019-02-01-preview |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| marketplaceordering | 2015-06-01 |

## `v26.2.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| mariadb | 2018-06-01 |
| netapp | 2017-08-15 |
| security | v2.0 |
| translatortext | v3.0 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| sql | 2015-05-01-preview<br/>2017-03-01-preview |

### Moved Packages

| Package Name | API Version |
| -----------: | :---------: |
| security | 2017-08-01-preview -> v1.0 |

## `v26.1.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| mixedreality | 2019-02-28 |
| trafficmanager | 2018-04-01 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| mysql | 2017-12-01 |
| postgresql | 2017-12-01 |
| search | 2015-08-19 |

## `v26.0.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| anomalyfinder | v2.0 |
| blueprint | 2018-11-01-preview |
| costmanagement | 2019-01-01 |
| devspaces | 2019-01-01-preview |
| edgegateway | 2019-03-01 |
| network | 2018-12-01 |
| privatedns | 2018-09-01 |

### Updated Packages

| Package Name | API Version |
| commitmentplans | 2016-05-01-preview |
| computervision | v2.0 |
| consumption | 2018-10-01 |
| hanaonazure | 2017-11-03-preview |
| operationalinsights | 2015-03-20 |
| webapi | web |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| containerservice | 2018-03-31<br/>2018-08-01-preview<br/>2018-09-30-preview |
| costmanagement | 2018-08-01-preview |
| datafactory | 2018-06-01 |
| eventgrid | 2018-01-01 |
| face | v1.0 |
| kusto | 2019-01-21 |
| policyinsights | 2018-07-01-preview |
| security | 2017-08-01-preview |
| securityinsight | 2017-08-01-preview |
| sql | 2015-05-01-preview<br/>2017-03-01-preview |
| storagedatalake | 2018-11-09 |
| textanalytics | v2.1 |
| web | 2018-02-01 |

### Removed Packages

| Package Name | API Version |
| -----------: | :---------: |
| blueprint | 2017-11-11-preview |
| edgegateway | 2018-07-01 |

## `v25.1.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| consumption | 2019-01-01 |
| kusto | 2019-01-21 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| authorization | 2015-07-01 |
| backup | 2017-07-01 |
| compute | 2018-10-01 |
| eventgrid | 2018-01-01 |
| eventhub | 2017-04-01 |
| sql | 2017-03-01-preview |

## `v25.0.0`

NOTE: Go 1.8 has been removed from CI due to a transitive dependency no longer supporting it.

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| devices | 2018-12-01-preview |
| insights | 2018-11-01-preview |
| securityinsight | 2017-08-01-preview |
| storagedatalake | 2018-11-09 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| cognitiveservices | 2017-04-18 |
| containerregistry | 2018-09-01 |
| eventgrid | 2018-01-01 |
| hdinsight | 2015-03-01-preview<br/>2018-06-01-preview |
| network | 2018-11-01 |
| runtime | luis |
| sql | 2015-05-01-preview<br/>2017-03-01-preview<br/>2017-10-01-preview |
| web | 2018-02-01 |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| adhybridhealthservice | 2014-01-01 |
| backup | 2016-12-01<br/>2017-07-01 |
| blueprint | 2017-11-11-preview |
| containerservice | 2018-03-31<br/>2018-08-01-preview<br/>2018-09-30-preview |
| datafactory | 2017-09-01-preview<br/>2018-06-01 |
| face | v1.0 |
| hanaonazure | 2017-11-03-preview |
| insights | 2017-05-01-preview |
| logic | 2018-07-01-preview |
| security | 2017-08-01-preview |
| storage | 2015-05-01-preview<br/>2015-06-15<br/>2016-01-01<br/>2016-05-01<br/>2016-12-01<br/>2017-06-01<br/>2017-10-01<br/>2018-02-01<br/>2018-03-01-preview<br/>2018-07-01 |
| virtualmachine | management |

### Removed Packages

NOTE: Some removed packages are preview packages that were incorrectly placed in the stable location.  The copies under `preview` still exist.

| Package Name | API Version |
| -----------: | :---------: |
| apimanagement | 2018-01-01 |
| datafactory | 2017-09-01-preview |
| dns | 2018-03-01-preview |
| insights | 2017-09-01 |
| iothub | 2017-08-21-preview |
| managedapplications | 2016-09-01-preview |
| managementpartner | 2018-02-01 |
| policy | 2015-10-01-preview<br/>2017-06-01-preview |
| servicefabric | 1.0.0<br/>5.6<br/>6.0<br/>6.1 |
| storagedatalake | 2018-06-17<br/>2018-11-09 |

## `v24.1.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| edgegateway | 2018-07-01 |
| network | 2018-11-01 |
| storagesync | 2018-10-01 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| apimanagement | 2018-06-01-preview |
| automation | 2017-05-15-preview<br/>2018-01-15-preview<br/>2018-06-30-preview |
| classic | management |
| containerservice | 2018-03-31<br/>2018-08-01-preview<br/>2018-09-30-preview |
| hanaonazure | 2017-11-03-preview |
| maps | 2018-05-01 |

## `v24.0.0`

### New Packages

| Package Name | API Version |
| -----------: | :---------: |
| batch | 2018-12-01<br/>2018-12-01.8.0 |
| devices | 2018-12-01-preview |
| eventgrid | 2019-01-01 |
| storagedatalake | 2018-06-17<br/>2018-11-09 |

### Updated Packages

| Package Name | API Version |
| -----------: | :---------: |
| apimanagement | 2018-01-01<br/>2018-06-01-preview |
| containerinstance | 2018-10-01 |
| containerregistry | 2017-10-01<br/>2018-02-01<br/>2018-09-01 |
| containerservice | 2018-08-01-preview<br/>2018-09-30-preview |
| costmanagement | 2018-08-01-preview |
| datafactory | 2018-06-01 |
| eventhub | 2018-01-01-preview |
| hanaonazure | 2017-11-03-preview |
| hdinsight | 2018-11-01-preview |
| network | 2018-10-01 |
| resourcehealth | 2015-01-01<br/>2017-07-01 |
| sql | 2017-03-01-preview |
| storagesync | 2018-10-01 |

### BreakingChanges

| Package Name | API Version |
| -----------: | :---------: |
| authoring | luis |
| cognitiveservices | 2017-04-18 |
| computervision | v2.0 |
| datamigration | 2018-04-19<br/>2018-07-15-preview |
| labservices | 2018-10-15 |
| logic | 2018-07-01-preview |
| media | 2018-07-01 |
| siterecovery | 2018-01-10 |
| sqlvirtualmachine | 2017-03-01-preview |
| workloadmonitor | 2018-08-31-preview |
