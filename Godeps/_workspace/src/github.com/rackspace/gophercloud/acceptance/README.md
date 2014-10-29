# Gophercloud Acceptance tests

The purpose of these acceptance tests is to validate that SDK features meet
the requirements of a contract - to consumers, other parts of the library, and
to a remote API.

> **Note:** Because every test will be run against a real API endpoint, you
> may incur bandwidth and service charges for all the resource usage. These
> tests *should* remove their remote products automatically. However, there may
> be certain cases where this does not happen; always double-check to make sure
> you have no stragglers left behind.

### Step 1. Set environment variables

A lot of tests rely on environment variables for configuration - so you will need
to set them before running the suite. If you're testing against pure OpenStack APIs,
you can download a file that contains all of these variables for you: just visit
the `project/access_and_security` page in your control panel and click the "Download
OpenStack RC File" button at the top right. For all other providers, you will need
to set them manually.

#### Authentication

|Name|Description|
|---|---|
|`OS_USERNAME`|Your API username|
|`OS_PASSWORD`|Your API password|
|`OS_AUTH_URL`|The identity URL you need to authenticate|
|`OS_TENANT_NAME`|Your API tenant name|
|`OS_TENANT_ID`|Your API tenant ID|
|`RS_USERNAME`|Your Rackspace username|
|`RS_API_KEY`|Your Rackspace API key|

#### General

|Name|Description|
|---|---|
|`OS_REGION_NAME`|The region you want your resources to reside in|
|`RS_REGION`|Rackspace region you want your resource to reside in|

#### Compute

|Name|Description|
|---|---|
|`OS_IMAGE_ID`|The ID of the image your want your server to be based on|
|`OS_FLAVOR_ID`|The ID of the flavor you want your server to be based on|
|`OS_FLAVOR_ID_RESIZE`|The ID of the flavor you want your server to be resized to|
|`RS_IMAGE_ID`|The ID of the image you want servers to be created with|
|`RS_FLAVOR_ID`|The ID of the flavor you want your server to be created with|

### 2. Run the test suite

From the root directory, run:

```
./script/acceptancetest
```
