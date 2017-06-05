<!--[metadata]>
+++
title = "GCS storage driver"
description = "Explains how to use the Google Cloud Storage drivers"
keywords = ["registry, service, driver, images, storage,  gcs, google, cloud"]
+++
<![end-metadata]-->


# Google Cloud Storage driver

An implementation of the `storagedriver.StorageDriver` interface which uses Google Cloud for object storage.

## Parameters


<table>
  <tr>
    <th>Parameter</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>
      <code>bucket</code>
    </td>
    <td>
      yes
    </td>
    <td>
      Storage bucket name.
    </td>
  </tr>
  <tr>
    <td>
      <code>keyfile</code>
    </td>
    <td>
      no
    </td>
    <td>
      A private service account key file in JSON format. Instead of a key file <a href="https://developers.google.com/identity/protocols/application-default-credentials">Google Application Default Credentials</a> can be used.
    </td>
  </tr>
   <tr>
    <td>
      <code>rootdirectory</code>
    </td>
    <td>
      no
    </td>
    <td>
      This is a prefix that will be applied to all Google Cloud Storage keys to allow you to segment data in your bucket if necessary.
  </tr>
  </tr>
   <tr>
    <td>
      <code>chunksize</code>
    </td>
    <td>
      no (default 5242880)
    </td>
    <td>
      This is the chunk size used for uploading large blobs, must be a multiple of 256*1024.
  </tr>

</table>


`bucket`: The name of your Google Cloud Storage bucket where you wish to store objects (needs to already be created prior to driver initialization).

`keyfile`: (optional) A private key file in JSON format, used for [Service Account Authentication](https://cloud.google.com/storage/docs/authentication#service_accounts).

**Note** Instead of a key file you can use [Google Application Default Credentials](https://developers.google.com/identity/protocols/application-default-credentials).

`rootdirectory`: (optional) The root directory tree in which all registry files will be stored. Defaults to the empty string (bucket root).
