<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# How to Use it?

Install *cifs-utils* on the Kubernetes host. For example, on Fedora based Linux

    # yum -y install cifs-utils

Note, as explained in [Azure File Storage for Linux](https://azure.microsoft.com/en-us/documentation/articles/storage-how-to-use-files-linux/), the Linux hosts and the file share must be in the same Azure region.

Obtain an Microsoft Azure storage account and create a [secret](secret/azure-secret.yaml) that contains the base64 encoded Azure Storage account name and key. In the secret file, base64-encode Azure Storage account name and pair it with name *azurestorageaccountname*, and base64-encode Azure Storage access key and pair it with name *azurestorageaccountkey*.

Then create a Pod using the volume spec based on [azure](azure.yaml).

In the pod, you need to provide the following information:

- *secretName*:  the name of the secret that contains both Azure storage account name and key.
- *shareName*: The share name to be used.
- *readOnly*: Whether the filesystem is used as readOnly.

Create the secret:

```console
    # kubectl create -f examples/volumes/azure_file/secret/azure-secret.yaml
```

You should see the account name and key from `kubectl get secret`

Then create the Pod:

```console
    # kubectl create -f examples/volumes/azure_file/azure.yaml
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/volumes/azure_file/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
