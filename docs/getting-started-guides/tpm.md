# Enabling TPM admission control in Kubernetes

Prerequistes:
 * A working TPM
 * Installed and configured [Trousers](http://trousers.sourceforge.net) stack
 * [tpmd](https://github.com/coreos/go-tspi) installed and running on port 23179 on all workers

Install Kubernetes and ensure that the apiserver is being started with TPMAdmit in the admission control arguments, eg:

```
--admission-control=TPMAdmit,NamespaceLifecycle,NamespaceExists,LimitRanger,SecurityContextDeny,ServiceAccount,ResourceQuota
```

Enable third party resources in Kubernetes by passing the following argument to apiserver:
```
--runtime-config=extensions/v1beta1=true,extensions/v1beta1/thirdpartyresources=true
```

and create the TPM type by putting the following content in a YAML file:
```yaml
metadata:
  name: "tpm.coreos.com"
apiVersion: "extensions/v1beta1"
kind: "ThirdPartyResource"
description: "A TPM"
versions:
  - name: v1
```

and then running
```
kubectl create -f tpms.yaml
```

Provide a configuration file with the --admission-control-config-file argument. This file should look similar to the following:

```json
{"tpmadmit.pcrconfigdir": "/etc/kubernetes/config/pcr",
 "tpmadmit.recurring": 10,
 "tpmadmit.allowunknown": true}
```

tpmadmit.pcrconfigdir should point to a directory containing json file that resembles the following:

```json
{"0": {"rawvalues": [{"Value": "DE4BB95F83D56B23980D9635D86E626ACA2D1784", "Description": "Firmware v1.0"}]},
 "1": {"rawvalues": [{"Value": "*", "Description": ""}]},
 "2": {"rawvalues": [{"Value": "DE4BB95F83D56B23980D9635D86E626ACA2D1784", "Description": "PXE config"}, {"Value": "0000000000000000000000000000000000000000", "Description": "Null configuration"}]},
 "3": {"asciivalues": [{"prefix": "grub_cmd", "values": [{"Value": "set .*", "Description": "Grub Command"}, {"Value": "echo .*", "Description": "Grub command"}]}]},
 "4": {"binvalues": [{"prefix": "grub_kernel", "values": [{"Value": "2acbccfecf9d0b808861a5956ccf059a9ba7770e", "Description": "Kernel 4.4.0"}]}]}
    }
```

The initial value indicates the PCR. PCRs that are not listed in the configuration are ignored. rawvalues items reflect the raw value of the PCR - the event log is ignored. A wildcard in the rawvalue field will match any PCR value. asciivalues items will verify that the event description for each log entry for the PCR matches the values. If a prefix argument is provided, the string will be broken into two components and only the second component will be validated, allowing separate rules for different types of entry in th same PCR. Strings are matched using standard regexp rules. binvalues items will verify that the event value for each log entry for the PCR matches the SHA values. If a prefix argument is provided, the entry will only be tested if the event description matches the prefix.

tpmadmit.recurring controls whether Kubernetes will revalidate node TPM state on a recurring basis. The argument is the number of seconds between each attempt to validate.

tpmadmit.allowunknown controls whether Kubernetes will permit nodes associated with previously unseen TPMs. If set to false, you will need to enroll TPM entries before the node will be permitted to join the cluster.

If you wish to use a single configuration file rather than multiple configuration fragments, specify tpmadmit.pcrconfig and point it at a single file.

To manually create TPM entries, obtain the EK certificate from the TPM and generate a SHA1 hash of it (TODO: example of how to do this). The write the following JSON:

```json
{
   "metadata": {
        "name": "ekhash"
   },
   "apiVersion": "coreos.com/v1",
   "kind": "Tpm"
}
```

replacing "ekhash" with the hash of the EK certificate. On the controller, run
```
curl -X POST -H "Content-Type: application/json" -d @tpm.json http://localhost:8080/apis/coreos.com/v1/namespaces/default/tpms
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
