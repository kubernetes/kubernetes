{% set params = "" -%}

{% if pillar['service_cluster_ip_range'] is defined -%}
{% set params = params + " --portal_net=" + pillar['service_cluster_ip_range'] -%}
{% endif -%}

{% if pillar['opencontrail_public_subnet'] is defined -%}
{% set params = params + " --public_net=" + pillar['opencontrail_public_subnet'] -%}
{% endif -%}

{
    "apiVersion": "v1",
    "kind": "Pod",
    "metadata": {"name":"kube-network-manager"},
    "spec":{
        "hostNetwork": true,
        "containers":[{
            "name": "kube-network-manager",
            "image": "opencontrail/kube-network-manager",
            "command": ["/bin/sh", "-c", "/go/kube-network-manager -- {{params}} 1>>/var/log/contrail/kube-network-manager.stdout 2>>/var/log/contrail/kube-network-manager.err"],
            "volumeMounts": [{
                    "name": "config",
                    "mountPath": "/etc/kubernetes"
            },
                {
                    "name": "logs",
                    "mountPath": "/var/log/contrail",
                    "readOnly": false
        }]
        }],
    "volumes": [{
        "name": "config",
        "hostPath": {"path": "/etc/contrail"}
    },
        {
        "name": "logs",
        "hostPath": {"path": "/var/log/contrail"}
        }]
    }
}
