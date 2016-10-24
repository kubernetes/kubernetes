# Goals
Enrich Kubernetes's config-map feature to enable the following:
    
    1. Versioned configuration management. 
    2. Consistent configuration management release cycle (following existing roll-forward, roll back process).
    3. Apply new configuration without the need to restart pod's containers.

> The above is must be **supported** without breaking existing features (i.e config maps to environment variables).


# Scenarios

## Creating, Updating & Applying ConfigMaps
The proposed design changes are centered around how ConfigMaps are managed and their relation to pods/pods' instances


### Creating ConfigMaps
The existing apis and schema will remain as-is with the following changes:
1. Configuration maps will become immutable (created, deleted but never changed). Only new versions allowed.
2. Adding new version property (free form value). For example

> i.e. POST /api/v1/namespaces/{namespace}/configmaps/{optional version}

```
apiVersion: v1
 kind: ConfigMap
 metadata:
   Name: example-configmap
   **version: 1.0.0** # or initial release or user specific value
 data:
   # property-like keys
   game-properties-file-name: game.properties
   ui-properties-file-name: ui.properties
   # file-like keys
   game.properties: |
     enemies=aliens
     lives=3
     enemies.cheat=true
     enemies.cheat.level=noGoodRotten
    secret.code.passphrase=UUDDLRLRBABAS
    secret.code.allowed=true
    secret.code.lives=30
  ui.properties: |
    color.good=purple
    color.bad=yellow
    allow.textmode=true
    how.nice.to.look=fairlyNice
```

Kubernetes will recognized and store multiple versions of the same ConfigMap. It will only recognize 
one **current** applied/activated version (currently used by pods) as descried in Applying ConfigMaps section. The string **current** 
can not be used as version name.

> the first version created of a ConfigMap is always considered **current**

### Updating Configuration Maps 

Users will not be allowed to update existing ConfigMaps instead they can create new versions. via

> POST /api/v1/namespaces/{namespace}/configmaps/{name}/{optional version}

The user will not be allowed to reuse an existing version.

### Deleting Configuration Maps

The same api will be used as is. Kubernetes will not deleting the currently applied ConfigMap map version


#### Delete a version

> DELETE /api/v1/namespaces/{namespace}/configmaps/{name}/{version}

#### Delete all version of a ConfigMap

> DELETE /api/v1/namespaces/{namespace}/configmaps/{name}/


### Get Currently Applied Version

Users can get currently applied version of a specific ConfigMap via the following api call.

> GET /api/version/namespaces/{namespace}/configmaps/{name}/current

### Applying/Activating a ConfigMap (making it current)
The process of applying/activating a configmap is used to push the configuration values held by the map into pod's containers. 


#### Administration View 

Administrators can push a new version via the following steps

    1. Optionally create new ConfigMap version. Or use existing for the following step.
    2. Apply/Activate via Performing the following 

> PUT /api/v1/namespaces/{namespace}/configmaps/{name}/current 


with the following payload


```
{
    version: VersionName #where VersionName is existing ConfigMap version to be applied.
}

```

>  The system will perform rolling update process 
(similar to standard pod update process) to replace 
ConfigMap for currently running pods. This **must** include health checks and
roll back in case of unhealthy pods (pods + new version of ConfigMap). This rolling update process
executes one of the actions based on pod configuration as described bellow.


#### Pod's Container View 

#### Default Container Config Update Process (Recycle Pods)

The default approach is to recycle pods' containers (that are using this ConfigMap version) one by one 
following rolling update process. Pods can still use ConfigMaps as they currently today. via
    
    1. Environment variables.
    2. Command line parameters. 
    3. Volume Plugins.

> The following methods does not apply on volume plugins.

#### Via SIGUSRx (1 || 2)

Containers that wish to pick up the new configuration without recycle will receive SIGUSERx 
from kubelet process when the new version of ConfigMap becomes current. Containers can react by calling 
pause containers via web calls to get the new configuration data.

> GET http://{MY_CONFIG_ENDPOINT} #get new configuration


The above will return a flat dictionary (key/value) of all configuration entries irrespective 
of ConfigMap name/version. ConfigMap names and versions can be standard keys in the same dictionary.


**Notes**:
    
    1. MY_CONFIG_ENDPOINT is an environment variable set by kubelet consists of https://pod_ip:{port}/config 
    2. The port in MY_CONFIG_ENDPOINT is fixed to i.e. 9001 and can be overridden in pod configuration file. 
    3. This config end point is listened to by pause container. And is available to all containers in the pod.  
    4. A simple random string is generated for each pod instance. This string is used as the following:
        1. Http authorization header to MY_CONFIG_ENDPOINT
        2. Set by kubelet as an environment variable to containers.
    5. A random https certificate generated by kubelet for each pod instance as the following:
        1. Private + Public keys are mounted into pause container. 
        2. Public key is mounted into every other container in the pod. 
        3. The certificate is used to secure the traffic to pause container.

#### Via Web Hooks

The pause container web server described in the **Via SIGUSRx** can be extend to support web hooks. Containers can choose
to get configuration upgrades via web hooks instead of SIGUSRx. The same previous notes apply.

## Maintaining Backward Compatibility

Pods by default does not subscribe to the new approach of ConfigMaps upgrades. Instead pods will have to declare subscription 
via additional property in pod configuration. This wil allow older pods to use new api as-is for example

```
apiVersion: v1
kind: ReplicationController
metadata:
  name: nginx
spec:
  ConfigRestarts: false # I will use SIGUSRx or Web Hooks // default value is true
  OverrideConfigPort: 9002 # overriding the default value of config endpoint port (passed to paused container)
  replicas: 2
  selector:
    ...
```

> For users upgrading Kubernetes to a version that contain these features all existing ConfigMaps will be base-lined
version to 1.0.0


## Securing Pods' Pause Containers to api server.
The following can be used to secure pods (traffic from pause container) to ConfigMap api.
    
    1. Kubernete generates a new random token for each pod definition.
    2. The random token is loaded in pause container as an environment variable.
    3. Authorization plugin allows the token to perform 
    GET on **/api/v1/namespaces/{namespace}/configmaps/{names of config maps used by pods}/current**


## Further Notes

    1. There is an edge case where pods are being created while configuration is being upgraded. for this
    a simple state machine for **current** can be followed *not current*->*being current*->*current*.
    2. A single pod can reference multiple ConfigMap. For this Any updates (changing current) to configuration
    will require the rolling update process to start. 
    3. The relationship between pods->config maps is does not necessary involve RCs. This will limit the config update 
    mechanism discussed earlier to replicated pods.      
