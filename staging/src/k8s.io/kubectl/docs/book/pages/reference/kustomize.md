{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/JH35X82)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Reference for `kustomization.yaml`
{% endpanel %}

# Kustomization.yaml Reference

#### Terms:

- **Generators**: Provide Resource Config to Kustomize - e.g. `resources`, `bases`, `secretGenerators`.
- **Transformers**: Modify Resource Config by adding, updating or deleting fields - e.g. `namespace`, `commonLabels`, `images`.
- **Meta**: Configure behavior of Generators and Transformers - e.g. generatorOptions, crds, configurations.

## Table of Contents

| Name                                             | Type             | Descriptions                                                                           | Guides                                                                                                                          |
| :----------------------------------------------- | :--------------- | -------------------------------------------------------------------------------------- |  ------------------------------------------------------------------------------------------------------------------------------ |
| [bases](#bases)                                  | Generator        | Add Resource Configs from another `kustomization.yaml`                                 | [Bases and Variants](../app_customization/bases_and_variants.md)                                                                |
| [commonAnnotations](#commonannotations)          | Transformer      | Set annotations on all Resources and Selectors.                                        | [Labels and Annotations](../app_management/labels_and_annotations.md#setting-annotations-for-all-resources)                     |
| [commonLabels](#commonlabels)                    | Transformer      | Set labels on all Resources and Selectors.                                             | [Labels and Annotations](../app_management/labels_and_annotations.md#setting-labels-for-all-resources)                          |
| [configMapGenerator](#configmapgenerator)        | Generator        | Generate ConfigMap Resources.                                                          | [Secrets and ConfigMaps](../app_management/secrets_and_configmaps.md#configmaps-from-files)                                     |
| [configurations](#configurations)                | Meta             | Extend functionality of builtin Transformers to work with additional types (e.g. CRDs).|                                                                                                                                 |
| [generatorOptions](#generatoroptions)            | Meta             | Configure how ConfigMaps and Secrets are generated.                                    |                                                                                                                                 |
| [images](#images)                                | Transformer      | Override image names and tags.                                                         | [Container Images](../app_management/container_images.md)                                                                       |
| [namespace](#namespace)                          | Transformer      | Override namespaces on all Resources.                                                  | [Namespaces and Names](../app_management/namespaces_and_names.md##setting-a-namespace-for-all-resources)                        |
| [namePrefix](#nameprefix)                        | Transformer      | Add a prefix to the names of all Resources and References.                             | [Namespaces and Names](../app_management/namespaces_and_names.md#setting-a-name-prefix-or-suffix-for-all-resources)             |
| [nameSuffix](#namesuffix)                        | Transformer      | Add a suffix to the name of all Resources and References.                              | [Namespaces and Names](../app_management/namespaces_and_names.md#setting-a-name-prefix-or-suffix-for-all-resources)             |
| [patchesJson6902](#patchesjson6902)              | Transformer      | Patch Resource Config using json patch.                                                | [Customizing Resource Fields](../app_customization/customizing_arbitrary_fields.md#customizing-arbitrary-fields-with-jsonpatch) |
| [patchesStrategicMerge](#patchesstrategicmerge)  | Transformer      | Patch Resource Config using an overlay.                                                | [Customizing Resource Fields](../app_customization/customizing_arbitrary_fields.md#customizing-arbitrary-fields-with-overlays)  |
| [resources](#resources)                          | Generator        | Add Raw Resource Configs.                                                              | [Apply](../app_management/apply.md#usage)                                                                                       | 
| [secretGenerator](#secretgenerator)              | Generator        | Generate Secret Resources.                                                             | [Secrets and ConfigMaps](../app_management/secrets_and_configmaps.md#secrets-from-files)                                        |
| [vars](#vars)                                    | Transformer      | Substitute Resource Config field values into Pod Arguments.                            | [Config Reflection](../app_customization/config_reflection.md)                                                                  |

See this [example kustomization.yaml](../examples/kustomize.md)

## Resource Generators

Resource Generators provide Resource Configs to Kustomize from sources such as files, urls, or
`kustomization.yaml` fields.

### bases

{% method %}

`bases` contains a list of paths to **directories or git repositories** containing `kustomization.yaml`s.

`bases` produce Resource Config by running Kustomize against the target.  The provided Resource Config
will then have Transformers from the current `kustomization.yaml` applied.

`bases` are conceptually similar to a base image referenced by `FROM` in a Dockerfile.

| Name          | Type      | Desc                                |
| :------------ | :-------- | :---------------------------------- |
| **base**      | []string  | List of paths must point to directories or git repositories containing `kustomization.yaml`s.   |


{% sample lang="yaml" %}

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
bases:
- path/to/dir/with/kust/
- https://github.com/org/repo/dir/
```

{% endmethod %}

### configMapGenerator

{% method %}

`configMapGenerator` contains a list of ConfigMaps to generate.

By default, generated ConfigMaps will have a hash appended to the name.  The ConfigMap hash is
appended after a `nameSuffix`, if one is specified. Changes to ConfigMap data will cause a ConfigMap
with a new name to be generated, triggering a rolling update to Workloads referencing the ConfigMap.

Resources such as PodTemplates should reference ConfigMaps by the `name` ConfigMapGenerator field,
and Kustomize will update the reference to match the generated name,
as well as `namePrefix`'s and `nameSuffix`'s.

**Note:** Hash suffix generation can be disabled for a subset of ConfigMaps by creating a separate
`kustomization.yaml` and  generating these ConfigMaps there.  This `kustomization.yaml` must set
`generatorOptions.disableNameSuffixHash=true`, and be used as a `base`.  See
[generatorOptions](#generatoroptions) for more details.


| Name                   | Type                      | Desc                                |
| :--------------------- | :------------------------ | :---------------------------------- |
| **configMapGenerator** | []ConfigMapGeneratorArgs  | List of ConfigMaps to generate.     |

##### ConfigMapGeneratorArgs

| Name          | Type      | Desc                                |
| :------------ | :-------- | :---------------------------------- |
| **behavior**  | string    | Merge behavior when the ConfigMap generator is defined in a base.  May be one of `create`, `replace`, `merge`. |
| **env**       | string    | Single file to generate ConfigMap data entries from.  Should be a path to a local *env* file, e.g. `path/to/file.env`, where each line of the file is a `key=value` pair.  *Each line* will appear as an entry in the ConfigMap data field. |
| **files**     | []string  | List of files to generate ConfigMap data entries from. Each item should be a path to a local file, e.g. `path/to/file.config`, and the filename will appear as an entry in the ConfigMap data field with its contents as a value.  |
| **literals**  | []string  | List of literal ConfigMap data entries. Each item should be a key and literal value, e.g. `somekey=somevalue`, and the key/value will appear as an entry in the ConfigMap data field.|
| **name**      | string    | Name for the ConfigMap.  Modified by the `namePrefix` and `nameSuffix` fields. |
| **namespace** | string    | Namespace for the ConfigMap.  Overridden by kustomize-wide `namespace` field.|


{% sample lang="yaml" %}

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
configMapGenerator:
# generate a ConfigMap named my-java-server-props-<some-hash> where each file
# in the list appears as a data entry (keyed by base filename).
- name: my-java-server-props
  files:
  - application.properties
  - more.properties
# generate a ConfigMap named my-java-server-env-vars-<some-hash> where each literal
# in the list appears as a data entry (keyed by literal key).
- name: my-java-server-env-vars
  literals:	
  - JAVA_HOME=/opt/java/jdk
  - JAVA_TOOL_OPTIONS=-agentlib:hprof
# generate a ConfigMap named my-system-env-<some-hash> where each key/value pair in the
# env.txt appears as a data entry (separated by \n).
- name: my-system-env
  env: env.txt
```

{% endmethod %}

### resources

{% method %}

`resources` contains a list of Resource Config file paths to be customized.  Each file may contain multiple
Resource Config definitions separated by `\n---\n`.

| Name          | Type    | Desc                                |
| :------------ | :------ | :---------------------------------- |
| **resources** | []string  | Paths to Resource Config files.   |


{% sample lang="yaml" %}

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
# list of files containing Resource Config to add
resources:
- path/to/resource.yaml
- another/path/to/resource.yaml
```

{% endmethod %}

### secretGenerator

{% method %}

`secretGenerator` contains a list of Secrets to generate.

By default, generated Secrets will have a hash appended to the name.  The Secrets hash is
appended after a `nameSuffix`, if one is specified. Changes to Secrets data will cause a Secrets
with a new name to be generated, triggering a rolling update to Workloads referencing the Secrets.

Resources such as PodTemplates should reference Secrets by the `name` secretsGenerator field,
and Kustomize will update the reference to match the generated name,
as well as `namePrefix`'s and `nameSuffix`'s.

**Note:** Hash suffix generation can be disabled for a subset of Secret by creating a separate
`kustomization.yaml` and  generating these Secret there.  This `kustomization.yaml` must set
`generatorOptions.disableNameSuffixHash=true`, and be used as a `base`.  See
[generatorOptions](#generatoroptions) for more details.



| Name          | Type    | Desc                                |
| :------------ | :------ | :---------------------------------- |
| **secretGenerator** | []SecretGeneratorArgs  | List of Secrets to generate. |

##### SecretGeneratorArgs

| Name          | Type    | Desc                                |
| :------------ | :------ | :---------------------------------- |
| **behavior**  | string  | Merge behavior when the Secret generator is defined in a base.  May be one of `create`, `replace`, `merge`. |
| **env**       | string  | Single file to generate Secret data entries from.  Should be a path to a local *env* file, e.g. `path/to/file.env`, where each line of the file is a `key=value` pair.  *Each line* will appear as an entry in the Secret data field. |
| **files**     | []string  | List of files to generate Secret data entries from. Each item should be a path to a local file, e.g. `path/to/file.config`, and the filename will appear as an entry in the ConfigMap data field with its contents as a value.  |
| **literals**  | []string  | List of literal Secret data entries. Each item should be a key and literal value, e.g. `somekey=somevalue`, and the key/value will appear as an entry in the Secret data field.|
| **name**      | string  | Name for the Secret.  Modified by the `namePrefix` and `nameSuffix` fields. |
| **namespace** | string  | Namespace for the Secret.  Overridden by kustomize-wide `namespace` field.|
| **type**      | string  | Type of Secret. If type is "kubernetes.io/tls", then "literals" or "files" must have exactly two keys: "tls.key" and "tls.crt". |

{% sample lang="yaml" %}

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
secretGenerator:
  # generate a tls Secret
- name: app-tls
  files:
    - secret/tls.cert
    - secret/tls.key
  type: "kubernetes.io/tls"
- name: env_file_secret
  # env is a path to a file to read lines of key=val
  # you can only specify one env file per secret.
  env: env.txt
  type: Opaque
```

{% endmethod %}

## Transformers

Transformers modify Resources by adding, updating or deleting fields.  Transformers work against Generated Resource
Config - e.g.

- `resources`
- `bases`
- `configMapGenerator`
- `secretGenerator`

### commonAnnotations

{% method %}

`commonAnnotations` sets annotations on all Resources.  `commonAnnotations`'s from bases will stack - e.g.
if a `commonAnnotations` was set in a `base`, the new `commonAnnotations` will be added
to or override the base `commonAnnotations`.

| Name          | Type    | Desc                                |
| :------------ | :------ | :---------------------------------- |
| **commonAnnotations** | map[string]string  | Keys/Values for annotations. |


{% sample lang="yaml" %}

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
commonAnnotations:
  annotationKey1: "annotationValue2"
  annotationKey2: "annotationValue2"
```

{% endmethod %}

### commonLabels

{% method %}

This field sets labels on all Resources.  `commonLabels`'s from bases will stack - e.g.
if a `commonLabels` was set in a `base`, the new `commonLabels` will be added
to or override the base `commonLabels`.

`commonLabels` will also be applied both to Label Selector fields and Label fields in PodTemplates.

**Note:**  Because `commonLabels` are applied to Selectors, they cannot be changed for some objects.

| Name          | Type    | Desc                                |
| :------------ | :------ | :---------------------------------- |
| **commonLabels** | map[string]string  | Keys/Values for labels. |


{% sample lang="yaml" %}

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
commonLabels:
  labelKey1: "labelValue1"
  labelKey2: "labelValue2"
```

{% endmethod %}

### images

{% method %}

`images` overrides image names and tags in all `[spec.template.]spec.containers.image` fields matching the
`name`.  This is an alternative to creating patches to change images.

| Name          | Type    | Desc                                |
| :------------ | :------ | :---------------------------------- |
| **images** | []Image  | Images to override. |

##### Image

Definitions:

- *name*: portion of the `image` field value before the `:` - e.g. for `foo:v1` the name would be `foo`.
- *tag*: portion of the `image` field value after the `:` - e.g. for `foo:v1` the name would be `v1`.
- *digest*: alternative to tag for referencing an image. 

| Name          | Type    | Desc                                |
| :------------ | :------ | :---------------------------------- |
| **name** | string  | Match all `image` fields with this value for the *name*|
| **nameName** | string  | Replace the `image` field *name* with this value. |
| **newTag** | string  | Replace the `image` field *tag* with this tag value. |
| **digest** | string  | Replace the `image` field *tag* with this digest value.  Includes the `sha256:` portion of the digest. |

{% sample lang="yaml" %}

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
images:
  - name: postgres
    newName: my-registry/my-postgres
    newTag: v1
  - name: nginx
    newTag: 1.8.0
  - name: my-demo-app
    newName: my-app
  - name: alpine
    digest: sha256:24a0c4b4a4c0eb97a1aabb8e29f18e917d05abfe1b7a7c07857230879ce7d3d3
```

{% endmethod %}

### patchesJson6902

{% method %}

Each entry in this list should resolve to a kubernetes object and a JSON patch that will be applied
to the object. The JSON patch schema is documented at https://tools.ietf.org/html/rfc6902

| Name          | Type    | Desc                                |
| :------------ | :------ | :---------------------------------- |
| **patchesJson6902** | []Json6902  | List of patch definitions. |


##### Json6902

Target field points to a kubernetes object by the object's group, version, kind, name and namespace.
Path field is a relative file path of a JSON patch file.  File contents can be either json or yaml.

| Name          | Type    | Desc                                |
| :------------ | :------ | :---------------------------------- |
| **target** | Target  | Target Resource for the patch. |
| **path** | string  | Path to json patch file.  Maybe json or yaml. |

 Example patch file:

```yaml
 - op: add
   path: /some/new/path
   value: value
 - op: replace
   path: /some/existing/path
   value: new value
``` 

##### Target

| Name          | Type    | Desc                                |
| :------------ | :------ | :---------------------------------- |
| **group** | string  | Group of the Resource to patch. |
| **kind** | string  | Kind of the Resource to patch. |
| **name** | string  | Name of the Resource to patch. |
| **namespace** | string  | Namespace of the Resource to patch. |
| **version** | string  | Version of the Resource to patch. |


{% sample lang="yaml" %}

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
patchesJson6902:
- target:
    version: v1
    kind: Deployment
    name: my-deployment
  path: add_init_container.yaml
- target:
    version: v1
    kind: Service
    name: my-service
  path: add_service_annotation.yaml
```

{% endmethod %}

### patchesStrategicMerge

{% method %}

`patchesStrategicMerge` applies patches to the matching Resource Config (by Group/Version/Kind + Name/Namespace).  Patch
files contain sparse Resource Config definitions - i.e. containing only the Resource Config fields to
add or override.  Strategic merge patches are also called *overlays*.

Small patches that do one thing are best, e.g. modify a memory request/limit.
Small patches are easy to review and easy to compose together.

| Name          | Type    | Desc                                |
| :------------ | :------ | :---------------------------------- |
| **patchesStrategicMerge** | []string  | Paths to files containing sparse Resource Config. |

{% sample lang="yaml" %}

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
patchesStrategicMerge:
- service_port_8888.yaml
- deployment_increase_replicas.yaml
- deployment_increase_memory.yaml
```

{% endmethod %}

### namespace

{% method %}

This field sets the `namespace` of all namespaced Resources.  If the namespace has already been set in the
Resource Config, this will override the namespace.

| Name          | Type    | Desc                                |
| :------------ | :------ | :---------------------------------- |
| **namespace** | String  | Namespace                           |


{% sample lang="yaml" %}

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: "my-app-namespace"
```

{% endmethod %}

### namePrefix

{% method %}

`namePrefix` sets a name prefix on all Resources.  `namePrefix`'s from bases will stack - 
e.g. if a `namePrefix` was set in a `base`, the new `namePrefix` will be pre-prended to the `namePrefix` in the
`base`.

Fields that references another Resource will also have the `namePrefix` applied so that the reference is
updated.

| Name          | Type    | Desc                                |
| :------------ | :------ | :---------------------------------- |
| **namePrefix** | String  | Value to prepend to all Resource names and references. |


{% sample lang="yaml" %}

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namePrefix: "my-app-name-prefix-"
```

{% endmethod %}

### nameSuffix

{% method %}

`nameSuffix` sets a `nameSuffix` on all Resources.  `nameSuffix`'s from bases will stack - 
e.g. if a `nameSuffix` was set in a `base`, the new `nameSuffix` will be appended to the `nameSuffix` in the
`base`.

Fields that references another Resource will also have the `nameSuffix` applied so that the reference is
updated.

| Name          | Type    | Desc                                |
| :------------ | :------ | :---------------------------------- |
| **nameSuffix** | String  | Value to append to all Resource names and references. |


{% sample lang="yaml" %}

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
nameSuffix: "-my-app-name-suffix"
```

{% endmethod %}

### vars

{% method %}

`vars` defines values that can be substituted into Pod container arguments and environment variables.
This is necessary for wiring post-transformed fields into container arguments and environment variables.
e.g. Services names may be transformed by `namePrefix` and containers may need to refer to Service names
at runtime. 

Vars are similar to the Kubernetes [Downward API](https://kubernetes.io/docs/tasks/inject-data-application/environment-variable-expose-pod-information/#use-container-fields-as-values-for-environment-variables)
in that they allow Pods to reference information about the environment in which they are run.

Variables are referenced from container argument using `$(MY_VAR_NAME)`

Example:

```yaml
containers:
- image: myimage
  command: ["start", "--host", "$(MY_SERVICE_NAME)"]
  env:
   - name: SECRET_TOKEN
     value: $(SOME_SECRET_NAME)
```


| Name          | Type    | Desc                                |
| :------------ | :------ | :---------------------------------- |
| **vars** | []Var  | List of variable declarations that may be referenced in container arguments. |


##### Var

| Name          | Type    | Desc                                |
| :------------ | :------ | :---------------------------------- |
| **name** | string  | Name of the variable.  Referenced by `$(NAME)`. |
| **objref** | string  | Reference to the object containing the field to be referenced.  ObjRef should use the unTransformed object name |
| **fieldref** | string  | Reference to the field in the object.  Defaults to `metadata.name` if unspecified.  |

{% sample lang="yaml" %}

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
vars:
  - name: SOME_SECRET_NAME
    objref:
      kind: Secret
      name: my-secret
      apiVersion: v1
  - name: MY_SERVICE_NAME
    objref:
      kind: Service
      name: my-service
      apiVersion: v1
    fieldref:
      fieldpath: metadata.name
  - name: ANOTHER_DEPLOYMENTS_POD_RESTART_POLICY
    objref:
      kind: Deployment
      name: my-deployment
      apiVersion: apps/v1
    fieldref:
      fieldpath: spec.template.spec.restartPolicy
```

{% endmethod %}

## Meta Options

Meta Options control how Kustomize generates and transforms Resource Config.

### configurations

`configurations` is used to configure the built-in Kustomize Transformers to work with CRDs.  The built-in
Kustomize configurations can be found [here](https://github.com/kubernetes-sigs/kustomize/blob/master/api/konfig/builtinpluginconsts)

Examples:

- *images* that should be updated by the `images` Transformer
- *object references* that should be updated by `namePrefix`, `nameSuffix`
- *secret* and *configmap* references that should be updated by `secretGenerator` and `configMapGenerator`

| Name          | Type      | Desc                                |
| :------------ | :-------- | :---------------------------------- |
| **configurations**      | []string  | List of paths to yaml files containing Kustomize meta configuration.   |


> kustomization.yaml

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
configurations:
- mykind_configuration.yaml
```

##### commonAnnotations

{% method %}

Specify `commonAnnotations` in the **configuration file** to configure the Kustomize `commonAnnotations` field
to find additional annotation fields on CRDs.

| Name          | Type      | Desc                                |
| :------------ | :-------- | :---------------------------------- |
| **commonAnnotations**      | []Annotation  | List of paths to annotations fields.   |

| Name          | Type      | Desc                                |
| :------------ | :-------- | :---------------------------------- |
| **create**    | bool      | If true, create the annotation field if it is not present on the Resource Config.   |
| **group**     | string    | API Group of the object to add the annotation to.  If unset, applies to all API Groups. |
| **kind**      | string    | Kind of the object to add the annotation to.  If unset, applies to all Kinds. |
| **path**      | string    | Path to annotation field.   |
| **version**   | string    | API Version of the object to add the annotation to.  If unset, applies to all Versions. |

[Built-in examples](https://github.com/kubernetes-sigs/kustomize/blob/master/api/konfig/builtinpluginconsts/commonannotations.go)

{% sample lang="yaml" %}

> mykind_configuration.yaml file referenced by the configurations field

```yaml
commonAnnotations:
  # set labels at metadata.annotations for all types
- path: metadata/annotations
  # create metadata.annotations if it doesn't exist
  create: true
```

{% endmethod %}

##### commonLabels

{% method %}

Specify `commonLabels` in the **configuration file** to configure the Kustomize `commonLabels` field find
additional labels and selector fields on CRDs.

| Name          | Type      | Desc                                |
| :------------ | :-------- | :---------------------------------- |
| **commonLabels**      | []Label  | List of paths to label fields.   |

| Name          | Type      | Desc                                |
| :------------ | :-------- | :---------------------------------- |
| **create**    | bool      | If true, create the label field if it is not present on the Resource Config.   |
| **group**     | string    | API Group of the object to add the label to.  If unset, applies to all API Groups. |
| **kind**      | string    | Kind of the object to add the label to.  If unset, applies to all Kinds. |
| **path**      | string    | Path to label field.   |
| **version**   | string    | API Version of the object to add the label to.  If unset, applies to all Versions. |

[Built-in examples](https://github.com/kubernetes-sigs/kustomize/blob/master/api/konfig/builtinpluginconsts/commonlabels.go)

{% sample lang="yaml" %}

> mykind_configuration.yaml file referenced by the configurations field

```yaml
commonLabels:
  # set labels at metadata.labels for all types
- path: metadata/labels
  # create metadata.annotations if it doesn't exist
  create: true
  
  # set labels at spec.selector for v1.Service types
- path: spec/selector
  create: true
  version: v1
  kind: Service

  # set labels at spec.selector.matchLabels for Deployment types
- path: spec/selector/matchLabels
  create: true
  kind: Deployment
  
  # set labels at  spec...podAffinity...matchLabels for apps.Deployment types
- path: spec/template/spec/affinity/podAffinity/preferredDuringSchedulingIgnoredDuringExecution/podAffinityTerm/labelSelector/matchLabels
  # do NOT create spec...podAffinity...matchLabels if it doesn't exist on the Deployment Resource Config
  create: false
  group: apps
  kind: Deployment
```

{% endmethod %}


##### images

{% method %}

Specify `images` in the **configuration file** to configure the Kustomize `images` field find additional
image fields on CRDs.

| Name          | Type      | Desc                                |
| :------------ | :-------- | :---------------------------------- |
| **images**      | []Image  |List of paths to image fields.   |

| Name          | Type      | Desc                                |
| :------------ | :-------- | :---------------------------------- |
| **group**     | string    | API Group of the object to add the label to.  If unset, applies to all API Groups. |
| **kind**      | string    | Kind of the object to add the label to.  If unset, applies to all Kinds. |
| **path**      | string    | Path to label field.   |
| **version**   | string    | API Version of the object to add the label to.  If unset, applies to all Versions. |

{% sample lang="yaml" %}

> mykind_configuration.yaml file referenced by the configurations field

```yaml
images:
  # set images at spec.runLatest.container.image for MyKind types
- path: spec/runLatest/container/image
  kind: MyKind
```

{% endmethod %}

##### Name References

{% method %}

Specify `nameReference` in the **configuration file** for CRDs that reference other objects by name - e.g.
Secrets, ConfigMaps, Services, etc.
 
`nameReference` registers for a given type, that **it is referenced by name from another type** - e.g.
Secrets are referenced by Pods.

Doing so will configure Generators and Transformers to update the field value with a new name when
names are modified - e.g. `namePrefix`, `secretGenerator`.

| Name          | Type      | Desc                                |
| :------------ | :-------- | :---------------------------------- |
| **nameReference**      | []Reference  |List of types of objects that are referenced by other objects.   |

| Name           | Type      | Desc                                |
| :------------- | :-------- | :---------------------------------- |
| **group**      | string    | API Group of the object **that is being referenced**.  If unset, applies to all API Groups. |
| **kind**       | string    | Kind of the object to **that is being referenced - e.g. Secret, ConfigMap**. |
| **fieldSpecs** | []FieldSpec | Object types that reference this object type. |
| **version**    | string    | API Version of the object **that is being referenced**.  If unset, applies to all Versions. |

| Name           | Type      | Desc                                |
| :------------- | :-------- | :---------------------------------- |
| **group**     | string    | API Group of the object **that contains a reference**.  If unset, applies to all API Groups. |
| **kind**      | string    | Kind of the object *that contains a reference - e.g. Pod, Deployment**.  If unset, applies to all Kinds. |
| **path**      | string    | Path to the name field that is a reference.   |
| **version**   | string    | API Version of the object *that contains a reference**.  If unset, applies to all Versions. |

[Built-In Examples](https://github.com/kubernetes-sigs/kustomize/blob/master/api/konfig/builtinpluginconsts/namereference.go)

{% sample lang="yaml" %}

> mykind_configuration.yaml file referenced by the configurations field

```yaml
nameReference:
# Configure named references to Secret objects to be updated by Transformers and Generators - e.g. namePrefix, secretGenerator, etc
- kind: Secret
  version: v1
  fieldSpecs:
  # v1.Pods that reference a Secret in spec.volumes.secret.secretName will have it updated
  - path: spec/volumes/secret/secretName
    version: v1
    kind: Pod
  # v1.Pods that reference a Secret in spec.containers.env.valueFrom.secretKeyRef.name will have it updated
  - path: spec/containers/env/valueFrom/secretKeyRef/name
    version: v1
    kind: Pod
```
{% endmethod %}

### generatorOptions

{% method %}

`generatorOptions` modifies behavior of all ConfigMap and Secret generators in the current `kustomization.yaml`.
generatorOptions from `bases` apply **only** to the Secrets and ConfigMaps generated within **the same
`kustomization.yaml`**.

**Note** It is possible to define generatorOptions for a subset of generated Resources by defining a `base` to generate
the Resources and setting the options there.  This supports generating some ConfigMaps with hash-suffixes, and some
without.

| Name          | Type    | Desc                                |
| :------------ | :------ | :---------------------------------- |
| **generatorOptions** | GeneratorOptions  | Options to define how Secrets and ConfigMaps are generated. |

##### GeneratorOptions

| Name          | Type    | Desc                                |
| :------------ | :------ | :---------------------------------- |
| **labels**  | map[string]string  | Labels to add to all Resources generated from this `kustomization.yaml`. |
| **annotations**       | map[string]annotations  | Annotations to add to all Resources generated from this `kustomization.yaml`. |
| **disableNameSuffixHash**     | bool  | If set to true, don't add a hash suffix to any Resources generated from this `kustomization.yaml`.  |

{% sample lang="yaml" %}

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
generatorOptions:
  # labels to add to all generated resources
  labels:
    kustomize.generated.resources: somevalue
  # annotations to add to all generated resources
  annotations:
    kustomize.generated.resource: somevalue
  # disableNameSuffixHash is true disables the default behavior of adding a
  # suffix to the names of generated resources that is a hash of
  # the resource contents.
  disableNameSuffixHash: true
```

{% endmethod %}

