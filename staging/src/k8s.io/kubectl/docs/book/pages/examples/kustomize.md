{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/JH35X82)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Examples for `kustomization.yaml`
{% endpanel %}

# Kustomization.yaml Examples

 {% method %}
This file declares the customization provided by the kustomize program.

Since customization is, by definition, _custom_, there are no default
values that should be copied from this file or that are recommended.

In practice, fields with no value should simply be  omitted from kustomization.yaml
to reduce the content visible in configuration reviews.

Example copied from the [kustomize repo](https://github.com/kubernetes-sigs/kustomize/blob/master/docs/kustomization.yaml)
 
 {% sample lang="yaml" %}
```yaml
# ----------------------------------------------------
# apiVersion and kind of Kustomization
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

# Adds namespace to all resources.
namespace: my-namespace

# Value of this field is prepended to the
# names of all resources, e.g. a deployment named
# "wordpress" becomes "alices-wordpress".
namePrefix: alices-

# Value of this field is appended to the
# names of all resources, e.g. a deployment named
# "wordpress" becomes "wordpress-v2".
# The suffix is appended before content hash
# if resource type is ConfigMap or Secret.
nameSuffix: -v2

# Labels to add to all resources and selectors.
commonLabels:
  someName: someValue
  owner: alice
  app: bingo

# Annotations (non-identifying metadata)
# to add to all resources.  Like labels,
# these are key value pairs.
commonAnnotations:
  oncallPager: 800-555-1212

# Each entry in this list must resolve to an existing
# resource definition in YAML.  These are the resource
# files that kustomize reads, modifies and emits as a
# YAML string, with resources separated by document
# markers ("---").
resources:
- some-service.yaml
- sub-dir/some-deployment.yaml

# Each entry in this list results in the creation of
# one ConfigMap resource (it's a generator of n maps).
# The example below creates two ConfigMaps. One with the
# names and contents of the given files, the other with
# key/value as data.
# Each configMapGenerator item accepts a parameter of
# behavior: [create|replace|merge]. This allows an overlay to modify or
# replace an existing configMap from the parent.
configMapGenerator:
- name: my-java-server-props
  files:
  - application.properties
  - more.properties
- name: my-java-server-env-vars
  literals:	
  - JAVA_HOME=/opt/java/jdk
  - JAVA_TOOL_OPTIONS=-agentlib:hprof

# Each entry in this list results in the creation of
# one Secret resource (it's a generator of n secrets).
secretGenerator:
- name: app-tls
  files:
    - secret/tls.cert
    - secret/tls.key
  type: "kubernetes.io/tls"
- name: app-tls-namespaced
  # you can define a namespace to generate secret in, defaults to: "default"
  namespace: apps
  files:
    - tls.crt=catsecret/tls.cert
    - tls.key=secret/tls.key
  type: "kubernetes.io/tls"
- name: env_file_secret
  # env is a path to a file to read lines of key=val
  # you can only specify one env file per secret.
  env: env.txt
  type: Opaque

# generatorOptions modify behavior of all ConfigMap and Secret generators
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

# Each entry in this list should resolve to a directory
# containing a kustomization file, else the
# customization fails.
#
# The entry could be a relative path pointing to a local directory
# or a url pointing to a directory in a remote repo.
# The url should follow hashicorp/go-getter URL format
# https://github.com/hashicorp/go-getter#url-format
#
# The presence of this field means this file (the file
# you a reading) is an _overlay_ that further
# customizes information coming from these _bases_.
#
# Typical use case: a dev, staging and production
# environment that are mostly identical but differing
# crucial ways (image tags, a few server arguments,
# etc. that differ from the common base).
bases:
- ../../base
- github.com/kubernetes-sigs/kustomize/examples/multibases?ref=v1.0.6
- github.com/Liujingfang1/mysql
- github.com/Liujingfang1/kustomize/examples/helloWorld?ref=test-branch

# Each entry in this list should resolve to
# a partial or complete resource definition file.
#
# The names in these (possibly partial) resource files
# must match names already loaded via the `resources`
# field or via `resources` loaded transitively via the
# `bases` entries.  These entries are used to _patch_
# (modify) the known resources.
#
# Small patches that do one thing are best, e.g. modify
# a memory request/limit, change an env var in a
# ConfigMap, etc.  Small patches are easy to review and
# easy to mix together in overlays.
patchesStrategicMerge:
- service_port_8888.yaml
- deployment_increase_replicas.yaml
- deployment_increase_memory.yaml

# Each entry in this list should resolve to
# a kubernetes object and a JSON patch that will be applied
# to the object.
# The JSON patch is documented at https://tools.ietf.org/html/rfc6902
#
# target field points to a kubernetes object within the same kustomization
# by the object's group, version, kind, name and namespace.
# path field is a relative file path of a JSON patch file.
# The content in this patch file can be either in JSON format as
#
#  [
#    {"op": "add", "path": "/some/new/path", "value": "value"},
#    {"op": "replace", "path": "/some/existing/path", "value": "new value"}
#  ]
#
# or in YAML format as
#
# - op: add
#   path: /some/new/path
#   value: value
# - op:replace
#   path: /some/existing/path
#   value: new value
#
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

# Each entry in this list should be a relative path to
# a file for custom resource definition(CRD) in openAPI definition.
#
# The presence of this field is to allow kustomize be
# aware of CRDs and apply proper
# transformation for any objects in those types.
#
# Typical use case: A CRD object refers to a ConfigMap object.
# In kustomization, the ConfigMap object name may change by adding namePrefix, nameSuffix, or hashing
# The name reference for this ConfigMap object in CRD object need to be
# updated with namePrefix, nameSuffix, or hashing in the same way.
#
# The annotations can be put into openAPI definitions are:
#   "x-kubernetes-annotation": ""
#   "x-kubernetes-label-selector": ""
#   "x-kubernetes-identity": ""
#   "x-kubernetes-object-ref-api-version": "v1",
#   "x-kubernetes-object-ref-kind": "Secret",
#   "x-kubernetes-object-ref-name-key": "name",
crds:
- crds/typeA.json
- crds/typeB.json

# Vars are used to capture text from one resource's field
# and insert that text elsewhere.
#
# For example, suppose someone specifies the name of a k8s Service
# object in a container's command line, and the name of a
# k8s Secret object in a container's environment variable,
# so that the following would work:
# 
#   containers:
#     - image: myimage
#       command: ["start", "--host", "$(MY_SERVICE_NAME)"]
#       env:
#         - name: SECRET_TOKEN
#           value: $(SOME_SECRET_NAME)
# 
#
# To do so, add an entry to `vars:` as follows:
#
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
#
# A var is a tuple of variable name, object reference and field
# reference within that object.  That's where the text is found.
#
# The field reference is optional; it defaults to `metadata.name`,
# a normal default, since kustomize is used to generate or
# modify the names of resources.
#
# At time of writing, only string type fields are supported.
# No ints, bools, arrays etc.  It's not possible to, say,
# extract the name of the image in container number 2 of
# some pod template.
#
# A variable reference, i.e. the string '$(FOO)', can only
# be placed in particular fields of particular objects as
# specified by kustomize's configuration data.
#
# The default config data for vars is at
# https://github.com/kubernetes-sigs/kustomize/blob/master/pkg/transformers/config/defaultconfig/varreference.go
# Long story short, the default targets are all
# container command args and env value fields.
#
# Vars should _not_ be used for inserting names in places
# where kustomize is already handling that job.  E.g.,
# a Deployment may reference a ConfigMap by name, and
# if kustomize changes the name of a ConfigMap, it knows
# to change the name reference in the Deployment.


# Images modify the name, tags and/or digest for images without creating patches.
# E.g. Given this kubernetes Deployment fragment:
# 
#  containers:
#    - name: mypostgresdb
#      image: postgres:8
#    - name: nginxapp
#      image: nginx:1.7.9
#    - name: myapp
#      image: my-demo-app:latest
#    - name: alpine-app
#      image: alpine:3.7
#
# one can change the `image` in the following ways:
# 
# - `postgres:8` to `my-registry/my-postgres:v1`,
# - nginx tag `1.7.9` to `1.8.0`,
# - image name `my-demo-app` to `my-app`,
# - alpine's tag `3.7` to a digest value
#
# all with the following *kustomization*:
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