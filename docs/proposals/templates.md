# Templates+Parameterization: Repeatedly instantiating user-customized application topologies.

## Motivation

Addresses https://github.com/kubernetes/kubernetes/issues/11492

There are two main motivators for Template functionality in Kubernetes:  Controller Instantiation and Application Definition

### Controller Instantiation

Today the replication controller defines a PodTemplate which allows it to instantiate multiple pods with identical characteristics.
This is useful but limited.  Stateful applications have a need to instantiate multiple instances of a more sophisticated topology
than just a single pod (e.g. they also need Volume definitions).  A Template concept would allow a Controller to stamp out multiple
instances of a given Template definition.  This capability would be immediately useful to the [StatefulSet](https://github.com/kubernetes/kubernetes/pull/18016) proposal.

Similarly the [Service Catalog proposal](https://github.com/kubernetes/kubernetes/pull/17543) could leverage template instantiation as a mechanism for claiming service instances.


### Application Definition

Kubernetes gives developers a platform on which to run images and many configuration objects to control those images, but
constructing a cohesive application made up of images and configuration objects is currently difficult.  Applications
require:

* Information sharing between images (e.g. one image provides a DB service, another consumes it)
* Configuration/tuning settings (memory sizes, queue limits)
* Unique/customizable identifiers (service names, routes)

Application authors know which values should be tunable and what information must be shared, but there is currently no
consistent way for an application author to define that set of information so that application consumers can easily deploy
an application and make appropriate decisions about the tunable parameters the author intended to expose.

Furthermore, even if an application author provides consumers with a set of API object definitions (e.g. a set of yaml files)
it is difficult to build a UI around those objects that would allow the deployer to modify names in one place without
potentially breaking assumed linkages to other pieces.  There is also no prescriptive way to define which configuration
values are appropriate for a deployer to tune or what the parameters control.

## Use Cases

### Use cases for templates in general

* Providing a full baked application experience in a single portable object that can be repeatably deployed in different environments.
  * e.g. Wordpress deployment with separate database pod/replica controller
  * Complex service/replication controller/volume topologies
* Bulk object creation
* Provide a management mechanism for deleting/uninstalling an entire set of components related to a single deployed application
* Providing a library of predefined application definitions that users can select from
* Enabling the creation of user interfaces that can guide an application deployer through the deployment process with descriptive help about the configuration value decisions they are making, and useful default values where appropriate
* Exporting a set of objects in a namespace as a template so the topology can be inspected/visualized or recreated in another environment
* Controllers that need to instantiate multiple instances of identical objects (e.g. StatefulSets).


### Use cases for parameters within templates

* Share passwords between components (parameter value is provided to each component as an environment variable or as a Secret reference, with the Secret value being parameterized or produced by an [initializer](https://github.com/kubernetes/kubernetes/issues/3585))
* Allow for simple deployment-time customization of “app” configuration via environment values or api objects, e.g. memory
  tuning parameters to a MySQL image, Docker image registry prefix for image strings, pod resource requests and limits, default
  scale size.
* Allow simple, declarative defaulting of parameter values and expose them to end users in an approachable way - a parameter
  like “MySQL table space” can be parameterized in images as an env var - the template parameters declare the parameter, give
  it a friendly name, give it a reasonable default, and informs the user what tuning options are available.
* Customization of component names to avoid collisions and ensure matched labeling (e.g. replica selector value and pod label are
  user provided and in sync).
* Customize cross-component references (e.g. user provides the name of a secret that already exists in their namespace, to use in
  a pod as a TLS cert).
* Provide guidance to users for parameters such as default values, descriptions, and whether or not a particular parameter value
  is required or can be left blank.
* Parameterize the replica count of a deployment or [StatefulSet](https://github.com/kubernetes/kubernetes/pull/18016)
* Parameterize part of the labels and selector for a DaemonSet
* Parameterize quota/limit values for a pod
* Parameterize a secret value so a user can provide a custom password or other secret at deployment time


## Design Assumptions

The goal for this proposal is a simple schema which addresses a few basic challenges:

* Allow application authors to expose configuration knobs for application deployers, with suggested defaults and
descriptions of the purpose of each knob
* Allow application deployers to easily customize exposed values like object names while maintaining referential integrity
  between dependent pieces (for example ensuring a pod's labels always match the corresponding selector definition of the service)
* Support maintaining a library of templates within Kubernetes that can be accessed and instantiated by end users
* Allow users to quickly and repeatedly deploy instances of well-defined application patterns produced by the community
* Follow established Kubernetes API patterns by defining new template related APIs which consume+return first class Kubernetes
  API (and therefore json conformant) objects.

We do not wish to invent a new Turing-complete templating language.  There are good options available
(e.g. https://github.com/mustache/mustache) for developers who want a completely flexible and powerful solution for creating
arbitrarily complex templates with parameters, and tooling can be built around such schemes.

This desire for simplicity also intentionally excludes template composability/embedding as a supported use case.

Allowing templates to reference other templates presents versioning+consistency challenges along with making the template
no longer a self-contained portable object.  Scenarios necessitating multiple templates can be handled in one of several
alternate ways:

* Explicitly constructing a new template that merges the existing templates (tooling can easily be constructed to perform this
  operation since the templates are first class api objects).
* Manually instantiating each template and utilizing [service linking](https://github.com/kubernetes/kubernetes/pull/17543) to share
  any necessary configuration data.

This document will also refrain from proposing server APIs or client implementations.  This has been a point of debate, and it makes
more sense to focus on the template/parameter specification/syntax than to worry about the tooling that will process or manage the
template objects.  However since there is a desire to at least be able to support a server side implementation, this proposal
does assume the specification will be k8s API friendly.

## Desired characteristics

* Fully k8s object json-compliant syntax.  This allows server side apis that align with existing k8s apis to be constructed
  which consume templates and existing k8s tooling to work with them.  It also allows for api versioning/migration to be managed by
  the existing k8s codec scheme rather than having to define/introduce a new syntax evolution mechanism.
  * (Even if they are not part of the k8s core, it would still be good if a server side template processing+managing api supplied
    as an ApiGroup consumed the same k8s object schema as the peer k8s apis rather than introducing a new one)
* Self-contained parameter definitions.  This allows a template to be a portable object which includes metadata that describe
  the inputs it expects, making it easy to wrapper a user interface around the parameterization flow.
* Object field primitive types include string, int, boolean, byte[].  The substitution scheme should support all of those types.
  * complex types (struct/map/list) can be defined in terms of the available primitives, so it's preferred to avoid the complexity
    of allowing for full complex-type substitution.
* Parameter metadata.  Parameters should include at a minimum, information describing the purpose of the parameter, whether it is
  required/optional, and a default/suggested value.  Type information could also be required to enable more intelligent client interfaces.
* Template metadata.  Templates should be able to include metadata describing their purpose or links to further documentation and
  versioning information.  Annotations on the Template's metadata field can fulfill this requirement.


## Proposed Implementation

### Overview

We began by looking at the List object which allows a user to easily group a set of objects together for easy creation via a
single CLI invocation.  It also provides a portable format which requires only a single file to represent an application.

From that starting point, we propose a Template API object which can encapsulate the definition of all components of an
application to be created.  The application definition is encapsulated in the form of an array of API objects (identical to
List), plus a parameterization section.  Components reference the parameter by name and the value of the parameter is
substituted during a processing step, prior to submitting each component to the appropriate API endpoint for creation.

The primary capability provided is that parameter values can easily be shared between components, such as a database password
that is provided by the user once, but then attached as an environment variable to both a database pod and a web frontend pod.

In addition, the template can be repeatedly instantiated for a consistent application deployment experience in different
namespaces or Kubernetes clusters.

Lastly, we propose the Template API object include a “Labels” section in which the template author can define a set of labels
to be applied to all objects created from the template.  This will give the template deployer an easy way to manage all the
components created from a given template.  These labels will also be applied to selectors defined by Objects within the template,
allowing a combination of templates and labels to be used to scope resources within a namespace.  That is, a given template
can be instantiated multiple times within the same namespace, as long as a different label value is used each for each
instantiation.  The resulting objects will be independent from a replica/load-balancing perspective.

Generation of parameter values for fields such as Secrets will be delegated to an [admission controller/initializer/finalizer](https://github.com/kubernetes/kubernetes/issues/3585) rather than being solved by the template processor.  Some discussion about a generation
service is occurring [here](https://github.com/kubernetes/kubernetes/issues/12732)

Labels to be assigned to all objects could also be generated in addition to, or instead of, allowing labels to be supplied in the
Template definition.

### API Objects

**Template Object**

```
// Template contains the inputs needed to produce a Config.
type Template struct {
    unversioned.TypeMeta
    kapi.ObjectMeta

    // Optional: Parameters is an array of Parameters used during the
    // Template to Config transformation.
    Parameters []Parameter

    // Required: A list of resources to create
    Objects []runtime.Object

    // Optional: ObjectLabels is a set of labels that are applied to every
    // object during the Template to Config transformation
    // These labels are also be applied to selectors defined by objects in the template
    ObjectLabels map[string]string
}
```

**Parameter Object**

```
// Parameter defines a name/value variable that is to be processed during
// the Template to Config transformation.
type Parameter struct {
    // Required: Parameter name must be set and it can be referenced in Template
    // Items using $(PARAMETER_NAME)
    Name string

    // Optional: The name that will show in UI instead of parameter 'Name'
    DisplayName string

    // Optional: Parameter can have description
    Description string

    // Optional: Value holds the Parameter data.
    // The value replaces all occurrences of the Parameter $(Name) or 
    // $((Name)) expression during the Template to Config transformation.
    Value string

    // Optional: Indicates the parameter must have a non-empty value either provided by the user or provided by a default.  Defaults to false.
    Required bool

    // Optional: Type-value of the parameter (one of string, int, bool, or base64)
    // Used by clients to provide validation of user input and guide users.
    Type ParameterType
}
```

As seen above, parameters allow for metadata which can be fed into client implementations to display information about the
parameter’s purpose and whether a value is required.  In lieu of type information, two reference styles are offered:  `$(PARAM)`
and `$((PARAM))`.  When the single parens option is used, the result of the substitution will remain quoted.  When the double
parens option is used, the result of the substitution will not be quoted.  For example, given a parameter defined with a value
of "BAR", the following behavior will be observed:

```
somefield: "$(FOO)"  ->  somefield: "BAR"
somefield: "$((FOO))"  ->  somefield: BAR
```

// for concatenation, the result value reflects the type of substitution (quoted or unquoted):

```
somefield: "prefix_$(FOO)_suffix"  ->  somefield: "prefix_BAR_suffix"
somefield: "prefix_$((FOO))_suffix"  ->  somefield: prefix_BAR_suffix
```

// if both types of substitution exist, quoting is performed:

```
somefield: "prefix_$((FOO))_$(FOO)_suffix"  ->  somefield: "prefix_BAR_BAR_suffix"
```

This mechanism allows for integer/boolean values to be substituted properly.

The value of the parameter can be explicitly defined in template.  This should be considered a default value for the parameter, clients
which process templates are free to override this value based on user input.


**Example Template**

Illustration of a template which defines a service and replication controller with parameters to specialized
the name of the top level objects, the number of replicas, and several environment variables defined on the
pod template.

```
{
  "kind": "Template",
  "apiVersion": "v1",
  "metadata": {
    "name": "mongodb-ephemeral",
    "annotations": {
      "description": "Provides a MongoDB database service"
    }
  },
  "labels": {
    "template": "mongodb-ephemeral-template"
  },
  "objects": [
    {
      "kind": "Service",
      "apiVersion": "v1",
      "metadata": {
        "name": "$(DATABASE_SERVICE_NAME)"
      },
      "spec": {
        "ports": [
          {
            "name": "mongo",
            "protocol": "TCP",
            "targetPort": 27017
          }
        ],
        "selector": {
          "name": "$(DATABASE_SERVICE_NAME)"
        }
      }
    },
    {
      "kind": "ReplicationController",
      "apiVersion": "v1",
      "metadata": {
        "name": "$(DATABASE_SERVICE_NAME)"
      },
      "spec": {
        "replicas": "$((REPLICA_COUNT))",
        "selector": {
          "name": "$(DATABASE_SERVICE_NAME)"
        },
        "template": {
          "metadata": {
              "creationTimestamp": null,
              "labels": {
                  "name": "$(DATABASE_SERVICE_NAME)"
              }
          },
          "spec": {
            "containers": [
              {
                "name": "mongodb",
                "image": "docker.io/centos/mongodb-26-centos7",
                "ports": [
                  {
                    "containerPort": 27017,
                    "protocol": "TCP"
                  }
                ],
                "env": [
                  {
                    "name": "MONGODB_USER",
                    "value": "$(MONGODB_USER)"
                  },
                  {
                    "name": "MONGODB_PASSWORD",
                    "value": "$(MONGODB_PASSWORD)"
                  },
                  {
                    "name": "MONGODB_DATABASE",
                    "value": "$(MONGODB_DATABASE)"
                  }
                ]
              }
            ]
          }
        }
      }
    }
  ],
  "parameters": [
    {
      "name": "DATABASE_SERVICE_NAME",
      "description": "Database service name",
      "value": "mongodb",
      "required": true
    },
    {
      "name": "MONGODB_USER",
      "description": "Username for MongoDB user that will be used for accessing the database",
      "value": "username",
      "required": true
    },
    {
      "name": "MONGODB_PASSWORD",
      "description": "Password for the MongoDB user",
      "required": true
    },
    {
      "name": "MONGODB_DATABASE",
      "description": "Database name",
      "value": "sampledb",
      "required": true
    },
    {
      "name": "REPLICA_COUNT",
      "description": "Number of mongo replicas to run",
      "value": "1",
      "required": true
    }
  ]
}
```

### API Endpoints

* **/processedtemplates** - when a template is POSTed to this endpoint, all parameters in the template are processed and
substituted into appropriate locations in the object definitions.  Validation is performed to ensure required parameters have
a value supplied.  In addition labels defined in the template are applied to the object definitions.  Finally the customized
template (still a `Template` object) is returned to the caller.  (The possibility of returning a List instead has
also been discussed and will be considered for implementation).

The client is then responsible for iterating the objects returned and POSTing them to the appropriate resource api endpoint to
create each object, if that is the desired end goal for the client.

Performing parameter substitution on the server side has the benefit of centralizing the processing so that new clients of
k8s, such as IDEs, CI systems, Web consoles, etc, do not need to reimplement template processing or embed the k8s binary.
Instead they can invoke the k8s api directly.

* **/templates** - the REST storage resource for storing and retrieving template objects, scoped within a namespace.

Storing templates within k8s has the benefit of enabling template sharing and securing via the same roles/resources
that are used to provide access control to other cluster resources.  It also enables sophisticated service catalog
flows in which selecting a service from a catalog results in a new instantiation of that service.  (This is not the
only way to implement such a flow, but it does provide a useful level of integration).

Creating a new template (POST to the /templates api endpoint) simply stores the template definition, it has no side
effects(no other objects are created).

This resource can also support a subresource "/templates/templatename/processed".  This resource would accept just a
Parameters object and would process the template stored in the cluster as "templatename".  The processed result would be
returned in the same form as `/processedtemplates`

### Workflow

#### Template Instantiation

Given a well-formed template, a client will

1. Optionally set an explicit `value` for any parameter values the user wishes to explicitly set
2. Submit the new template object to the `/processedtemplates` api endpoint

The api endpoint will then:

1. Validate the template including confirming “required” parameters have an explicit value.
2. Walk each api object in the template.
3. Adding all labels defined in the template’s ObjectLabels field.
4. For each field, check if the value matches a parameter name and if so, set the value of the field to the value of the parameter.
  * Partial substitutions are accepted, such as `SOME_$(PARAM)` which would be transformed into `SOME_XXXX` where `XXXX` is the value
    of the `$(PARAM)` parameter.
  * If a given $(VAL) could be resolved to either a parameter or an environment variable/downward api reference, an error will be
    returned.
5. Return the processed template object. (or List, depending on the choice made when this is implemented)

The client can now either return the processed template to the user in a desired form (e.g. json or yaml), or directly iterate the
api objects within the template, invoking the appropriate object creation api endpoint for each element.  (If the api returns
a List, the client would simply iterate the list to create the objects).

The result is a consistently recreatable application configuration, including well-defined labels for grouping objects created by
the template, with end-user customizations as enabled by the template author.

#### Template Authoring

To aid application authors in the creation of new templates, it should be possible to export existing objects from a project
in template form.  A user should be able to export all or a filtered subset of objects from a namespace, wrappered into a
Template API object.  The user will still need to customize the resulting object to enable parameterization and labeling,
though sophisticated export logic could attempt to auto-parameterize well understood api fields.  Such logic is not considered
in this proposal.

#### Tooling

As described above, templates can be instantiated by posting them to a template processing endpoint.  CLI tools should
exist which can input parameter values from the user as part of the template instantiation flow.

More sophisticated UI implementations should also guide the user through which parameters the template expects, the description
of those templates, and the collection of user provided values.

In addition, as described above, existing objects in a namespace can be exported in template form, making it easy to recreate a
set of objects in a new namespace or a new cluster.


## Examples

### Example Templates

These examples reflect the current OpenShift template schema, not the exact schema proposed in this document, however this
proposal, if accepted, provides sufficient capability to support the examples defined here, with the exception of
automatic generation of passwords.

* [Jenkins template](https://github.com/openshift/origin/blob/master/examples/jenkins/jenkins-persistent-template.json)
* [MySQL DB service template](https://github.com/openshift/origin/blob/master/examples/db-templates/mysql-persistent-template.json)

### Examples of OpenShift Parameter Usage

(mapped to use cases described above)

* [Share passwords](https://github.com/jboss-openshift/application-templates/blob/master/eap/eap64-mongodb-s2i.json#L146-L152)
* [Simple deployment-time customization of “app” configuration via environment values](https://github.com/jboss-openshift/application-templates/blob/master/eap/eap64-mongodb-s2i.json#L108-L126) (e.g. memory tuning, resource limits, etc)
* [Customization of component names with referential integrity](https://github.com/jboss-openshift/application-templates/blob/master/eap/eap64-mongodb-s2i.json#L199-L207)
* [Customize cross-component references](https://github.com/jboss-openshift/application-templates/blob/master/eap/eap64-mongodb-s2i.json#L78-L83) (e.g. user provides the name of a secret that already exists in their namespace, to use in a pod as a TLS cert)

## Requirements analysis

There has been some discussion of desired goals for a templating/parameterization solution [here](https://github.com/kubernetes/kubernetes/issues/11492#issuecomment-160853594).  This section will attempt to address each of those points.

*The primary goal is that parameterization should facilitate reuse of declarative configuration templates in different environments in
  a "significant number" of common cases without further expansion, substitution, or other static preprocessing.*

* This solution provides for templates that can be reused as is (assuming parameters are not used or provide sane default values) across
  different environments, they are a self-contained description of a topology.

*Parameterization should not impede the ability to use kubectl commands with concrete resource specifications.*

* The parameterization proposal here does not extend beyond Template objects.  That is both a strength and limitation of this proposal.
  Parameterizable objects must be wrapped into a Template object, rather than existing on their own.

*Parameterization should work with all kubectl commands that accept --filename, and should work on templates comprised of multiple resources.*

* Same as above.

*The parameterization mechanism should not prevent the ability to wrap kubectl with workflow/orchestration tools, such as Deployment manager.*

* Since this proposal uses standard API objects, a DM or Helm flow could still be constructed around a set of templates, just as those flows are
  constructed around other API objects today.

*Any parameterization mechanism we add should not preclude the use of a different parameterization mechanism, it should be possible
to use different mechanisms for different resources, and, ideally, the transformation should be composable with other
substitution/decoration passes.*

* This templating scheme does not preclude layering an additional templating mechanism over top of it.  For example, it would be
  possible to write a Mustache template which, after Mustache processing, resulted in a Template which could then be instantiated
  through the normal template instantiating process.

*Parameterization should not compromise reproducibility. For instance, it should be possible to manage template arguments as well as
templates under version control.*

* Templates are a single file, including default or chosen values for parameters.  They can easily be managed under version control.

*It should be possible to specify template arguments (i.e., parameter values) declaratively, in a way that is "self-describing"
(i.e., naming the parameters and the template to which they correspond). It should be possible to write generic commands to
process templates.*

* Parameter definitions include metadata which describes the purpose of the parameter.  Since parameter definitions are part of the template,
  there is no need to indicate which template they correspond to.

*It should be possible to validate templates and template parameters, both values and the schema.*

* Template objects are subject to standard api validation.

*It should also be possible to validate and view the output of the substitution process.*

* The `/processedtemplates` api returns the result of the substitution process, which is itself a Template object that can be validated.

*It should be possible to generate forms for parameterized templates, as discussed in #4210 and #6487.*

* Parameter definitions provide metadata that allows for the construction of form-based UIs to gather parameter values from users.

*It shouldn't be inordinately difficult to evolve templates. Thus, strategies such as versioning and encapsulation should be
encouraged, at least by convention.*

* Templates can be versioned via annotations on the template object.

## Key discussion points

The preceding document is opinionated about each of these topics, however they have been popular topics of discussion so they are called out explicitly below.

### Where to define parameters

There has been some discussion around where to define parameters that are being injected into a Template

1. In a separate standalone file
2. Within the Template itself

This proposal suggests including the parameter definitions within the Template, which provides a self-contained structure that
can be easily versioned, transported, and instantiated without risk of mismatching content.  In addition, a Template can easily
be validated to confirm that all parameter references are resolveable.

Separating the parameter definitions makes for a more complex process with respect to
* Editing a template (if/when first class editing tools are created)
* Storing/retrieving template objects with a central store

Note that the `/templates/sometemplate/processed` subresource would accept a standalone set of parameters to be applied to `sometemplate`.

### How to define parameters

There has also been debate about how a parameter should be referenced from within a template.  This proposal suggests that
fields to be substituted by a parameter value use the "$(parameter)" syntax which is already used elsewhere within k8s.  The
value of `parameter` should be matched to a parameter with that name, and the value of the matched parameter substituted into
the field value.

Other suggestions include a path/map approach in which a list of field paths (e.g. json path expressions) and corresponding
parameter names are provided.  The substitution process would walk the map, replacing fields with the appropriate
parameter value.  This approach makes templates more fragile from the perspective of editing/refactoring as field paths
may change, thus breaking the map.  There is of course also risk of breaking references with the previous scheme, but
renaming parameters seems less likely than changing field paths.

### Storing templates in k8s

Openshift defines templates as a first class resource so they can be created/retrieved/etc via standard tools.  This allows client tools to list available templates (available in the openshift cluster), allows existing resource security controls to be applied to templates, and generally provides a more integrated feel to templates.  However there is no explicit requirement that for k8s to adopt templates, it must also adopt storing them in the cluster.

### Processing templates (server vs. client)

Openshift handles template processing via a server endpoint which consumes a template object from the client and returns the list of objects
produced by processing the template.  It is also possible to handle the entire template processing flow via the client, but this was deemed
undesirable as it would force each client tool to reimplement template processing (e.g. the standard CLI tool, an eclipse plugin, a plugin for a CI system like Jenkins, etc).  The assumption in this proposal is that server side template processing is the preferred implementation approach for
this reason.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/templates.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
