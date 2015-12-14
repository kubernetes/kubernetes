<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/proposals/openshift_templates.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# OpenShift Template Syntax: Upstreaming the OpenShift template syntax as the k8s template syntax

## Proposed Design

(This design proposes an implementation of the template requirements defined in PR [18215](https://github.com/kubernetes/kubernetes/pull/18215)).

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
components created from a given template.

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

    // Required: Type-value of the parameter (one of string, int, bool, or base64)
    Type ParameterType

    // Optional: The name that will show in UI instead of parameter 'Name'
    DisplayName string

    // Optional: Parameter can have description
    Description string

    // Optional: Value holds the Parameter data. If specified, the generator
    // will be ignored. The value replaces all occurrences of the Parameter
    // $(Name) expression during the Template to Config transformation.
    Value string

    // Optional: Generate specifies the generator to be used to generate
    // random string from an input value specified by From field. The result
    // string is stored into Value field. If empty, no generator is being
    // used, leaving the result Value untouched.
    Generate string

    // Optional: From is an input value for the generator.
    From string

    // Optional: Indicates the parameter must have a value.  Defaults to false.
    Required bool
}
```

As seen above, parameters allow for metadata which can be fed into client implementations to display information about the 
parameter’s purpose and whether a value is required.  Type information is also provided which enables parameter 
substitution into non-string fields.

The value of the parameter can be explicitly defined (this should be considered a default value for the parameter, clients 
which process templates are free to override this value based on user input), or determined via a defined `Generator`.  
Generators are named by a string, and can be optionally supplied an input parameter via the `From` field.

For example:

```
{
      "name": "GITHUB_WEBHOOK_SECRET",
      "description": "A secret string used to configure the GitHub webhook",
      "generate": "expression",
      "from": "[a-zA-Z0-9]{40}"
}
```

This parameter will utilize the expression generator, which takes as an input a regex style string and will generate a 
random value which conforms to the provided regex pattern.

Generators are defined by an interface, so [new generators can easily be added](https://github.com/openshift/origin/tree/master/pkg/template/generator):
```
type Generator interface {
    GenerateValue(expression string) (interface{}, error)
}
```

**Example Template**

Illustration of a template which defines a service and replication controller with parameters to specialized
the name of the top level objects, the number of replicas, and serveral environment variables defined on the 
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
        "name": "$(DATABASE_SERVICE_NAME)",
      },
      "spec": {
        "replicas": "$(REPLICA_COUNT)",
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
                    "value": "${MONGODB_USER}"
                  },
                  {
                    "name": "MONGODB_PASSWORD",
                    "value": "${MONGODB_PASSWORD}"
                  },
                  {
                    "name": "MONGODB_DATABASE",
                    "value": "${MONGODB_DATABASE}"
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
      "required": true,
      "type": "string"
    },
    {
      "name": "MONGODB_USER",
      "description": "Username for MongoDB user that will be used for accessing the database",
      "generate": "expression",
      "from": "user[A-Z0-9]{3}",
      "required": true,
      "type": "string"
    },
    {
      "name": "MONGODB_PASSWORD",
      "description": "Password for the MongoDB user",
      "generate": "expression",
      "from": "[a-zA-Z0-9]{16}",
      "required": true,
      "type" :"string"
    },
    {
      "name": "MONGODB_DATABASE",
      "description": "Database name",
      "value": "sampledb",
      "required": true,
      "type" :"string"      
    },
    {
      "name": "REPLICA_COUNT",
      "description": "Number of mongo replicas to run",
      "value": "1",
      "required": true,
      "type" :"int"      
    }
  ]
}
```

### API Endpoints

* **/processedTemplates** - when a template is POSTed to this endpoint, all parameters in the template are processed, values 
generated as necessary, and substituted into appropriate locations in the object definitions.  Validation is performed to ensure
required parameters have a value (either supplied in the original template or generated by a defined generator).  In addition 
labels defined in the template are applied to the object definitions.  Finally the customized template (still a `Template` object) 
is returned to the caller.

Performing parameter substitution on the server side has the benefit of centralizing the processing so that new clients of 
k8s, such as IDEs, CI systems, Web consoles, etc, do not need to reimplement template processing or embed the k8s binary.
Instead they can invoke the k8s api directly.

* **/templates** - the REST storage resource for storing and retrieving template objects, scoped within a namespace.

Storing templates within k8s has the benefit of enabling template sharing and securing via the same roles/resources
that are used to provide access control to other cluster resoures.  It also enables sophisticated service catalog
flows in which selecting a service from a catalog results in a new instantiation of that service.  (This is not the
only way to implement such a flow, but it does provide a useful level of integration).

### Workflow

#### Template Instantiation

Given a well-formed template, a client will

1. Optionally set an explicit `value` for any parameter values the user wishes to explicitly set (eg non-generated parameters for 
   which the default value is not desired)
2. Optionally set a generator input (`from`) value to override the default input to a generator for a parameter (eg password 
   generation rules) 
3. Submit the new template object to the `/processedTemplates` api endpoint

The api endpoint will then:

1. Validate the template including confirming “required” parameters have an explicit value or an associated generator.
2. For any parameter with no “value” and an associated generator, invoke the generator with the “from” input and set the 
   value of the parameter to that value.
3. Walk each api object in the template.
4. Adding all labels defined in the template’s ObjectLabels field.
5. For each field, check if the value matches a parameter name and if so, set the value of the field to the value of the parameter.
  * Partial substitutions are accepted, such as `SOME_$(PARAM)` which would be transformed into `SOME_XXXX` where `XXXX` is the value
    of the `$(PARAM)` parameter.
6. Return the processed template object.

The client can now either return the processed template to the user in a desired form (eg json or yaml), or directly iterate the 
api objects within the template, invoking the appropriate object creation api endpoint for each element.

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

#### Example Templates

* [Jenkins template](https://github.com/openshift/origin/blob/master/examples/jenkins/jenkins-persistent-template.json)
* [MySQL DB service template](https://github.com/openshift/origin/blob/master/examples/db-templates/mysql-persistent-template.json)

#### Examples of OpenShift Parameter Usage

(mapped to use cases described above)

* [Generate and share passwords](https://github.com/jboss-openshift/application-templates/blob/master/eap/eap64-mongodb-s2i.json#L146-L152)
* [Simple deployment-time customization of “app” configuration via environment values](https://github.com/jboss-openshift/application-templates/blob/master/eap/eap64-mongodb-s2i.json#L108-L126) (eg memory tuning, resource limits, etc)
* [Customization of component names with referential integrity](https://github.com/jboss-openshift/application-templates/blob/master/eap/eap64-mongodb-s2i.json#L199-L207)
* [Customize cross-component references](https://github.com/jboss-openshift/application-templates/blob/master/eap/eap64-mongodb-s2i.json#L78-L83) (eg user provides the name of a secret that already exists in their namespace, to use in a pod as a TLS cert)

## Known Limitations

* Generators must be statically compiled into the template api endpoint, however we feel the provided generators, possibly with some
  additions, can cover 80% of user requirements, avoiding the need for a more complex scheme.
* No template nesting/embedding/references/composing.  
  * Composition creates versioning challenges with respect to upgrading a template from
    one api version to another, as discussed [here](https://github.com/kubernetes/kubernetes/issues/11492#issuecomment-161471745).
  * It also produces more fragile templates as individual pieces can be modified independently, or disappear entirely due to 
    changing access controls on sub-components or removal by the provider.

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

* The `/processedTemplates` api returns the result of the substitution process, which is itself a Template object that can be validated.

*It should be possible to generate forms for parameterized templates, as discussed in #4210 and #6487.*

* Parameter definitions provide metadata that allows for the construction of form-based UIs to gather parameter values from users.

*It shouldn't be inordinately difficult to evolve templates. Thus, strategies such as versioning and encapsulation should be 
encouraged, at least by convention.*

* Templates can be versioned via annotations on the template object.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/templates.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
