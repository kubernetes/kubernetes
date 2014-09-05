Policies should be able to constriain on kind of object.
more about the context for a policy eval call.
predefined labels in the selector or other policy language primitive for object type (spoons moving away from labelselector such as VanillaST (gmail syntax).

namespace for attributes.
namespace for other modules?

### Policy objects
Policy objects are API objects.  They express http://en.wikipedia.org/wiki/Attribute_Based_Access_Control 

Simple Profile:
- one Policy object that allows the single `userAccount` to CRUD objects in the single `project`.

Enterprise Profile:
- Many policy objects in each of many projects.
- Tools and services that wrap policy creation interface to enforce meta policies, do template expansions, report, etc.



Initial Features:
- Policy object is immutable
- Policy objects are statically populated in the K8s API store by reading a config file.  Only a K8s Cluster Admin can do this.
- Just a few policies per `project` which list which users can create objects, which can just view, them, etc.
- Objects are created with reference to these default policies.

Improvements:
- Have API calls to create and delete and modify Policy objects.   These would be in a separate API group from core K8s APIs.  This allows for replacing the K8s authorization service with an alternate implementation, and to centralize policies that might apply to services other than K8s.
- Ability to change policy for an object.
- Ability to create an object with a non-default policy effective immediately at creation time.
- Ability to defer policy object checking to a policy server.
- Authorization tokens to authorize entities without a `userAccount`.


### Policy object format

Policy Object:
```go
type policies map[string]policy

type policy {
    project string, // ref to project of this Policy, to namespace the name.  
    name string, // name within the project name of the Policy
    a PolicyType,
    s Subject,
    v Verb,
    o Object 
    expires string // RFC3339
}
type PolicyType string
const {
   ALLOW PolicyType = "ALLOW",
   // Later DENY, etc.
}

type Subject string // Serialized label selector
type Verb string
const {
   GET Verb = "GET",
   // ... other HTTP methods.
   ANY Verb = "ANY" // any http method
   CREATE Verb = "CREATE" // PUT or POST
   // ...
}
type Object {
  exact string  // any resource with exactly this path
  // OR
  prefix string // any resource with this path prefix (after removing "/api/<version>/")
  // OR
  where string // serialized label selector.
}
```

Ideally, lists of policy objects would have relatively concise and readable YAML forms, such as:
```
{name: bob_can_read_pods, a: ALLOW, s: user.name is bob@example.com, v: GET, prefix: /pods}
{name: admins_can_delete_pods, a: ALLOW, s: user.role is admin, v: DELETE, prefix: /pods}
{name: tmp1234, a: ALLOW, s: user.name is "some.agent", v: POST, prefix: /pods/somepod, expires: 2014-08-13 16:21:42-07:00 }
```

Requests that don't match at least one ALLOW are not allowed.
TODO: define DENY, and other operations and their precedence.

Delegation can be implemented by writing new narrowly tailored policies.
TODO: example of policy to delegate pod creation from a podTemplate (see https://github.com/GoogleCloudPlatform/kubernetes/issues/170).

### Architecture for Authorization
When the APIserver receives a new request, it passes the
the `userAccount`, http method, and http path to an `Authorize() method`.

In a simple implementation, the Authorize() module:
  - runs in the APIserver
  - searches all policy objects for a match.
  - updates its cache  when new Policy is added.

In alternate implementations, it may:
  - have indexes to speed matching.  (Maybe this can share code with Label Queries.)
  - defer to a centralized auth server for the enterprise. 


### Labels
Initially, IIUC, labels are strings and not API objects in their own right. 
Eventually, labels may have policies or namespaces which restrict application of certain labels.
  

### Use cases:

only allow this pod to be started as this role if it was from a trusted build process.
Seems easy enough to add with labels and centralized policy store.
Seems like a PITA to implement and understand with three-legged oauth flow.

flows for adding three users

policy is immutible how to add subject.

inherit vs override

Make users have one of (admin, edit, view)

i
- Allow cluster admin power to create or remove any object.
- Allow project admin power to create or remove any object in the project
- Allow project admin power to add or edit any policies scoped to that project
-Create users with the 3 key roles (simulate RBAC)
- View capabilities by user
- View allowed actions by object
- control who has restart capability on a pod.
8.  Can we define a Policy Template as a global resource?  Can we associate a Policy Template(s) with a Project Template to make initial setup easier?
9.  A policy object should have timestamps.
Typical scenario:
I have a global administrator for my OpenShift deployment that needs to access each Project.
Each project is managed by a team where each member may have particular roles with varying rights.
System view:
We have a Policy object that denotes global admin access added to each Project.
Policy{
    project: "the project", 
    name:"openshift_admin",
    prefix: "/", 
    s:"group=openshift_admin"
    v: "ANY"
    o: {
        prefix: "/"
    }
}
Option 1:
For each unique role, we would need to have a Policy object, but in this case our Policy subject would need to use an OR in the label selector that enumerated each person or group.  This could get awkward to manage.  If a policy object has timestamps, harder to reconcile when a person was added or removed from the project from this model.
Option 2:
For each person or group, we have a dedicated Policy object.  Simpler to enumerate, more resources to compare potentially.  Easier to audit when a person was added or removed from data model.
Issues:
There is a note that "Policy object is immutable, and is statically populated by admin"
It sounds like there is a need for a policy template, but don't you need to edit a policy in order to modify subjects to grow access?

### Blah blah blah

Have Account concept?
Cross-project permissions?

Per-api-group Actions
which are initially defined.
plugins can register their own actions.

Amazon:
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "s3:ListBucket",
    "Resource": "arn:aws:s3:::example_bucket"
  }]
}
http://docs.aws.amazon.com/IAM/latest/UserGuide/PoliciesOverview.html 

User creation after cluster exists:
create user object
create userCredentials and attach to user.
add labels to user or put in groups.


### Blah
What is plan for verbs per resource type?
Seems like Client already defines these.

No-one can do anything that is not allowed.

### AWS CloudFormation


wget https://raw.githubusercontent.com/marceldegraaf/blog-coreos-1/master/stack.yml
ruby -r json -r yaml -e "yaml = YAML.load(File.read('./stack.yml')); print yaml.to_json" > stack.json


Check it out.

Check out how.
### Concepts:
Enforcement point (reads a API call, adds attributes, and calls Decision Point)
Attribute Sources (adds attributes)
Decision Piont (evals policies) 

