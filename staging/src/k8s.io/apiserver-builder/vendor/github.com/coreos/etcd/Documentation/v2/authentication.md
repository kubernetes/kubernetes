# Authentication Guide

## Overview

Authentication -- having users and roles in etcd -- was added in etcd 2.1. This guide will help you set up basic authentication in etcd.

etcd before 2.1 was a completely open system; anyone with access to the API could change keys. In order to preserve backward compatibility and upgradability, this feature is off by default.

For a full discussion of the RESTful API, see [the authentication API documentation][auth-api]

## Special Users and Roles

There is one special user, `root`, and there are two special roles, `root` and `guest`.

### User `root`

User `root` must be created before security can be activated. It has the `root` role and allows for the changing of anything inside etcd. The idea behind the `root` user is for recovery purposes -- a password is generated and stored somewhere -- and the root role is granted to the administrator accounts on the system. In the future, for troubleshooting and recovery, we will need to assume some access to the system, and future documentation will assume this root user (though anyone with the role will suffice). 

### Role `root`

Role `root` cannot be modified, but it may be granted to any user. Having access via the root role not only allows global read-write access (as was the case before 2.1) but allows modification of the authentication policy and all administrative things, like modifying the cluster membership.

### Role `guest`

The `guest` role defines the permissions granted to any request that does not provide an authentication. This will be created on security activation (if it doesn't already exist) to have full access to all keys, as was true in etcd 2.0. It may be modified at any time, and cannot be removed.

## Working with users

The `user` subcommand for `etcdctl` handles all things having to do with user accounts.

A listing of users can be found with

```
$ etcdctl user list
```

Creating a user is as easy as

```
$ etcdctl user add myusername
```

And there will be prompt for a new password.

Roles can be granted and revoked for a user with

```
$ etcdctl user grant myusername -roles foo,bar,baz
$ etcdctl user revoke myusername -roles bar,baz
```

We can look at this user with

```
$ etcdctl user get myusername
```

And the password for a user can be changed with

```
$ etcdctl user passwd myusername
```

Which will prompt again for a new password.

To delete an account, there's always
```
$ etcdctl user remove myusername
```


## Working with roles

The `role` subcommand for `etcdctl` handles all things having to do with access controls for particular roles, as were granted to individual users.

A listing of roles can be found with

```
$ etcdctl role list
```

A new role can be created with

```
$ etcdctl role add myrolename
```

A role has no password; we are merely defining a new set of access rights.

Roles are granted access to various parts of the keyspace, a single path at a time.

Reading a path is simple; if the path ends in `*`, that key **and all keys prefixed with it**, are granted to holders of this role. If it does not end in `*`, only that key and that key alone is granted.

Access can be granted as either read, write, or both, as in the following examples:

```
# Give read access to keys under the /foo directory
$ etcdctl role grant myrolename -path '/foo/*' -read

# Give write-only access to the key at /foo/bar
$ etcdctl role grant myrolename -path '/foo/bar' -write

# Give full access to keys under /pub
$ etcdctl role grant myrolename -path '/pub/*' -readwrite
```

Beware that 

```
# Give full access to keys under /pub??
$ etcdctl role grant myrolename -path '/pub*' -readwrite
```

Without the slash may include keys under `/publishing`, for example. To do both, grant `/pub` and `/pub/*`

To see what's granted, we can look at the role at any time:

```
$ etcdctl role get myrolename
```

Revocation of permissions is done the same logical way:

```
$ etcdctl role revoke myrolename -path '/foo/bar' -write
```

As is removing a role entirely

```
$ etcdctl role remove myrolename
```

## Enabling authentication

The minimal steps to enabling auth are as follows. The administrator can set up users and roles before or after enabling authentication, as a matter of preference. 

Make sure the root user is created:

```
$ etcdctl user add root 
New password:
```

And enable authentication

```
$ etcdctl auth enable
```

After this, etcd is running with authentication enabled. To disable it for any reason, use the reciprocal command:

```
$ etcdctl -u root:rootpw auth disable
```

It would also be good to check what guests (unauthenticated users) are allowed to do:
```
$ etcdctl -u root:rootpw role get guest
```

And modify this role appropriately, depending on your policies.

## Using `etcdctl` to authenticate

`etcdctl` supports a similar flag as `curl` for authentication.

```
$ etcdctl -u user:password get foo
```

or if you prefer to be prompted:

```
$ etcdctl -u user get foo
```

Otherwise, all `etcdctl` commands remain the same. Users and roles can still be created and modified, but require authentication by a user with the root role.

[auth-api]: auth_api.md
