<!--[metadata]>
+++
title = "Deploying a registry server"
description = "Explains how to deploy a registry"
keywords = ["registry, on-prem, images, tags, repository, distribution, deployment"]
[menu.main]
parent="smn_registry"
weight=3
+++
<![end-metadata]-->

# Deploying a registry server

You need to [install Docker version 1.6.0 or newer](https://docs.docker.com/installation/).

## Running on localhost

Start your registry:

    docker run -d -p 5000:5000 --restart=always --name registry registry:2

You can now use it with docker.

Get any image from the hub and tag it to point to your registry:

    docker pull ubuntu && docker tag ubuntu localhost:5000/ubuntu

... then push it to your registry:

    docker push localhost:5000/ubuntu

... then pull it back from your registry:

    docker pull localhost:5000/ubuntu

To stop your registry, you would:

    docker stop registry && docker rm -v registry

## Storage

By default, your registry data is persisted as a [docker volume](https://docs.docker.com/userguide/dockervolumes/) on the host filesystem. Properly understanding volumes is essential if you want to stick with a local filesystem storage.

Specifically, you might want to point your volume location to a specific place in order to more easily access your registry data. To do so you can:

    docker run -d -p 5000:5000 --restart=always --name registry \
      -v `pwd`/data:/var/lib/registry \
      registry:2

### Alternatives

You should usually consider using [another storage backend](https://github.com/docker/distribution/blob/master/docs/storagedrivers.md) instead of the local filesystem. Use the [storage configuration options](https://github.com/docker/distribution/blob/master/docs/configuration.md#storage) to configure an alternate storage backend.

Using one of these will allow you to more easily scale your registry, and leverage your storage redundancy and availability features. 

## Running a domain registry

While running on `localhost` has its uses, most people want their registry to be more widely available. To do so, the Docker engine requires you to secure it using TLS, which is conceptually very similar to configuring your web server with SSL.

### Get a certificate

Assuming that you own the domain `myregistrydomain.com`, and that its DNS record points to the host where you are running your registry, you first need to get a certificate from a CA.

Create a `certs` directory:

    mkdir -p certs

Then move and/or rename your crt file to: `certs/domain.crt`, and your key file to: `certs/domain.key`.

Make sure you stopped your registry from the previous steps, then start your registry again with TLS enabled:

    docker run -d -p 5000:5000 --restart=always --name registry \
      -v `pwd`/certs:/certs \
      -e REGISTRY_HTTP_TLS_CERTIFICATE=/certs/domain.crt \
      -e REGISTRY_HTTP_TLS_KEY=/certs/domain.key \
      registry:2

You should now be able to access your registry from another docker host:

    docker pull ubuntu
    docker tag ubuntu myregistrydomain.com:5000/ubuntu
    docker push myregistrydomain.com:5000/ubuntu
    docker pull myregistrydomain.com:5000/ubuntu

#### Gotcha

A certificate issuer may supply you with an *intermediate* certificate. In this case, you must combine your certificate with the intermediate's to form a *certificate bundle*. You can do this using the `cat` command: 

    cat domain.crt intermediate-certificates.pem > certs/domain.crt

### Alternatives

While rarely advisable, you may want to use self-signed certificates instead, or use your registry in an insecure fashion. You will find instructions [here](insecure.md).

## Load Balancing Considerations

One may want to use a load balancer to distribute load, terminate TLS or
provide high availability. While a full load balancing setup is outside the
scope of this document, there are a few considerations that can make the process
smoother.

The most important aspect is that a load balanced cluster of registries must
share the same resources. For the current version of the registry, this means
the following must be the same:

  - Storage Driver
  - HTTP Secret
  - Redis Cache (if configured)

If any of these are different, the registry will have trouble serving requests.
As an example, if you're using the filesystem driver, all registry instances
must have access to the same filesystem root, which means they should be in
the same machine. For other drivers, such as s3 or azure, they should be
accessing the same resource, and will likely share an identical configuration.
The _HTTP Secret_ coordinates uploads, so also must be the same across
instances. Configuring different redis instances will work (at the time
of writing), but will not be optimal if the instances are not shared, causing
more requests to be directed to the backend.

#### Important/Required HTTP-Headers
Getting the headers correct is very important. For all responses to any
request under the "/v2/" url space, the `Docker-Distribution-API-Version`
header should be set to the value "registry/2.0", even for a 4xx response.
This header allows the docker engine to quickly resolve authentication realms
and fallback to version 1 registries, if necessary. Confirming this is setup
correctly can help avoid problems with fallback.

In the same train of thought, you must make sure you are properly sending the
`X-Forwarded-Proto`, `X-Forwarded-For` and `Host` headers to their "client-side"
values. Failure to do so usually makes the registry issue redirects to internal
hostnames or downgrading from https to http.

A properly secured registry should return 401 when the "/v2/" endpoint is hit
without credentials. The response should include a `WWW-Authenticate`
challenge, providing guidance on how to authenticate, such as with basic auth
or a token service. If the load balancer has health checks, it is recommended
to configure it to consider a 401 response as healthy and any other as down.
This will secure your registry by ensuring that configuration problems with
authentication don't accidentally expose an unprotected registry. If you're
using a less sophisticated load balancer, such as Amazon's Elastic Load
Balancer, that doesn't allow one to change the healthy response code, health
checks can be directed at "/", which will always return a `200 OK` response.

## Restricting access

Except for registries running on secure local networks, registries should always implement access restrictions.

### Native basic auth

The simplest way to achieve access restriction is through basic authentication (this is very similar to other web servers' basic authentication mechanism).

> **Warning**: You **cannot** use authentication with an insecure registry. You have to [configure TLS first](#running-a-domain-registry) for this to work.

First create a password file with one entry for the user "testuser", with password "testpassword":

    mkdir auth
    docker run --entrypoint htpasswd registry:2 -Bbn testuser testpassword > auth/htpasswd

Make sure you stopped your registry from the previous step, then start it again:

    docker run -d -p 5000:5000 --restart=always --name registry \
      -v `pwd`/auth:/auth \
      -e "REGISTRY_AUTH=htpasswd" \
      -e "REGISTRY_AUTH_HTPASSWD_REALM=Registry Realm" \
      -e REGISTRY_AUTH_HTPASSWD_PATH=/auth/htpasswd \
      -v `pwd`/certs:/certs \
      -e REGISTRY_HTTP_TLS_CERTIFICATE=/certs/domain.crt \
      -e REGISTRY_HTTP_TLS_KEY=/certs/domain.key \
      registry:2

You should now be able to:

    docker login myregistrydomain.com:5000

And then push and pull images as an authenticated user.

#### Gotcha

Seeing X509 errors is usually a sign you are trying to use self-signed certificates, and failed to [configure your docker daemon properly](insecure.md).

### Alternatives

1. You may want to leverage more advanced basic auth implementations through a proxy design, in front of the registry. You will find examples of such patterns in the [recipes list](recipes.md).

2. Alternatively, the Registry also supports delegated authentication, redirecting users to a specific, trusted token server. That approach requires significantly more investment, and only makes sense if you want to fully configure ACLs and more control over the Registry integration into your global authorization and authentication systems.

You will find [background information here](spec/auth/token.md), and [configuration information here](configuration.md#auth).

Beware that you will have to implement your own authentication service for this to work, or leverage a third-party implementation.

## Managing with Compose

As your registry configuration grows more complex, dealing with it can quickly become tedious.

It's highly recommended to use [Docker Compose](https://docs.docker.com/compose/) to facilitate operating your registry. 

Here is a simple `docker-compose.yml` example that condenses everything explained so far:

```
registry:
  restart: always
  image: registry:2
  ports:
    - 5000:5000
  environment:
    REGISTRY_HTTP_TLS_CERTIFICATE: /certs/domain.crt
    REGISTRY_HTTP_TLS_KEY: /certs/domain.key
    REGISTRY_AUTH: htpasswd
    REGISTRY_AUTH_HTPASSWD_PATH: /auth/htpasswd
    REGISTRY_AUTH_HTPASSWD_REALM: Registry Realm
  volumes:
    - /path/data:/var/lib/registry
    - /path/certs:/certs
    - /path/auth:/auth
```

> **Warning**: replace `/path` by whatever directory that holds your `certs` and `auth` folder from above.

You can then start your registry with a simple

    docker-compose up -d

## Next

You will find more specific and advanced informations in the following sections:

 - [Configuration reference](configuration.md)
 - [Working with notifications](notifications.md)
 - [Advanced "recipes"](recipes.md)
 - [Registry API](spec/api.md)
 - [Storage driver model](storagedrivers.md)
 - [Token authentication](spec/auth/token.md)
