<!--[metadata]>
+++
title = "Using certificates for repository client verification"
description = "How to set up and use certificates with a registry to verify access"
keywords = ["Usage, registry, repository, client, root, certificate, docker, apache, ssl, tls, documentation, examples, articles,  tutorials"]
[menu.main]
parent = "mn_docker_hub"
weight = 7
+++
<![end-metadata]-->

# Using certificates for repository client verification

In [Running Docker with HTTPS](/articles/https), you learned that, by default,
Docker runs via a non-networked Unix socket and TLS must be enabled in order
to have the Docker client and the daemon communicate securely over HTTPS.

Now, you will see how to allow the Docker registry (i.e., *a server*) to
verify that the Docker daemon (i.e., *a client*) has the right to access the
images being hosted with *certificate-based client-server authentication*.

We will show you how to install a Certificate Authority (CA) root certificate
for the registry and how to set the client TLS certificate for verification.

## Understanding the configuration

A custom certificate is configured by creating a directory under
`/etc/docker/certs.d` using the same name as the registry's hostname (e.g.,
`localhost`). All `*.crt` files are added to this directory as CA roots.

> **Note:**
> In the absence of any root certificate authorities, Docker
> will use the system default (i.e., host's root CA set).

The presence of one or more `<filename>.key/cert` pairs indicates to Docker
that there are custom certificates required for access to the desired
repository.

> **Note:**
> If there are multiple certificates, each will be tried in alphabetical
> order. If there is an authentication error (e.g., 403, 404, 5xx, etc.), Docker
> will continue to try with the next certificate.

Our example is set up like this:

    /etc/docker/certs.d/        <-- Certificate directory
    └── localhost               <-- Hostname
       ├── client.cert          <-- Client certificate
       ├── client.key           <-- Client key
       └── localhost.crt        <-- Registry certificate

## Creating the client certificates

You will use OpenSSL's `genrsa` and `req` commands to first generate an RSA
key and then use the key to create the certificate.   

    $ openssl genrsa -out client.key 4096
    $ openssl req -new -x509 -text -key client.key -out client.cert

> **Warning:**: 
> Using TLS and managing a CA is an advanced topic.
> You should be familiar with OpenSSL, x509, and TLS before
> attempting to use them in production. 

> **Warning:**
> These TLS commands will only generate a working set of certificates on Linux.
> The version of OpenSSL in Mac OS X is incompatible with the type of
> certificate Docker requires.

## Testing the verification setup

You can test this setup by using Apache to host a Docker registry.
For this purpose, you can copy a registry tree (containing images) inside
the Apache root.

> **Note:**
> You can find such an example [here](
> http://people.gnome.org/~alexl/v1.tar.gz) - which contains the busybox image.

Once you set up the registry, you can use the following Apache configuration
to implement certificate-based protection.

    # This must be in the root context, otherwise it causes a re-negotiation
    # which is not supported by the TLS implementation in go
    SSLVerifyClient optional_no_ca

    <Location /v1>
    Action cert-protected /cgi-bin/cert.cgi
    SetHandler cert-protected

    Header set x-docker-registry-version "0.6.2"
    SetEnvIf Host (.*) custom_host=$1
    Header set X-Docker-Endpoints "%{custom_host}e"
    </Location>

Save the above content as `/etc/httpd/conf.d/registry.conf`, and
continue with creating a `cert.cgi` file under `/var/www/cgi-bin/`.

    #!/bin/bash
    if [ "$HTTPS" != "on" ]; then
        echo "Status: 403 Not using SSL"
        echo "x-docker-registry-version: 0.6.2"
        echo
        exit 0
    fi
    if [ "$SSL_CLIENT_VERIFY" == "NONE" ]; then
        echo "Status: 403 Client certificate invalid"
        echo "x-docker-registry-version: 0.6.2"
        echo
        exit 0
    fi
    echo "Content-length: $(stat --printf='%s' $PATH_TRANSLATED)"
    echo "x-docker-registry-version: 0.6.2"
    echo "X-Docker-Endpoints: $SERVER_NAME"
    echo "X-Docker-Size: 0"
    echo

    cat $PATH_TRANSLATED

This CGI script will ensure that all requests to `/v1` *without* a valid
certificate will be returned with a `403` (i.e., HTTP forbidden) error.
