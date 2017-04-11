<!--[metadata]>
+++
title = "Configuring a registry"
description = "Explains how to configure a registry"
keywords = ["registry, on-prem, images, tags, repository, distribution, configuration"]
[menu.main]
parent="smn_registry"
weight=4
+++
<![end-metadata]-->

# Registry Configuration Reference

The Registry configuration is based on a YAML file, detailed below. While it comes with sane default values out of the box, you are heavily encouraged to review it exhaustively before moving your systems to production.

## Override specific configuration options

In a typical setup where you run your Registry from the official image, you can specify a configuration variable from the environment by passing `-e` arguments to your `docker run` stanza, or from within a Dockerfile using the `ENV` instruction.

To override a configuration option, create an environment variable named
`REGISTRY_variable` where *`variable`* is the name of the configuration option
and the `_` (underscore) represents indention levels. For example, you can
configure the `rootdirectory` of the `filesystem` storage backend:

    storage:
      filesystem:
        rootdirectory: /var/lib/registry

To override this value, set an environment variable like this:

    REGISTRY_STORAGE_FILESYSTEM_ROOTDIRECTORY=/somewhere

This variable overrides the `/var/lib/registry` value to the `/somewhere`
directory.

>**NOTE**: It is highly recommended to create a base configuration file with which environment variables can be used to tweak individual values.  Overriding configuration sections with environment variables is not recommended.

## Overriding the entire configuration file

If the default configuration is not a sound basis for your usage, or if you are having issues overriding keys from the environment, you can specify an alternate YAML configuration file by mounting it as a volume in the container.

Typically, create a new configuration file from scratch, and call it `config.yml`, then:

    docker run -d -p 5000:5000 --restart=always --name registry \
      -v `pwd`/config.yml:/etc/docker/registry/config.yml \
      registry:2

You can (and probably should) use [this as a starting point](https://github.com/docker/distribution/blob/master/cmd/registry/config-example.yml).

## List of configuration options

This section lists all the registry configuration options. Some options in
the list are mutually exclusive. So, make sure to read the detailed reference
information about each option that appears later in this page.

    version: 0.1
    log:
      level: debug
      formatter: text
      fields:
        service: registry
        environment: staging
      hooks:
        - type: mail
          disabled: true
          levels:
            - panic
          options:
            smtp:
              addr: mail.example.com:25
              username: mailuser
              password: password
              insecure: true
            from: sender@example.com
            to:
              - errors@example.com
    loglevel: debug # deprecated: use "log"
    storage:
      filesystem:
        rootdirectory: /var/lib/registry
      azure:
        accountname: accountname
        accountkey: base64encodedaccountkey
        container: containername
      gcs:
        bucket: bucketname
        keyfile: /path/to/keyfile
        rootdirectory: /gcs/object/name/prefix
        chunksize: 5242880
      s3:
        accesskey: awsaccesskey
        secretkey: awssecretkey
        region: us-west-1
        regionendpoint: http://myobjects.local
        bucket: bucketname
        encrypt: true
        keyid: mykeyid
        secure: true
        v4auth: true
        chunksize: 5242880
        rootdirectory: /s3/object/name/prefix
      swift:
        username: username
        password: password
        authurl: https://storage.myprovider.com/auth/v1.0 or https://storage.myprovider.com/v2.0 or https://storage.myprovider.com/v3/auth
        tenant: tenantname
        tenantid: tenantid
        domain: domain name for Openstack Identity v3 API
        domainid: domain id for Openstack Identity v3 API
        insecureskipverify: true
        region: fr
        container: containername
        rootdirectory: /swift/object/name/prefix
      oss:
        accesskeyid: accesskeyid
        accesskeysecret: accesskeysecret
        region: OSS region name
        endpoint: optional endpoints
        internal: optional internal endpoint
        bucket: OSS bucket
        encrypt: optional data encryption setting
        secure: optional ssl setting
        chunksize: optional size valye
        rootdirectory: optional root directory
      inmemory:  # This driver takes no parameters
      delete:
        enabled: false
      redirect:
        disable: false
      cache:
        blobdescriptor: redis
      maintenance:
        uploadpurging:
          enabled: true
          age: 168h
          interval: 24h
          dryrun: false
        readonly:
          enabled: false
    auth:
      silly:
        realm: silly-realm
        service: silly-service
      token:
        realm: token-realm
        service: token-service
        issuer: registry-token-issuer
        rootcertbundle: /root/certs/bundle
      htpasswd:
        realm: basic-realm
        path: /path/to/htpasswd
    middleware:
      registry:
        - name: ARegistryMiddleware
          options:
            foo: bar
      repository:
        - name: ARepositoryMiddleware
          options:
            foo: bar
      storage:
        - name: cloudfront
          options:
            baseurl: https://my.cloudfronted.domain.com/
            privatekey: /path/to/pem
            keypairid: cloudfrontkeypairid
            duration: 3000s
    reporting:
      bugsnag:
        apikey: bugsnagapikey
        releasestage: bugsnagreleasestage
        endpoint: bugsnagendpoint
      newrelic:
        licensekey: newreliclicensekey
        name: newrelicname
        verbose: true
    http:
      addr: localhost:5000
      prefix: /my/nested/registry/
      host: https://myregistryaddress.org:5000
      secret: asecretforlocaldevelopment
      relativeurls: false
      tls:
        certificate: /path/to/x509/public
        key: /path/to/x509/private
        clientcas:
          - /path/to/ca.pem
          - /path/to/another/ca.pem
      debug:
        addr: localhost:5001
      headers:
        X-Content-Type-Options: [nosniff]
    notifications:
      endpoints:
        - name: alistener
          disabled: false
          url: https://my.listener.com/event
          headers: <http.Header>
          timeout: 500
          threshold: 5
          backoff: 1000
    redis:
      addr: localhost:6379
      password: asecret
      db: 0
      dialtimeout: 10ms
      readtimeout: 10ms
      writetimeout: 10ms
      pool:
        maxidle: 16
        maxactive: 64
        idletimeout: 300s
    health:
      storagedriver:
        enabled: true
        interval: 10s
        threshold: 3
      file:
        - file: /path/to/checked/file
          interval: 10s
      http:
        - uri: http://server.to.check/must/return/200
          headers:
            Authorization: [Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==]
          statuscode: 200
          timeout: 3s
          interval: 10s
          threshold: 3
      tcp:
        - addr: redis-server.domain.com:6379
          timeout: 3s
          interval: 10s
          threshold: 3
    proxy:
      remoteurl: https://registry-1.docker.io
      username: [username]
      password: [password]
    compatibility:
      schema1:
        signingkeyfile: /etc/registry/key.json
        disablesignaturestore: true

In some instances a configuration option is **optional** but it contains child
options marked as **required**. This indicates that you can omit the parent with
all its children. However, if the parent is included, you must also include all
the children marked **required**.

## version

    version: 0.1

The `version` option is **required**. It specifies the configuration's version.
It is expected to remain a top-level field, to allow for a consistent version
check before parsing the remainder of the configuration file.

## log

The `log` subsection configures the behavior of the logging system. The logging
system outputs everything to stdout. You can adjust the granularity and format
with this configuration section.

    log:
      level: debug
      formatter: text
      fields:
        service: registry
        environment: staging

<table>
  <tr>
    <th>Parameter</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>
      <code>level</code>
    </td>
    <td>
      no
    </td>
    <td>
      Sets the sensitivity of logging output. Permitted values are
      <code>error</code>, <code>warn</code>, <code>info</code> and
      <code>debug</code>. The default is <code>info</code>.
    </td>
  </tr>
  <tr>
    <td>
      <code>formatter</code>
    </td>
    <td>
      no
    </td>
    <td>
      This selects the format of logging output. The format primarily affects how keyed
      attributes for a log line are encoded. Options are <code>text</code>, <code>json</code> or
      <code>logstash</code>. The default is <code>text</code>.
    </td>
  </tr>
    <tr>
    <td>
      <code>fields</code>
    </td>
    <td>
      no
    </td>
    <td>
      A map of field names to values. These are added to every log line for
      the context. This is useful for identifying log messages source after
      being mixed in other systems.
    </td>
</table>

## hooks

    hooks:
      - type: mail
        levels:
          - panic
        options:
          smtp:
            addr: smtp.sendhost.com:25
            username: sendername
            password: password
            insecure: true
          from: name@sendhost.com
          to:
            - name@receivehost.com

The `hooks` subsection configures the logging hooks' behavior. This subsection
includes a sequence handler which you can use for sending mail, for example.
Refer to `loglevel` to configure the level of messages printed.

## loglevel

> **DEPRECATED:** Please use [log](#log) instead.

    loglevel: debug

Permitted values are `error`, `warn`, `info` and `debug`. The default is
`info`.

## storage

    storage:
      filesystem:
        rootdirectory: /var/lib/registry
      azure:
        accountname: accountname
        accountkey: base64encodedaccountkey
        container: containername
      gcs:
        bucket: bucketname
        keyfile: /path/to/keyfile
        rootdirectory: /gcs/object/name/prefix
      s3:
        accesskey: awsaccesskey
        secretkey: awssecretkey
        region: us-west-1
        regionendpoint: http://myobjects.local
        bucket: bucketname
        encrypt: true
        keyid: mykeyid
        secure: true
        v4auth: true
        chunksize: 5242880
        rootdirectory: /s3/object/name/prefix
      swift:
        username: username
        password: password
        authurl: https://storage.myprovider.com/auth/v1.0 or https://storage.myprovider.com/v2.0 or https://storage.myprovider.com/v3/auth
        tenant: tenantname
        tenantid: tenantid
        domain: domain name for Openstack Identity v3 API
        domainid: domain id for Openstack Identity v3 API
        insecureskipverify: true
        region: fr
        container: containername
        rootdirectory: /swift/object/name/prefix
      oss:
        accesskeyid: accesskeyid
        accesskeysecret: accesskeysecret
        region: OSS region name
        endpoint: optional endpoints
        internal: optional internal endpoint
        bucket: OSS bucket
        encrypt: optional data encryption setting
        secure: optional ssl setting
        chunksize: optional size valye
        rootdirectory: optional root directory
      inmemory:
      delete:
        enabled: false
      cache:
        blobdescriptor: inmemory
      maintenance:
        uploadpurging:
          enabled: true
          age: 168h
          interval: 24h
          dryrun: false
      redirect:
        disable: false

The storage option is **required** and defines which storage backend is in use.
You must configure one backend; if you configure more, the registry returns an error. You can choose any of these backend storage drivers:

| Storage&nbsp;driver | Description
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `filesystem`        | Uses the local disk to store registry files. It is ideal for development and may be appropriate for some small-scale production applications. See the [driver's reference documentation](storage-drivers/filesystem.md). |
| `azure`             | Uses Microsoft's Azure Blob Storage. See the [driver's reference documentation](storage-drivers/azure.md).                                                                                                               |
| `gcs`               | Uses Google Cloud Storage. See the [driver's reference documentation](storage-drivers/gcs.md).                                                                                                                           |
| `s3`                | Uses Amazon's Simple Storage Service (S3) and compatible Storage Services. See the [driver's reference documentation](storage-drivers/s3.md).                                                                            |
| `swift`             | Uses Openstack Swift object storage. See the [driver's reference documentation](storage-drivers/swift.md).                                                                                                               |
| `oss`               | Uses Aliyun OSS for object storage. See the [driver's reference documentation](storage-drivers/oss.md).                                                                                                                  |

For purely tests purposes, you can use the [`inmemory` storage
driver](storage-drivers/inmemory.md). If you would like to run a registry from
volatile memory, use the [`filesystem` driver](storage-drivers/filesystem.md) on
a ramdisk.

If you are deploying a registry on Windows, be aware that a Windows volume
mounted from the host is not recommended. Instead, you can use a S3, or Azure,
backing data-store. If you do use a Windows volume, you must ensure that the
`PATH` to the mount point is within Windows' `MAX_PATH` limits (typically 255
characters). Failure to do so can result in the following error message:

    mkdir /XXX protocol error and your registry will not function properly.

### Maintenance

Currently upload purging and read-only mode are the only maintenance functions available.
These and future maintenance functions which are related to storage can be configured under
the maintenance section.

### Upload Purging

Upload purging is a background process that periodically removes orphaned files from the upload
directories of the registry.  Upload purging is enabled by default.  To
configure upload directory purging, the following parameters
must be set.


| Parameter | Required | Description
  --------- | -------- | -----------
`enabled` | yes | Set to true to enable upload purging.  Default=true. |
`age` | yes | Upload directories which are older than this age will be deleted.  Default=168h (1 week)
`interval` | yes | The interval between upload directory purging.  Default=24h.
`dryrun` | yes |  dryrun can be set to true to obtain a summary of what directories will be deleted.  Default=false.

Note: `age` and `interval` are strings containing a number with optional fraction and a unit suffix: e.g. 45m, 2h10m, 168h (1 week).

### Read-only mode

If the `readonly` section under `maintenance` has `enabled` set to `true`,
clients will not be allowed to write to the registry. This mode is useful to
temporarily prevent writes to the backend storage so a garbage collection pass
can be run.  Before running garbage collection, the registry should be
restarted with readonly's `enabled` set to true. After the garbage collection
pass finishes, the registry may be restarted again, this time with `readonly`
removed from the configuration (or set to false).

### delete

Use the `delete` subsection to enable the deletion of image blobs and manifests
by digest. It defaults to false, but it can be enabled by writing the following
on the configuration file:

    delete:
      enabled: true

### cache

Use the `cache` subsection to enable caching of data accessed in the storage
backend. Currently, the only available cache provides fast access to layer
metadata. This, if configured, uses the `blobdescriptor` field.

You can set `blobdescriptor` field to `redis` or `inmemory`.  The `redis` value uses
a Redis pool to cache layer metadata.  The `inmemory` value uses an in memory
map.

>**NOTE**: Formerly, `blobdescriptor` was known as `layerinfo`. While these
>are equivalent, `layerinfo` has been deprecated, in favor or
>`blobdescriptor`.

### redirect

The `redirect` subsection provides configuration for managing redirects from
content backends. For backends that support it, redirecting is enabled by
default. Certain deployment scenarios may prefer to route all data through the
Registry, rather than redirecting to the backend. This may be more efficient
when using a backend that is not co-located or when a registry instance is
doing aggressive caching.

Redirects can be disabled by adding a single flag `disable`, set to `true`
under the `redirect` section:

    redirect:
      disable: true


## auth

    auth:
      silly:
        realm: silly-realm
        service: silly-service
      token:
        realm: token-realm
        service: token-service
        issuer: registry-token-issuer
        rootcertbundle: /root/certs/bundle
      htpasswd:
        realm: basic-realm
        path: /path/to/htpasswd

The `auth` option is **optional**. There are
currently 3 possible auth providers, `silly`, `token` and `htpasswd`. You can configure only
one `auth` provider.

### silly

The `silly` auth is only for development purposes. It simply checks for the
existence of the `Authorization` header in the HTTP request. It has no regard for
the header's value. If the header does not exist, the `silly` auth responds with a
challenge response, echoing back the realm, service, and scope that access was
denied for.

The following values are used to configure the response:

<table>
  <tr>
    <th>Parameter</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>
      <code>realm</code>
    </td>
    <td>
      yes
    </td>
    <td>
      The realm in which the registry server authenticates.
    </td>
  </tr>
    <tr>
    <td>
      <code>service</code>
    </td>
    <td>
      yes
    </td>
    <td>
      The service being authenticated.
    </td>
  </tr>
</table>



### token

Token based authentication allows the authentication system to be decoupled from
the registry. It is a well established authentication paradigm with a high
degree of security.

<table>
  <tr>
    <th>Parameter</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>
      <code>realm</code>
    </td>
    <td>
      yes
    </td>
    <td>
      The realm in which the registry server authenticates.
    </td>
  </tr>
    <tr>
    <td>
      <code>service</code>
    </td>
    <td>
      yes
    </td>
    <td>
      The service being authenticated.
    </td>
  </tr>
    <tr>
    <td>
      <code>issuer</code>
    </td>
    <td>
      yes
    </td>
    <td>
The name of the token issuer. The issuer inserts this into
the token so it must match the value configured for the issuer.
    </td>
  </tr>
    <tr>
    <td>
      <code>rootcertbundle</code>
    </td>
    <td>
      yes
     </td>
    <td>
The absolute path to the root certificate bundle. This bundle contains the
public part of the certificates that is used to sign authentication tokens.
     </td>
  </tr>
</table>

For more information about Token based authentication configuration, see the [specification](spec/auth/token.md).

### htpasswd

The _htpasswd_ authentication backed allows one to configure basic auth using an
[Apache htpasswd
file](https://httpd.apache.org/docs/2.4/programs/htpasswd.html). Only
[`bcrypt`](http://en.wikipedia.org/wiki/Bcrypt) format passwords are supported.
Entries with other hash types will be ignored. The htpasswd file is loaded once,
at startup. If the file is invalid, the registry will display an error and will
not start.

> __WARNING:__ This authentication scheme should only be used with TLS
> configured, since basic authentication sends passwords as part of the http
> header.

<table>
  <tr>
    <th>Parameter</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>
      <code>realm</code>
    </td>
    <td>
      yes
    </td>
    <td>
      The realm in which the registry server authenticates.
    </td>
  </tr>
    <tr>
    <td>
      <code>path</code>
    </td>
    <td>
      yes
    </td>
    <td>
      Path to htpasswd file to load at startup.
    </td>
  </tr>
</table>

## middleware

The `middleware` option is **optional**. Use this option to inject middleware at
named hook points. All middleware must implement the same interface as the
object they're wrapping. This means a registry middleware must implement the
`distribution.Namespace` interface, repository middleware must implement
`distribution.Repository`, and storage middleware must implement
`driver.StorageDriver`.

Currently only one middleware, `cloudfront`, a storage middleware, is supported
in the registry implementation.

    middleware:
      registry:
        - name: ARegistryMiddleware
          options:
            foo: bar
      repository:
        - name: ARepositoryMiddleware
          options:
            foo: bar
      storage:
        - name: cloudfront
          options:
            baseurl: https://my.cloudfronted.domain.com/
            privatekey: /path/to/pem
            keypairid: cloudfrontkeypairid
            duration: 3000s

Each middleware entry has `name` and `options` entries. The `name` must
correspond to the name under which the middleware registers itself. The
`options` field is a map that details custom configuration required to
initialize the middleware. It is treated as a `map[string]interface{}`. As such,
it supports any interesting structures desired, leaving it up to the middleware
initialization function to best determine how to handle the specific
interpretation of the options.

### cloudfront

<table>
  <tr>
    <th>Parameter</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>
      <code>baseurl</code>
    </td>
    <td>
      yes
    </td>
    <td>
      <code>SCHEME://HOST[/PATH]</code> at which Cloudfront is served.
    </td>
  </tr>
    <tr>
    <td>
      <code>privatekey</code>
    </td>
    <td>
      yes
    </td>
    <td>
      Private Key for Cloudfront provided by AWS.
    </td>
  </tr>
    <tr>
    <td>
      <code>keypairid</code>
    </td>
    <td>
      yes
    </td>
    <td>
      Key pair ID provided by AWS.
    </td>
  </tr>
    <tr>
    <td>
      <code>duration</code>
    </td>
    <td>
      no
    </td>
    <td>
      Specify a `duration` by providing an integer and a unit. Valid time units are `ns`, `us` (or `µs`), `ms`, `s`, `m`, `h`. For example, `3000s` is a valid duration; there should be no space between the integer and unit. If you do not specify a `duration` or specify an integer without a time unit, this defaults to 20 minutes.
    </td>
  </tr>
</table>


## reporting

    reporting:
      bugsnag:
        apikey: bugsnagapikey
        releasestage: bugsnagreleasestage
        endpoint: bugsnagendpoint
      newrelic:
        licensekey: newreliclicensekey
        name: newrelicname
        verbose: true

The `reporting` option is **optional** and configures error and metrics
reporting tools. At the moment only two services are supported, [New
Relic](http://newrelic.com/) and [Bugsnag](http://bugsnag.com), a valid
configuration may contain both.

### bugsnag

<table>
  <tr>
    <th>Parameter</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>
      <code>apikey</code>
    </td>
    <td>
      yes
    </td>
    <td>
      API Key provided by Bugsnag
    </td>
  </tr>
  <tr>
    <td>
      <code>releasestage</code>
    </td>
    <td>
      no
    </td>
    <td>
      Tracks where the registry is deployed, for example,
      <code>production</code>,<code>staging</code>, or
      <code>development</code>.
    </td>
  </tr>
  <tr>
    <td>
      <code>endpoint</code>
    </td>
    <td>
      no
    </td>
    <td>
      Specify the enterprise Bugsnag endpoint.
    </td>
  </tr>
</table>


### newrelic

<table>
  <tr>
    <th>Parameter</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>
      <code>licensekey</code>
    </td>
    <td>
      yes
    </td>
    <td>
      License key provided by New Relic.
    </td>
  </tr>
   <tr>
    <td>
      <code>name</code>
    </td>
    <td>
      no
    </td>
    <td>
      New Relic application name.
    </td>
  </tr>
     <tr>
    <td>
      <code>verbose</code>
    </td>
    <td>
      no
    </td>
    <td>
      Enable New Relic debugging output on stdout.
    </td>
  </tr>
</table>

## http

    http:
      addr: localhost:5000
      net: tcp
      prefix: /my/nested/registry/
      host: https://myregistryaddress.org:5000
      secret: asecretforlocaldevelopment
      relativeurls: false
      tls:
        certificate: /path/to/x509/public
        key: /path/to/x509/private
        clientcas:
          - /path/to/ca.pem
          - /path/to/another/ca.pem
      debug:
        addr: localhost:5001
      headers:
        X-Content-Type-Options: [nosniff]

The `http` option details the configuration for the HTTP server that hosts the registry.

<table>
  <tr>
    <th>Parameter</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>
      <code>addr</code>
    </td>
    <td>
      yes
    </td>
    <td>
     The address for which the server should accept connections. The form depends on a network type (see <code>net</code> option):
     <code>HOST:PORT</code> for tcp and <code>FILE</code> for a unix socket.
    </td>
  </tr>
  <tr>
    <td>
      <code>net</code>
    </td>
    <td>
      no
    </td>
    <td>
     The network which is used to create a listening socket. Known networks are <code>unix</code> and <code>tcp</code>.
     The default empty value means tcp.
    </td>
  </tr>
  <tr>
    <td>
      <code>prefix</code>
    </td>
    <td>
      no
    </td>
    <td>
If the server does not run at the root path use this value to specify the
prefix. The root path is the section before <code>v2</code>. It
should have both preceding and trailing slashes, for example <code>/path/</code>.
    </td>
  </tr>
  <tr>
    <td>
      <code>host</code>
    </td>
    <td>
      no
    </td>
    <td>
This parameter specifies an externally-reachable address for the registry, as a
fully qualified URL. If present, it is used when creating generated URLs.
Otherwise, these URLs are derived from client requests.
    </td>
  </tr>
  <tr>
    <td>
      <code>secret</code>
    </td>
    <td>
      yes
    </td>
    <td>
A random piece of data. This is used to sign state that may be stored with the
client to protect against tampering. For production environments you should generate a
random piece of data using a cryptographically secure random generator. This
configuration parameter may be omitted, in which case the registry will automatically
generate a secret at launch.
<p />
<b>WARNING: If you are building a cluster of registries behind a load balancer, you MUST
ensure the secret is the same for all registries.</b>
    </td>
  </tr>
  <tr>
    <td>
      <code>relativeurls</code>
    </td>
    <td>
      no
    </td>
    <td>
       Specifies that the registry should return relative URLs in Location headers.
       The client is responsible for resolving the correct URL.  This option is not
       compatible with Docker 1.7 and earlier.
    </td>
  </tr>
</table>


### tls

The `tls` struct within `http` is **optional**. Use this to configure TLS
for the server. If you already have a server such as Nginx or Apache running on
the same host as the registry, you may prefer to configure TLS termination there
and proxy connections to the registry server.

<table>
  <tr>
    <th>Parameter</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>
      <code>certificate</code>
    </td>
    <td>
      yes
    </td>
    <td>
       Absolute path to x509 cert file
    </td>
  </tr>
    <tr>
    <td>
      <code>key</code>
    </td>
    <td>
      yes
    </td>
    <td>
      Absolute path to x509 private key file.
    </td>
  </tr>
  <tr>
    <td>
      <code>clientcas</code>
    </td>
    <td>
      no
    </td>
    <td>
      An array of absolute paths to a x509 CA file
    </td>
  </tr>
</table>


### debug

The `debug` option is **optional** . Use it to configure a debug server that
can be helpful in diagnosing problems. The debug endpoint can be used for
monitoring registry metrics and health, as well as profiling. Sensitive
information may be available via the debug endpoint. Please be certain that
access to the debug endpoint is locked down in a production environment.

The `debug` section takes a single, required `addr` parameter. This parameter
specifies the `HOST:PORT` on which the debug server should accept connections.


### headers

The `headers` option is **optional** . Use it to specify headers that the HTTP
server should include in responses. This can be used for security headers such
as `Strict-Transport-Security`.

The `headers` option should contain an option for each header to include, where
the parameter name is the header's name, and the parameter value a list of the
header's payload values.

Including `X-Content-Type-Options: [nosniff]` is recommended, so that browsers
will not interpret content as HTML if they are directed to load a page from the
registry. This header is included in the example configuration files.


## notifications

    notifications:
      endpoints:
        - name: alistener
          disabled: false
          url: https://my.listener.com/event
          headers: <http.Header>
          timeout: 500
          threshold: 5
          backoff: 1000

The notifications option is **optional** and currently may contain a single
option, `endpoints`.

### endpoints

Endpoints is a list of named services (URLs) that can accept event notifications.

<table>
  <tr>
    <th>Parameter</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>
      <code>name</code>
    </td>
    <td>
      yes
    </td>
    <td>
A human readable name for the service.
</td>
  </tr>
  <tr>
    <td>
      <code>disabled</code>
    </td>
    <td>
      no
    </td>
    <td>
A boolean to enable/disable notifications for a service.
    </td>
  </tr>
  <tr>
    <td>
      <code>url</code>
    </td>
    <td>
    yes
    </td>
    <td>
The URL to which events should be published.
    </td>
  </tr>
   <tr>
    <td>
      <code>headers</code>
    </td>
    <td>
      yes
    </td>
    <td>
      Static headers to add to each request. Each header's name should be a key
      underneath headers, and each value is a list of payloads for that
      header name. Note that values must always be lists.
    </td>
  </tr>
  <tr>
    <td>
      <code>timeout</code>
    </td>
    <td>
      yes
    </td>
    <td>
      An HTTP timeout value. This field takes a positive integer and an optional
      suffix indicating the unit of time. Possible units are:
      <ul>
        <li><code>ns</code> (nanoseconds)</li>
        <li><code>us</code> (microseconds)</li>
        <li><code>ms</code> (milliseconds)</li>
        <li><code>s</code> (seconds)</li>
        <li><code>m</code> (minutes)</li>
        <li><code>h</code> (hours)</li>
      </ul>
    If you omit the suffix, the system interprets the value as nanoseconds.
    </td>
  </tr>
  <tr>
    <td>
      <code>threshold</code>
    </td>
    <td>
      yes
    </td>
    <td>
      An integer specifying how long to wait before backing off a failure.
    </td>
  </tr>
  <tr>
    <td>
      <code>backoff</code>
    </td>
    <td>
      yes
    </td>
    <td>
      How long the system backs off before retrying. This field takes a positive
      integer and an optional suffix indicating the unit of time. Possible units
      are:
      <ul>
        <li><code>ns</code> (nanoseconds)</li>
        <li><code>us</code> (microseconds)</li>
        <li><code>ms</code> (milliseconds)</li>
        <li><code>s</code> (seconds)</li>
        <li><code>m</code> (minutes)</li>
        <li><code>h</code> (hours)</li>
      </ul>
    If you omit the suffix, the system interprets the value as nanoseconds.
    </td>
  </tr>
</table>


## redis

    redis:
      addr: localhost:6379
      password: asecret
      db: 0
      dialtimeout: 10ms
      readtimeout: 10ms
      writetimeout: 10ms
      pool:
        maxidle: 16
        maxactive: 64
        idletimeout: 300s

Declare parameters for constructing the redis connections. Registry instances
may use the Redis instance for several applications. The current purpose is
caching information about immutable blobs. Most of the options below control
how the registry connects to redis. You can control the pool's behavior
with the [pool](#pool) subsection.

It's advisable to configure Redis itself with the **allkeys-lru** eviction policy
as the registry does not set an expire value on keys.

<table>
  <tr>
    <th>Parameter</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>
      <code>addr</code>
    </td>
    <td>
      yes
    </td>
    <td>
      Address (host and port) of redis instance.
    </td>
  </tr>
  <tr>
    <td>
      <code>password</code>
    </td>
    <td>
      no
    </td>
    <td>
      A password used to authenticate to the redis instance.
    </td>
  </tr>
  <tr>
    <td>
      <code>db</code>
    </td>
    <td>
      no
    </td>
    <td>
      Selects the db for each connection.
    </td>
  </tr>
  <tr>
    <td>
      <code>dialtimeout</code>
    </td>
    <td>
      no
    </td>
    <td>
      Timeout for connecting to a redis instance.
    </td>
  </tr>
  <tr>
    <td>
      <code>readtimeout</code>
    </td>
    <td>
      no
    </td>
    <td>
      Timeout for reading from redis connections.
    </td>
  </tr>
  <tr>
    <td>
      <code>writetimeout</code>
    </td>
    <td>
      no
    </td>
    <td>
      Timeout for writing to redis connections.
    </td>
  </tr>
</table>


### pool

    pool:
      maxidle: 16
      maxactive: 64
      idletimeout: 300s

Configure the behavior of the Redis connection pool.

<table>
  <tr>
    <th>Parameter</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>
      <code>maxidle</code>
    </td>
    <td>
      no
    </td>
    <td>
      Sets the maximum number of idle connections.
    </td>
  </tr>
  <tr>
    <td>
      <code>maxactive</code>
    </td>
    <td>
      no
    </td>
    <td>
      sets the maximum number of connections that should
  be opened before blocking a connection request.
    </td>
  </tr>
  <tr>
    <td>
      <code>idletimeout</code>
    </td>
    <td>
      no
    </td>
    <td>
      sets the amount time to wait before closing
  inactive connections.
    </td>
  </tr>
</table>

## health

    health:
      storagedriver:
        enabled: true
        interval: 10s
        threshold: 3
      file:
        - file: /path/to/checked/file
          interval: 10s
      http:
        - uri: http://server.to.check/must/return/200
          headers:
            Authorization: [Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==]
          statuscode: 200
          timeout: 3s
          interval: 10s
          threshold: 3
      tcp:
        - addr: redis-server.domain.com:6379
          timeout: 3s
          interval: 10s
          threshold: 3

The health option is **optional**. It may contain preferences for a periodic
health check on the storage driver's backend storage, and optional periodic
checks on local files, HTTP URIs, and/or TCP servers. The results of the health
checks are available at /debug/health on the debug HTTP server if the debug
HTTP server is enabled (see http section).

### storagedriver

storagedriver contains options for a health check on the configured storage
driver's backend storage. enabled must be set to true for this health check to
be active.

<table>
  <tr>
    <th>Parameter</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>
      <code>enabled</code>
    </td>
    <td>
      yes
    </td>
    <td>
"true" to enable the storage driver health check or "false" to disable it.
</td>
  </tr>
  <tr>
    <td>
      <code>interval</code>
    </td>
    <td>
      no
    </td>
    <td>
      The length of time to wait between repetitions of the check. This field
      takes a positive integer and an optional suffix indicating the unit of
      time. Possible units are:
      <ul>
        <li><code>ns</code> (nanoseconds)</li>
        <li><code>us</code> (microseconds)</li>
        <li><code>ms</code> (milliseconds)</li>
        <li><code>s</code> (seconds)</li>
        <li><code>m</code> (minutes)</li>
        <li><code>h</code> (hours)</li>
      </ul>
    If you omit the suffix, the system interprets the value as nanoseconds.
    The default value is 10 seconds if this field is omitted.
    </td>
  </tr>
  <tr>
    <td>
      <code>threshold</code>
    </td>
    <td>
      no
    </td>
    <td>
      An integer specifying the number of times the check must fail before the
      check triggers an unhealthy state. If this filed is not specified, a
      single failure will trigger an unhealthy state.
    </td>
  </tr>
</table>

### file

file is a list of paths to be periodically checked for the existence of a file.
If a file exists at the given path, the health check will fail. This can be
used as a way of bringing a registry out of rotation by creating a file.

<table>
  <tr>
    <th>Parameter</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>
      <code>file</code>
    </td>
    <td>
      yes
    </td>
    <td>
The path to check for the existence of a file.
</td>
  </tr>
  <tr>
    <td>
      <code>interval</code>
    </td>
    <td>
      no
    </td>
    <td>
      The length of time to wait between repetitions of the check. This field
      takes a positive integer and an optional suffix indicating the unit of
      time. Possible units are:
      <ul>
        <li><code>ns</code> (nanoseconds)</li>
        <li><code>us</code> (microseconds)</li>
        <li><code>ms</code> (milliseconds)</li>
        <li><code>s</code> (seconds)</li>
        <li><code>m</code> (minutes)</li>
        <li><code>h</code> (hours)</li>
      </ul>
    If you omit the suffix, the system interprets the value as nanoseconds.
    The default value is 10 seconds if this field is omitted.
    </td>
  </tr>
</table>

### http

http is a list of HTTP URIs to be periodically checked with HEAD requests. If
a HEAD request doesn't complete or returns an unexpected status code, the
health check will fail.

<table>
  <tr>
    <th>Parameter</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>
      <code>uri</code>
    </td>
    <td>
      yes
    </td>
    <td>
The URI to check.
</td>
  </tr>
   <tr>
    <td>
      <code>headers</code>
    </td>
    <td>
      no
    </td>
    <td>
      Static headers to add to each request. Each header's name should be a key
      underneath headers, and each value is a list of payloads for that
      header name. Note that values must always be lists.
    </td>
  </tr>
  <tr>
    <td>
      <code>statuscode</code>
    </td>
    <td>
      no
    </td>
    <td>
Expected status code from the HTTP URI. Defaults to 200.
</td>
  </tr>
  <tr>
    <td>
      <code>timeout</code>
    </td>
    <td>
      no
    </td>
    <td>
      The length of time to wait before timing out the HTTP request. This field
      takes a positive integer and an optional suffix indicating the unit of
      time. Possible units are:
      <ul>
        <li><code>ns</code> (nanoseconds)</li>
        <li><code>us</code> (microseconds)</li>
        <li><code>ms</code> (milliseconds)</li>
        <li><code>s</code> (seconds)</li>
        <li><code>m</code> (minutes)</li>
        <li><code>h</code> (hours)</li>
      </ul>
    If you omit the suffix, the system interprets the value as nanoseconds.
    </td>
  </tr>
  <tr>
    <td>
      <code>interval</code>
    </td>
    <td>
      no
    </td>
    <td>
      The length of time to wait between repetitions of the check. This field
      takes a positive integer and an optional suffix indicating the unit of
      time. Possible units are:
      <ul>
        <li><code>ns</code> (nanoseconds)</li>
        <li><code>us</code> (microseconds)</li>
        <li><code>ms</code> (milliseconds)</li>
        <li><code>s</code> (seconds)</li>
        <li><code>m</code> (minutes)</li>
        <li><code>h</code> (hours)</li>
      </ul>
    If you omit the suffix, the system interprets the value as nanoseconds.
    The default value is 10 seconds if this field is omitted.
    </td>
  </tr>
  <tr>
    <td>
      <code>threshold</code>
    </td>
    <td>
      no
    </td>
    <td>
      An integer specifying the number of times the check must fail before the
      check triggers an unhealthy state. If this filed is not specified, a
      single failure will trigger an unhealthy state.
    </td>
  </tr>
</table>

### tcp

tcp is a list of TCP addresses to be periodically checked with connection
attempts. The addresses must include port numbers. If a connection attempt
fails, the health check will fail.

<table>
  <tr>
    <th>Parameter</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>
      <code>addr</code>
    </td>
    <td>
      yes
    </td>
    <td>
The TCP address to connect to, including a port number.
</td>
  </tr>
  <tr>
    <td>
      <code>timeout</code>
    </td>
    <td>
      no
    </td>
    <td>
      The length of time to wait before timing out the TCP connection. This
      field takes a positive integer and an optional suffix indicating the unit
      of time. Possible units are:
      <ul>
        <li><code>ns</code> (nanoseconds)</li>
        <li><code>us</code> (microseconds)</li>
        <li><code>ms</code> (milliseconds)</li>
        <li><code>s</code> (seconds)</li>
        <li><code>m</code> (minutes)</li>
        <li><code>h</code> (hours)</li>
      </ul>
    If you omit the suffix, the system interprets the value as nanoseconds.
    </td>
  </tr>
  <tr>
    <td>
      <code>interval</code>
    </td>
    <td>
      no
    </td>
    <td>
      The length of time to wait between repetitions of the check. This field
      takes a positive integer and an optional suffix indicating the unit of
      time. Possible units are:
      <ul>
        <li><code>ns</code> (nanoseconds)</li>
        <li><code>us</code> (microseconds)</li>
        <li><code>ms</code> (milliseconds)</li>
        <li><code>s</code> (seconds)</li>
        <li><code>m</code> (minutes)</li>
        <li><code>h</code> (hours)</li>
      </ul>
    If you omit the suffix, the system interprets the value as nanoseconds.
    The default value is 10 seconds if this field is omitted.
    </td>
  </tr>
  <tr>
    <td>
      <code>threshold</code>
    </td>
    <td>
      no
    </td>
    <td>
      An integer specifying the number of times the check must fail before the
      check triggers an unhealthy state. If this filed is not specified, a
      single failure will trigger an unhealthy state.
    </td>
  </tr>
</table>

## Proxy

    proxy:
      remoteurl: https://registry-1.docker.io
      username: [username]
      password: [password]

Proxy enables a registry to be configured as a pull through cache to the official Docker Hub.  See [mirror](mirror.md) for more information. Pushing to a registry configured as a pull through cache is currently unsupported.

<table>
  <tr>
    <th>Parameter</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>
      <code>remoteurl</code>
    </td>
    <td>
      yes
    </td>
    <td>
     The URL of the official Docker Hub
    </td>
  </tr>
  <tr>
    <td>
      <code>username</code>
    </td>
    <td>
      no
    </td>
    <td>
     The username of the Docker Hub account
    </td>
  </tr>
  <tr>
    <td>
      <code>password</code>
    </td>
    <td>
      no
    </td>
    <td>
     The password for the official Docker Hub account
    </td>
  </tr>
</table>

To enable pulling private repositories (e.g. `batman/robin`) a username and password for user `batman` must be specified.  Note: These private repositories will be stored in the proxy cache's storage and relevant measures should be taken to protect access to this.

## Compatibility

    compatibility:
      schema1:
        signingkeyfile: /etc/registry/key.json
        disablesignaturestore: true

Configure handling of older and deprecated features. Each subsection
defines a such a feature with configurable behavior.

### Schema1

<table>
  <tr>
    <th>Parameter</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>
      <code>signingkeyfile</code>
    </td>
    <td>
      no
    </td>
    <td>
     The signing private key used for adding signatures to schema1 manifests.
     If no signing key is provided, a new ECDSA key will be generated on
     startup.
    </td>
  </tr>
  <tr>
    <td>
      <code>disablesignaturestore</code>
    </td>
    <td>
      no
    </td>
    <td>
     Disables storage of signatures attached to schema1 manifests. By default
     signatures are detached from schema1 manifests, stored, and reattached
     when the manifest is requested. When this is true, the storage is disabled
     and a new signature is always generated for schema1 manifests using the
     schema1 signing key. Disabling signature storage will cause all newly
     uploaded signatures to be discarded. Existing stored signatures will not
     be removed but they will not be re-attached to the corresponding manifest.
    </td>
  </tr>
</table>

## Example: Development configuration

The following is a simple example you can use for local development:

    version: 0.1
    log:
      level: debug
    storage:
        filesystem:
            rootdirectory: /var/lib/registry
    http:
        addr: localhost:5000
        secret: asecretforlocaldevelopment
        debug:
            addr: localhost:5001

The above configures the registry instance to run on port `5000`, binding to
`localhost`, with the `debug` server enabled. Registry data storage is in the
`/var/lib/registry` directory. Logging is in `debug` mode, which is the most
verbose.

A similar simple configuration is available at
[config-example.yml](https://github.com/docker/distribution/blob/master/cmd/registry/config-example.yml).
Both are generally useful for local development.


## Example: Middleware configuration

This example illustrates how to configure storage middleware in a registry.
Middleware allows the registry to serve layers via a content delivery network
(CDN). This is useful for reducing requests to the storage layer.

Currently, the registry supports [Amazon
Cloudfront](http://aws.amazon.com/cloudfront/). You can only use Cloudfront in
conjunction with the S3 storage driver.

<table>
  <tr>
    <th>Parameter</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>name</code></td>
    <td>The storage middleware name. Currently <code>cloudfront</code> is an accepted value.</td>
  </tr>
  <tr>
    <td><code>disabled<code></td>
    <td>Set to <code>false</code> to easily disable the middleware.</td>
  </tr>
  <tr>
    <td><code>options:</code></td>
    <td>
    A set of key/value options to configure the middleware.
    <ul>
    <li><code>baseurl:</code> The Cloudfront base URL.</li>
    <li><code>privatekey:</code> The location of your AWS private key on the filesystem. </li>
    <li><code>keypairid:</code> The ID of your Cloudfront keypair. </li>
    <li><code>duration:</code> The duration in minutes for which the URL is valid. Default is 20. </li>
    </ul>
    </td>
  </tr>
</table>

The following example illustrates these values:

    middleware:
        storage:
            - name: cloudfront
              disabled: false
              options:
                 baseurl: http://d111111abcdef8.cloudfront.net
                 privatekey: /path/to/asecret.pem
                 keypairid: asecret
                 duration: 60


>**Note**: Cloudfront keys exist separately to other AWS keys.  See
>[the documentation on AWS credentials](http://docs.aws.amazon.com/general/latest/gr/aws-security-credentials.html)
>for more information.
