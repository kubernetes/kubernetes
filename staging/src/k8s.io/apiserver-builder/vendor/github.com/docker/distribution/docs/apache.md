<!--[metadata]>
+++
title = "Authenticating proxy with apache"
description = "Restricting access to your registry using an apache proxy"
keywords = ["registry, on-prem, images, tags, repository, distribution, authentication, proxy, apache, httpd, TLS, recipe, advanced"]
+++
<![end-metadata]-->

# Authenticating proxy with apache

## Use-case

People already relying on an apache proxy to authenticate their users to other services might want to leverage it and have Registry communications tunneled through the same pipeline.

Usually, that includes enterprise setups using LDAP/AD on the backend and a SSO mechanism fronting their internal http portal.

### Alternatives

If you just want authentication for your registry, and are happy maintaining users access separately, you should really consider sticking with the native [basic auth registry feature](deploying.md#native-basic-auth). 

### Solution

With the method presented here, you implement basic authentication for docker engines in a reverse proxy that sits in front of your registry.

While we use a simple htpasswd file as an example, any other apache authentication backend should be fairly easy to implement once you are done with the example.

We also implement push restriction (to a limited user group) for the sake of the example. Again, you should modify this to fit your mileage. 

### Gotchas

While this model gives you the ability to use whatever authentication backend you want through the secondary authentication mechanism implemented inside your proxy, it also requires that you move TLS termination from the Registry to the proxy itself.

Furthermore, introducing an extra http layer in your communication pipeline will make it more complex to deploy, maintain, and debug, and will possibly create issues.

## Setting things up

Read again [the requirements](recipes.md#requirements).

Ready?

Run the following script:

```
mkdir -p auth
mkdir -p data

# This is the main apache configuration you will use
cat <<EOF > auth/httpd.conf
LoadModule headers_module modules/mod_headers.so

LoadModule authn_file_module modules/mod_authn_file.so
LoadModule authn_core_module modules/mod_authn_core.so
LoadModule authz_groupfile_module modules/mod_authz_groupfile.so
LoadModule authz_user_module modules/mod_authz_user.so
LoadModule authz_core_module modules/mod_authz_core.so
LoadModule auth_basic_module modules/mod_auth_basic.so
LoadModule access_compat_module modules/mod_access_compat.so

LoadModule log_config_module modules/mod_log_config.so

LoadModule ssl_module modules/mod_ssl.so

LoadModule proxy_module modules/mod_proxy.so
LoadModule proxy_http_module modules/mod_proxy_http.so

LoadModule unixd_module modules/mod_unixd.so

<IfModule ssl_module>
    SSLRandomSeed startup builtin
    SSLRandomSeed connect builtin
</IfModule>

<IfModule unixd_module>
    User daemon
    Group daemon
</IfModule>

ServerAdmin you@example.com

ErrorLog /proc/self/fd/2

LogLevel warn

<IfModule log_config_module>
    LogFormat "%h %l %u %t \"%r\" %>s %b \"%{Referer}i\" \"%{User-Agent}i\"" combined
    LogFormat "%h %l %u %t \"%r\" %>s %b" common

    <IfModule logio_module>
      LogFormat "%h %l %u %t \"%r\" %>s %b \"%{Referer}i\" \"%{User-Agent}i\" %I %O" combinedio
    </IfModule>

    CustomLog /proc/self/fd/1 common
</IfModule>

ServerRoot "/usr/local/apache2"

Listen 5043

<Directory />
    AllowOverride none
    Require all denied
</Directory>

<VirtualHost *:5043>

  ServerName myregistrydomain.com

  SSLEngine on
  SSLCertificateFile /usr/local/apache2/conf/domain.crt
  SSLCertificateKeyFile /usr/local/apache2/conf/domain.key

  ## SSL settings recommandation from: https://raymii.org/s/tutorials/Strong_SSL_Security_On_Apache2.html
  # Anti CRIME
  SSLCompression off

  # POODLE and other stuff
  SSLProtocol all -SSLv2 -SSLv3 -TLSv1

  # Secure cypher suites
  SSLCipherSuite EECDH+AESGCM:EDH+AESGCM:AES256+EECDH:AES256+EDH
  SSLHonorCipherOrder on

  Header always set "Docker-Distribution-Api-Version" "registry/2.0"
  Header onsuccess set "Docker-Distribution-Api-Version" "registry/2.0"
  RequestHeader set X-Forwarded-Proto "https"

  ProxyRequests     off
  ProxyPreserveHost on

  # no proxy for /error/ (Apache HTTPd errors messages)
  ProxyPass /error/ !

  ProxyPass        /v2 http://registry:5000/v2
  ProxyPassReverse /v2 http://registry:5000/v2

  <Location /v2>
    Order deny,allow
    Allow from all
    AuthName "Registry Authentication"
    AuthType basic
    AuthUserFile "/usr/local/apache2/conf/httpd.htpasswd"
    AuthGroupFile "/usr/local/apache2/conf/httpd.groups"

    # Read access to authentified users
    <Limit GET HEAD>
      Require valid-user
    </Limit>

    # Write access to docker-deployer only
    <Limit POST PUT DELETE PATCH>
      Require group pusher
    </Limit>

  </Location>

</VirtualHost>
EOF

# Now, create a password file for "testuser" and "testpassword"
docker run --entrypoint htpasswd httpd:2.4 -Bbn testuser testpassword > auth/httpd.htpasswd
# Create another one for "testuserpush" and "testpasswordpush"
docker run --entrypoint htpasswd httpd:2.4 -Bbn testuserpush testpasswordpush >> auth/httpd.htpasswd

# Create your group file
echo "pusher: testuserpush" > auth/httpd.groups

# Copy over your certificate files
cp domain.crt auth
cp domain.key auth

# Now create your compose file

cat <<EOF > docker-compose.yml
apache:
  image: "httpd:2.4"
  hostname: myregistrydomain.com
  ports:
    - 5043:5043
  links:
    - registry:registry
  volumes:
    - `pwd`/auth:/usr/local/apache2/conf

registry:
  image: registry:2
  ports:
    - 127.0.0.1:5000:5000
  volumes:
    - `pwd`/data:/var/lib/registry

EOF
```

## Starting and stopping

Now, start your stack:

    docker-compose up -d

Login with a "push" authorized user (using `testuserpush` and `testpasswordpush`), then tag and push your first image: 

    docker login myregistrydomain.com:5043
    docker tag ubuntu myregistrydomain.com:5043/test
    docker push myregistrydomain.com:5043/test

Now, login with a "pull-only" user (using `testuser` and `testpassword`), then pull back the image:

    docker login myregistrydomain.com:5043
    docker pull myregistrydomain.com:5043/test

Verify that the "pull-only" can NOT push:

    docker push myregistrydomain.com:5043/test
