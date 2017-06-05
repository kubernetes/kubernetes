<!--[metadata]>
+++
title = "Authenticating proxy with nginx"
description = "Restricting access to your registry using a nginx proxy"
keywords = ["registry, on-prem, images, tags, repository, distribution, nginx, proxy, authentication, TLS, recipe, advanced"]
+++
<![end-metadata]-->

# Authenticating proxy with nginx


## Use-case

People already relying on a nginx proxy to authenticate their users to other services might want to leverage it and have Registry communications tunneled through the same pipeline.

Usually, that includes enterprise setups using LDAP/AD on the backend and a SSO mechanism fronting their internal http portal.

### Alternatives

If you just want authentication for your registry, and are happy maintaining users access separately, you should really consider sticking with the native [basic auth registry feature](deploying.md#native-basic-auth).

### Solution

With the method presented here, you implement basic authentication for docker engines in a reverse proxy that sits in front of your registry.

While we use a simple htpasswd file as an example, any other nginx authentication backend should be fairly easy to implement once you are done with the example.

We also implement push restriction (to a limited user group) for the sake of the example. Again, you should modify this to fit your mileage.

### Gotchas

While this model gives you the ability to use whatever authentication backend you want through the secondary authentication mechanism implemented inside your proxy, it also requires that you move TLS termination from the Registry to the proxy itself.

Furthermore, introducing an extra http layer in your communication pipeline will make it more complex to deploy, maintain, and debug, and will possibly create issues. Make sure the extra complexity is required.

For instance, Amazon's Elastic Load Balancer (ELB) in HTTPS mode already sets the following client header:

```
X-Real-IP
X-Forwarded-For
X-Forwarded-Proto
```

So if you have an nginx sitting behind it, should remove these lines from the example config below:

```
X-Real-IP         $remote_addr; # pass on real client's IP
X-Forwarded-For   $proxy_add_x_forwarded_for;
X-Forwarded-Proto $scheme;
```

Otherwise nginx will reset the ELB's values, and the requests will not be routed properly. For more information, see [#970](https://github.com/docker/distribution/issues/970).

## Setting things up

Read again [the requirements](recipes.md#requirements).

Ready?

--

Create the required directories

```
mkdir -p auth
mkdir -p data
```

Create the main nginx configuration you will use.

```

cat <<EOF > auth/nginx.conf
events {
    worker_connections  1024;
}

http {
  
  upstream docker-registry {
    server registry:5000;
  }

  ## Set a variable to help us decide if we need to add the
  ## 'Docker-Distribution-Api-Version' header.
  ## The registry always sets this header.
  ## In the case of nginx performing auth, the header will be unset
  ## since nginx is auth-ing before proxying.
  map \$upstream_http_docker_distribution_api_version \$docker_distribution_api_version {
    'registry/2.0' '';
    default registry/2.0;
  }

  server {
    listen 443 ssl;
    server_name myregistrydomain.com;

    # SSL
    ssl_certificate /etc/nginx/conf.d/domain.crt;
    ssl_certificate_key /etc/nginx/conf.d/domain.key;
  
    # Recommendations from https://raymii.org/s/tutorials/Strong_SSL_Security_On_nginx.html
    ssl_protocols TLSv1.1 TLSv1.2;
    ssl_ciphers 'EECDH+AESGCM:EDH+AESGCM:AES256+EECDH:AES256+EDH';
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
  
    # disable any limits to avoid HTTP 413 for large image uploads
    client_max_body_size 0;
  
    # required to avoid HTTP 411: see Issue #1486 (https://github.com/docker/docker/issues/1486)
    chunked_transfer_encoding on;
  
    location /v2/ {
      # Do not allow connections from docker 1.5 and earlier
      # docker pre-1.6.0 did not properly set the user agent on ping, catch "Go *" user agents
      if (\$http_user_agent ~ "^(docker\/1\.(3|4|5(?!\.[0-9]-dev))|Go ).*\$" ) {
        return 404;
      }
  
      # To add basic authentication to v2 use auth_basic setting.
      auth_basic "Registry realm";
      auth_basic_user_file /etc/nginx/conf.d/nginx.htpasswd;
  
      ## If $docker_distribution_api_version is empty, the header will not be added.
      ## See the map directive above where this variable is defined.
      add_header 'Docker-Distribution-Api-Version' \$docker_distribution_api_version always;
  
      proxy_pass                          http://docker-registry;
      proxy_set_header  Host              \$http_host;   # required for docker client's sake
      proxy_set_header  X-Real-IP         \$remote_addr; # pass on real client's IP
      proxy_set_header  X-Forwarded-For   \$proxy_add_x_forwarded_for;
      proxy_set_header  X-Forwarded-Proto \$scheme;
      proxy_read_timeout                  900;
    }
  }
}
EOF
```

Now create a password file for "testuser" and "testpassword"

```
docker run --rm --entrypoint htpasswd registry:2 -bn testuser testpassword > auth/nginx.htpasswd
```

Copy over your certificate files

```
cp domain.crt auth
cp domain.key auth
```

Now create your compose file

```
cat <<EOF > docker-compose.yml
nginx:
  image: "nginx:1.9"
  ports:
    - 5043:443
  links:
    - registry:registry
  volumes:
    - ./auth:/etc/nginx/conf.d
    - ./auth/nginx.conf:/etc/nginx/nginx.conf:ro

registry:
  image: registry:2
  ports:
    - 127.0.0.1:5000:5000
  volumes:
    - `pwd`./data:/var/lib/registry
EOF
```

## Starting and stopping

Now, start your stack:

    docker-compose up -d

Login with a "push" authorized user (using `testuser` and `testpassword`), then tag and push your first image:

    docker login -p=testuser -u=testpassword -e=root@example.ch myregistrydomain.com:5043
    docker tag ubuntu myregistrydomain.com:5043/test
    docker push myregistrydomain.com:5043/test
    docker pull myregistrydomain.com:5043/test
