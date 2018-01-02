# Docker Compose V1 + V2 registry

This compose configuration configures a `v1` and `v2` registry behind an `nginx`
proxy. By default, you can access the combined registry at `localhost:5000`.

The configuration does not support pushing images to `v2` and pulling from `v1`.
If a `docker` client has a version less than 1.6, Nginx will route its requests
to the 1.0 registry. Requests from newer clients will route to the 2.0 registry.

### Install Docker Compose

1. Open a new terminal on the host with your `distribution` source.

2. Get the `docker-compose` binary.

		$ sudo wget https://github.com/docker/compose/releases/download/1.1.0/docker-compose-`uname  -s`-`uname -m` -O /usr/local/bin/docker-compose

	This command installs the binary in the `/usr/local/bin` directory. 
	
3. Add executable permissions to the binary.

		$  sudo chmod +x /usr/local/bin/docker-compose
		
## Build and run with Compose
	
1. In your terminal, navigate to the `distribution/contrib/compose` directory

	This directory includes a single `docker-compose.yml` configuration.
	
		nginx:
			build: "nginx"
			ports:
				- "5000:5000"
			links:
				- registryv1:registryv1
				- registryv2:registryv2
		registryv1:
			image: registry
			ports:
				- "5000"
		registryv2:
			build: "../../"
			ports:
				- "5000"

	This configuration builds a new `nginx` image as specified by the
	`nginx/Dockerfile` file. The 1.0 registry comes from Docker's official
	public image. Finally, the registry 2.0 image is built from the
	`distribution/Dockerfile` you've used previously.
 		
2. Get a registry 1.0 image.

		$ docker pull registry:0.9.1 

	The Compose configuration looks for this image locally. If you don't do this
	step, later steps can fail.
	
3. Build `nginx`, the registry 2.0 image, and 

		$ docker-compose build
		registryv1 uses an image, skipping
		Building registryv2...
		Step 0 : FROM golang:1.4
		
		...
		
		Removing intermediate container 9f5f5068c3f3
		Step 4 : COPY docker-registry-v2.conf /etc/nginx/docker-registry-v2.conf
		 ---> 74acc70fa106
		Removing intermediate container edb84c2b40cb
		Successfully built 74acc70fa106
		
	The commmand outputs its progress until it completes.

4. Start your configuration with compose.

		$ docker-compose up
		Recreating compose_registryv1_1...
		Recreating compose_registryv2_1...
		Recreating compose_nginx_1...
		Attaching to compose_registryv1_1, compose_registryv2_1, compose_nginx_1
		...
	

5. In another terminal, display the running configuration.

		$ docker ps
		CONTAINER ID        IMAGE                       COMMAND                CREATED             STATUS              PORTS                                     NAMES
		a81ad2557702        compose_nginx:latest        "nginx -g 'daemon of   8 minutes ago       Up 8 minutes        80/tcp, 443/tcp, 0.0.0.0:5000->5000/tcp   compose_nginx_1        
		0618437450dd        compose_registryv2:latest   "registry cmd/regist   8 minutes ago       Up 8 minutes        0.0.0.0:32777->5000/tcp                   compose_registryv2_1   
		aa82b1ed8e61        registry:latest             "docker-registry"      8 minutes ago       Up 8 minutes        0.0.0.0:32776->5000/tcp                   compose_registryv1_1   
	
### Explore a bit

1. Check for TLS on your `nginx` server.

		$ curl -v https://localhost:5000
		* Rebuilt URL to: https://localhost:5000/
		* Hostname was NOT found in DNS cache
		*   Trying 127.0.0.1...
		* Connected to localhost (127.0.0.1) port 5000 (#0)
		* successfully set certificate verify locations:
		*   CAfile: none
			CApath: /etc/ssl/certs
		* SSLv3, TLS handshake, Client hello (1):
		* SSLv3, TLS handshake, Server hello (2):
		* SSLv3, TLS handshake, CERT (11):
		* SSLv3, TLS alert, Server hello (2):
		* SSL certificate problem: self signed certificate
		* Closing connection 0
		curl: (60) SSL certificate problem: self signed certificate
		More details here: http://curl.haxx.se/docs/sslcerts.html
		
2. Tag the `v1` registry image.

		 $ docker tag registry:latest localhost:5000/registry_one:latest

2. Push it to the localhost.

		 $ docker push localhost:5000/registry_one:latest
		
	If you are using the 1.6 Docker client, this pushes the image the `v2 `registry.

4. Use `curl` to list the image in the registry.

			$ curl -v -X GET http://localhost:5000/v2/registry_one/tags/list
			* Hostname was NOT found in DNS cache
			*   Trying 127.0.0.1...
			* Connected to localhost (127.0.0.1) port 32777 (#0)
			> GET /v2/registry1/tags/list HTTP/1.1
			> User-Agent: curl/7.36.0
			> Host: localhost:5000
			> Accept: */*
			> 
			< HTTP/1.1 200 OK
			< Content-Type: application/json; charset=utf-8
			< Docker-Distribution-Api-Version: registry/2.0
			< Date: Tue, 14 Apr 2015 22:34:13 GMT
			< Content-Length: 39
			< 
			{"name":"registry_one","tags":["latest"]}
			* Connection #0 to host localhost left intact
		
	This example refers to the specific port assigned to the 2.0 registry. You saw
	this port earlier, when you used `docker ps` to show your running containers.


