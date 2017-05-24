FROM debian:wheezy

EXPOSE 443

#Install Nginx and stop it from running
RUN apt-get update && apt-get install openssl nginx apache2-utils -qq -y && rm -rf /var/lib/apt/lists/*
RUN service nginx stop

VOLUME ["/data"]

# Add our Conf files and disable the default site
ADD nginx.conf /etc/nginx/
ADD kubernetes-site /etc/nginx/sites-enabled/
RUN rm -f /etc/nginx/sites-enabled/default

ENTRYPOINT nginx
CMD []