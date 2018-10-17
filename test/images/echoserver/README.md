# Echoserver

This is a simple server that responds with the http headers it received.

Image Versions >= 1.10 support HTTP2 on :8443.
Image Versions >= 1.9 expose HTTPS endpoint on :8443.
Image versions >= 1.4 removes the redirect introduced in 1.3.  
Image versions >= 1.3 redirect requests on :80 with `X-Forwarded-Proto: http` to :443.  
Image versions > 1.0 run an nginx server, and implement the echoserver using lua in the nginx config.  
Image versions <= 1.0 run a python http server instead of nginx, and don't redirect any requests.
