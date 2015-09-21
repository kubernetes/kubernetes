# Updating Salt debs

We are caching all of the salt debs in GCS for speed and reliability.

To update them, follow this simple N step process:

1. Start up a new base image without salt installed.  SSH into this image.
2. Install salt via their recommended method: `curl -L https://bootstrap.saltstack.com | sudo sh -s -- -M -X`
3. Find and download the debs that originated at the saltstack.com repo: `aptitude search --disable-columns -F "%p %V" "?installed?origin(saltstack.com)" | xargs aptitude download`
4. Upload these to GCS: `gsutil cp *.deb gs://kubernetes-release/salt/`
5. Make sure that everything is publicly readable: `gsutil acl ch -R -g all:R gs://kubernetes-release/salt/`
6. Test things well :)
