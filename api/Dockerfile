FROM google/nodejs
RUN npm i -g raml2html
ADD . /data
CMD ["-i", "/data/lmktfyrnetes.raml", "-o", "/data/lmktfyrnetes.html"]
ENTRYPOINT ["raml2html"]
