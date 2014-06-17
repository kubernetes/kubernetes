FROM google/nodejs
RUN npm i -g raml2html
VOLUME ["/data"]
CMD ["-i", "/data/kubernetes.raml", "-o", "/data/kubernetes.html"]
ENTRYPOINT ["raml2html"]
