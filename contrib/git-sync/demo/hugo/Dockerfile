FROM golang
RUN go get -v github.com/spf13/hugo
RUN git clone --recursive https://github.com/spf13/hugoThemes.git /themes
VOLUME ["/src", "/dest"]
EXPOSE 1313
ENV SRC /src
ENV DEST /dest
ENV THEME hyde
ENV BUILD_DRAFT false
ENV BASE_URL ""
ADD run-hugo /run-hugo
ENTRYPOINT ["/run-hugo"]
CMD ["server", "--source=${SRC}", "--theme=${THEME}", "--buildDrafts=${BUILD_DRAFT}", "--baseUrl=${BASE_URL}", "--watch", "--destination=${DEST}", "--appendPort=false"]
