FROM golang:1.10.0-alpine

# cache buster
RUN echo 4 

# git is needed for "go get" below
RUN apk add --no-cache git make

# these are my standard testing / linting tools
RUN /bin/true \
    && go get -u github.com/golang/dep/cmd/dep \
    && go get -u github.com/alecthomas/gometalinter \
    && gometalinter --install \
    && rm -rf /go/src /go/pkg
#
# * SCOWL word list
#
# Downloads
#  http://wordlist.aspell.net/dicts/
#  --> http://app.aspell.net/create
#

# use en_US large size
# use regular size for others
ENV SOURCE_US_BIG http://app.aspell.net/create?max_size=70&spelling=US&max_variant=2&diacritic=both&special=hacker&special=roman-numerals&download=wordlist&encoding=utf-8&format=inline

# should be able tell difference between English variations using this
ENV SOURCE_US http://app.aspell.net/create?max_size=60&spelling=US&max_variant=1&diacritic=both&download=wordlist&encoding=utf-8&format=inline
ENV SOURCE_GB_ISE http://app.aspell.net/create?max_size=60&spelling=GBs&max_variant=2&diacritic=both&download=wordlist&encoding=utf-8&format=inline
ENV SOURCE_GB_IZE http://app.aspell.net/create?max_size=60&spelling=GBz&max_variant=2&diacritic=both&download=wordlist&encoding=utf-8&format=inline
ENV SOURCE_CA http://app.aspell.net/create?max_size=60&spelling=CA&max_variant=2&diacritic=both&download=wordlist&encoding=utf-8&format=inline

RUN /bin/true \
  && mkdir /scowl-wl \
  && wget -O /scowl-wl/words-US-60.txt ${SOURCE_US} \
  && wget -O /scowl-wl/words-GB-ise-60.txt ${SOURCE_GB_ISE} 

